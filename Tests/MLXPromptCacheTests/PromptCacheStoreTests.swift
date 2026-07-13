import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite struct PromptCacheStoreTests {
    func makeStore(_ dir: URL, signature: CacheSignature = Fixture.signature, budget: Int = 1_000_000_000) throws -> PromptCacheStore {
        try PromptCacheStore(directory: dir, budgetBytes: budget, signature: signature, blockSize: 4)
    }
    
    func makeHotStore(_ dir: URL, hot: Int = 1_000_000_000) throws -> PromptCacheStore {
        try PromptCacheStore(directory: dir, budgetBytes: 1_000_000_000, signature: Fixture.signature, blockSize: 4, hotBudgetBytes: hot)
    }
    
    private func deleteSnapshots(in dir: URL) throws {
        for f in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" { try FileManager.default.removeItem(at: f) }
    }
    
    @Test func recordThenReuseHits() throws {
        let store = try makeStore(Fixture.tempDir())
        let toks = Fixture.tokens(8)
        try store.record(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))
        let reused = store.reuse(forTokens: toks)
        #expect(reused?.matchedTokens == 8)
        #expect(reused?.cache.first?.offset == 8)
    }
    
    @Test func emptyStoreMisses() throws {
        let store = try makeStore(Fixture.tempDir())
        #expect(store.reuse(forTokens: Fixture.tokens(8)) == nil)
    }
    
    @Test func unrelatedPromptMisses() throws {
        let store = try makeStore(Fixture.tempDir())
        try store.record(prefixTokens: Fixture.tokens(8, seed: 0), cache: Fixture.syntheticCache(tokens: 8))
        #expect(store.reuse(forTokens: Fixture.tokens(8, seed: 100)) == nil)
    }
    
    @Test func crossPromptPartialMatch() throws {
        let store = try makeStore(Fixture.tempDir())
        try store.record(prefixTokens: [1, 2, 3, 4, 5, 6, 7, 8], cache: Fixture.syntheticCache(tokens: 8))
        let reused = store.reuse(forTokens: [1, 2, 3, 4, 9, 9, 9, 9])
        #expect(reused?.matchedTokens == 4)
        #expect(reused?.cache.first?.offset == 4)
    }
    
    @Test func survivesRestart() throws {
        let dir = Fixture.tempDir()
        let toks = Fixture.tokens(8)
        do {
            let store = try makeStore(dir)
            try store.record(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))
        }
        let reopened = try makeStore(dir)
        #expect(reopened.reuse(forTokens: toks)?.matchedTokens == 8)
    }
    
    @Test func signatureChangeWipesOnRestart() throws {
        let dir = Fixture.tempDir()
        let toks = Fixture.tokens(8)
        try makeStore(dir).record(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))
        let other = CacheSignature(modelId: "different", kvDType: "bf16", kvBits: nil, buildVersion: "t1")
        let reopened = try makeStore(dir, signature: other)
        #expect(reopened.reuse(forTokens: toks) == nil)
    }
    
    @Test func budgetEvictsAndDeletes() throws {
        let store = try makeStore(Fixture.tempDir(), budget: 900_000)     // ≈ holds one ~512 KB snapshot, not two
        func big() -> [KVCache] { Fixture.syntheticCache(tokens: 8, layers: 8, kvHeads: 16, headDim: 128) }
        try store.record(prefixTokens: Fixture.tokens(8, seed: 0), cache: big())
        try store.record(prefixTokens: Fixture.tokens(8, seed: 100), cache: big())   // pushes over budget
        #expect(store.reuse(forTokens: Fixture.tokens(8, seed: 0)) == nil)           // oldest evicted
        #expect(store.reuse(forTokens: Fixture.tokens(8, seed: 100)) != nil)         // newest kept
    }

    @Test func recordIsIdempotent() throws {
        let store = try makeStore(Fixture.tempDir())
        let toks = Fixture.tokens(8)
        try store.record(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))
        try store.record(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))   // no-op, already cached
        #expect(store.reuse(forTokens: toks)?.matchedTokens == 8)
    }
    
    @Test func hotHitServesAfterDiskDeleted() throws {
        let dir = Fixture.tempDir(); let store = try makeHotStore(dir); let toks = Fixture.tokens(8)
        try store.record(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))
        _ = store.reuse(forTokens: toks)              // cold hit → promotes bytes into RAM
        try deleteSnapshots(in: dir)                   // now a hit can ONLY come from RAM
        let hot = store.reuse(forTokens: toks)
        #expect(hot?.matchedTokens == 8)
        #expect(hot?.cache.first?.offset == 8)
    }
    
    @Test func preloadWarmsRAMImmediately() throws {
        let dir = Fixture.tempDir(); let store = try makeHotStore(dir); let toks = Fixture.tokens(8)
        try store.preload(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))
        try deleteSnapshots(in: dir)
        #expect(store.reuse(forTokens: toks)?.matchedTokens == 8)
    }
    
    @Test func clearHotDropsRAM() throws {
        let dir = Fixture.tempDir(); let store = try makeHotStore(dir); let toks = Fixture.tokens(8)
        try store.preload(prefixTokens: toks, cache: Fixture.syntheticCache(tokens: 8))
        store.clearHot(); try deleteSnapshots(in: dir)
        #expect(store.reuse(forTokens: toks) == nil) // RAM cleared + disk gone ⇒ miss
    }
    
    @Test func partialMatchIsNotPromotedToHot() throws {
        let dir = Fixture.tempDir(); let store = try makeHotStore(dir)
        try store.record(prefixTokens: [1,2,3,4,5,6,7,8], cache: Fixture.syntheticCache(tokens: 8))
        _ = store.reuse(forTokens: [1,2,3,4,9,9,9,9]) // partial (matched 4) → must NOT hot-cache
        try deleteSnapshots(in: dir)
        #expect(store.reuse(forTokens: [1,2,3,4,9,9,9,9]) == nil) // proves it wasn't in RAM
    }

    // MARK: - peek (K0 idempotence probe)

    @Test func peekOnEmptyStoreIsZero() throws {
        let store = try makeStore(Fixture.tempDir())
        #expect(store.peek(forTokens: Fixture.tokens(8)) == 0)
    }

    @Test func peekReportsRecordedAlignedLength() throws {
        let store = try makeStore(Fixture.tempDir())                    // blockSize 4
        try store.record(prefixTokens: Fixture.tokens(10), cache: Fixture.syntheticCache(tokens: 10))
        // 10 tokens at blockSize 4 → 2 full blocks stored → the probe reports 8, not 10.
        #expect(store.peek(forTokens: Fixture.tokens(10)) == 8)
    }

    @Test func peekAgreesWithReuseOnPartialMatch() throws {
        let store = try makeStore(Fixture.tempDir())
        try store.record(prefixTokens: [1, 2, 3, 4, 5, 6, 7, 8], cache: Fixture.syntheticCache(tokens: 8))
        let query = [1, 2, 3, 4, 9, 9, 9, 9]
        #expect(store.peek(forTokens: query) == 4)
        #expect(store.peek(forTokens: query) == store.reuse(forTokens: query)?.matchedTokens)
    }

    @Test func peekIsZeroAfterSignatureReset() throws {
        let dir = Fixture.tempDir()
        do {
            let store = try makeStore(dir)
            try store.record(prefixTokens: Fixture.tokens(8), cache: Fixture.syntheticCache(tokens: 8))
        }
        let other = CacheSignature(modelId: "other-model", kvDType: "bf16", kvBits: nil, buildVersion: "t1")
        let reopened = try makeStore(dir, signature: other)             // loadOrReset wipes
        #expect(reopened.peek(forTokens: Fixture.tokens(8)) == 0)
    }

    /// Integration guard that `peek` is wired to `Catalog.probe`, not `lookup`: probing an entry many
    /// times must NOT freshen its LRU stamp, so an over-budget record still evicts the true oldest.
    /// If `peek` regresses to the mutating `lookup`, the probed entry (A) would look hottest and B
    /// would be evicted instead — flipping both expectations below.
    @Test func peekDoesNotFreshenLRU() throws {
        let store = try makeStore(Fixture.tempDir(), budget: 1_200_000)  // holds two ~512 KB snapshots, not three
        func big(_ seed: Int) -> [KVCache] { Fixture.syntheticCache(tokens: 8, layers: 8, kvHeads: 16, headDim: 128) }
        try store.record(prefixTokens: Fixture.tokens(8, seed: 0),   cache: big(0))   // A: oldest
        try store.record(prefixTokens: Fixture.tokens(8, seed: 100), cache: big(100)) // B: newer
        for _ in 0..<5 { _ = store.peek(forTokens: Fixture.tokens(8, seed: 0)) }       // probe A hard
        try store.record(prefixTokens: Fixture.tokens(8, seed: 200), cache: big(200)) // C: pushes over budget
        #expect(store.reuse(forTokens: Fixture.tokens(8, seed: 0)) == nil)             // A still evicted (LRU)
        #expect(store.reuse(forTokens: Fixture.tokens(8, seed: 100)) != nil)           // B survives
    }

    @Test func canonicalRenderingIsStable() {
        // The string is a durable provenance value in CyberBench (APRV-005) — format-pinned.
        #expect(Fixture.signature.canonical == "test-model|bf16|-|t1")
        #expect(CacheSignature(modelId: "m", kvDType: "float16", kvBits: 8, buildVersion: "2").canonical
                == "m|float16|8|2")
    }
}
