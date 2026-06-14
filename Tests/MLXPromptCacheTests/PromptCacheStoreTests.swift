import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite struct PromptCacheStoreTests {
    func makeStore(_ dir: URL, signature: CacheSignature = Fixture.signature, budget: Int = 1_000_000_000) throws -> PromptCacheStore {
        try PromptCacheStore(directory: dir, budgetBytes: budget, signature: signature, blockSize: 4)
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
}
