import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite
struct SessionStoreTests {
    let sig = Fixture.signature

    private func twoLayerModel() -> StubModel { StubModel { [KVCacheSimple(), KVCacheSimple()] as [KVCache] } }
    private func hybridModel()   -> StubModel { StubModel { [MambaCache(), KVCacheSimple()] as [KVCache] } }

    private func makeStore() throws -> (URL, PromptCacheStore, PromptCacheCoordinator) {
        let dir = Fixture.tempDir()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 30, signature: sig)
        return (dir, store, PromptCacheCoordinator(store: store))   // default blockSize 256
    }

    /// One turn: advance (seeds on the first call for `id`), prefill the delta, append `answerLen`
    /// "generated" tokens. StubModel advances each KVCacheSimple by the input length, so the held cache
    /// grows by exactly (delta + answer). Returns (deltaLen, the live cache).
    @discardableResult
    private func runTurn(_ sessions: SessionStore, id: UUID, model: StubModel,
                         fullPrompt: [Int], answerLen: Int,
                         warmRoot: () -> Reused?, makeCache: () -> [KVCache]) -> (Int, [KVCache]) {
        let (delta, cache) = sessions.advance(id: id, fullPromptTokens: fullPrompt,
                                              warmRoot: warmRoot, makeCache: makeCache)
        _ = model.callAsFunction(delta.text, cache: cache, state: nil)                    // prefill the delta
        if answerLen > 0 {
            _ = model.callAsFunction(LMInput(tokens: MLXArray(Fixture.tokens(answerLen))).text,
                                     cache: cache, state: nil)                             // append the answer
        }
        return (delta.text.tokens.shape.last ?? 0, cache)
    }

    // 1 — warm root seeds the session; turn 1's history is resident, not re-prefilled.
    @Test func seedFromWarmRootResidentNotReprefilled() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)                                                     // boundary 512
        guard case .complete = coord.warm(promptTokens: root, model: twoLayerModel(),
                                          parameters: GenerateParameters())
        else { Issue.record("warm did not complete"); return }
        let sessions = SessionStore()
        let (_, cache) = sessions.advance(id: UUID(), fullPromptTokens: root,
            warmRoot: { store.reuse(forTokens: root) },
            makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect(PromptCacheIO.tokenLength(cache) == 512)
    }

    // 2 — advance returns only the new turn's tokens.
    @Test func advanceReturnsOnlyDelta() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: twoLayerModel(), parameters: GenerateParameters())
        let sessions = SessionStore()
        let (delta, _) = sessions.advance(id: UUID(), fullPromptTokens: Fixture.tokens(512 + 20),
            warmRoot: { store.reuse(forTokens: root) },
            makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect((delta.text.tokens.shape.last ?? 0) == 20)
    }

    // 3 — THE CRUX: the cache is HELD and extended, never reloaded, and store.reuse fires exactly once.
    @Test func growsAcrossTurnsHeldNotReloaded() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: twoLayerModel(), parameters: GenerateParameters())
        let model = twoLayerModel()
        let sessions = SessionStore()
        let id = UUID()
        var reuseCalls = 0
        let warm: () -> Reused? = { reuseCalls += 1; return store.reuse(forTokens: root) }
        let make: () -> [KVCache] = { [KVCacheSimple(), KVCacheSimple()] as [KVCache] }

        var resident = 512
        var attnId: ObjectIdentifier?
        for (qLen, aLen) in [(20, 10), (15, 8), (30, 12)] {
            let (d, cache) = runTurn(sessions, id: id, model: model,
                                     fullPrompt: Fixture.tokens(resident + qLen), answerLen: aLen,
                                     warmRoot: warm, makeCache: make)
            if attnId == nil { attnId = ObjectIdentifier(cache[1] as AnyObject) }
            #expect(d == qLen)                                                            // only the new turn prefilled
            resident += qLen + aLen
            #expect(PromptCacheIO.tokenLength(cache) == resident)                         // grew by turn + answer
            #expect(ObjectIdentifier(cache[1] as AnyObject) == attnId)                    // SAME held object every turn
        }
        #expect(reuseCalls == 1)                                                          // seeded once, never re-reused
    }

    // 4 — hybrid seed: the recurrent (Mamba) layer is held live across turns, never serialised away.
    @Test func hybridSeedRidesRecurrentLive() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: hybridModel(), parameters: GenerateParameters())
        let model = hybridModel()
        let sessions = SessionStore()
        let id = UUID()
        let warm: () -> Reused? = { store.reuse(forTokens: root) }
        let make: () -> [KVCache] = { [MambaCache(), KVCacheSimple()] as [KVCache] }

        var resident = 512
        var last: [KVCache] = []
        for (qLen, aLen) in [(20, 10), (15, 8)] {
            let (d, cache) = runTurn(sessions, id: id, model: model,
                                     fullPrompt: Fixture.tokens(resident + qLen), answerLen: aLen,
                                     warmRoot: warm, makeCache: make)
            #expect(d == qLen)
            #expect(cache.first is MambaCache)                                            // recurrent layer held live
            resident += qLen + aLen
            #expect(PromptCacheIO.tokenLength(cache) == resident)
            last = cache
        }
        #expect(last.first is MambaCache)
    }

    // 5 — no warm root: the first advance is the whole prompt.
    @Test func emptySeedFirstAdvanceIsFullPrompt() {
        let sessions = SessionStore()
        let (delta, _) = sessions.advance(id: UUID(), fullPromptTokens: Fixture.tokens(30),
            warmRoot: { nil }, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect((delta.text.tokens.shape.last ?? 0) == 30)
    }

    // 6 — a prompt shorter than the resident cache clamps rather than underflowing.
    @Test func divergedPrefixClamps() {
        let model = twoLayerModel()
        let sessions = SessionStore()
        let id = UUID()
        let (_, cache) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(40),
            warmRoot: { nil }, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        _ = model.callAsFunction(LMInput(tokens: MLXArray(Fixture.tokens(40))).text, cache: cache, state: nil)
        let (delta, _) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(25),
            warmRoot: { nil }, makeCache: { [] as [KVCache] })                            // not re-seeded (already live)
        #expect((delta.text.tokens.shape.last ?? 0) == 0)
    }

    // 7 — release frees the cache (and is idempotent): a later advance for the same id RE-SEEDS.
    @Test func releaseFreesCache() {
        let sessions = SessionStore()
        let id = UUID()
        let (_, cache) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(30),
            warmRoot: { nil }, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect(!cache.isEmpty)
        sessions.release(id)
        var reseeded = false
        let (_, cache2) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(10),
            warmRoot: { nil }, makeCache: { reseeded = true; return [KVCacheSimple()] as [KVCache] })
        #expect(reseeded)                                                                 // entry was dropped → makeCache ran
        #expect(PromptCacheIO.tokenLength(cache2) == 0)                                    // brand-new empty cache
        sessions.release(id); sessions.release(id)                                        // idempotent — no crash
    }
}
