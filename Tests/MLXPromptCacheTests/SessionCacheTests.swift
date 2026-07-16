import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite
struct SessionCacheTests {
    let sig = Fixture.signature

    private func twoLayerModel() -> StubModel { StubModel { [KVCacheSimple(), KVCacheSimple()] as [KVCache] } }
    private func hybridModel()   -> StubModel { StubModel { [MambaCache(), KVCacheSimple()] as [KVCache] } }

    private func makeStore() throws -> (URL, PromptCacheStore, PromptCacheCoordinator) {
        let dir = Fixture.tempDir()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 30, signature: sig)
        return (dir, store, PromptCacheCoordinator(store: store))   // default blockSize 256
    }

    /// One turn against a StubModel: take the delta, prefill it, then append `answerLen` "generated"
    /// tokens. StubModel advances each KVCacheSimple by the input length, so the held cache grows by
    /// exactly (delta + answer). Returns the delta token count.
    @discardableResult
    private func runTurn(_ session: SessionCache, model: StubModel, fullPrompt: [Int], answerLen: Int) -> Int {
        let delta = session.advance(fullPromptTokens: fullPrompt)
        _ = model.callAsFunction(delta.text, cache: session.cache, state: nil)          // prefill the delta
        if answerLen > 0 {
            _ = model.callAsFunction(LMInput(tokens: MLXArray(Fixture.tokens(answerLen))).text,
                                     cache: session.cache, state: nil)                   // append the answer
        }
        return delta.text.tokens.shape.last ?? 0
    }

    // 1 — the warm root seeds the session; turn 1's history is resident, not re-prefilled.
    @Test func seedFromWarmRootResidentNotReprefilled() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)                                                   // boundary 512
        guard case .complete = coord.warm(promptTokens: root, model: twoLayerModel(),
                                          parameters: GenerateParameters())
        else { Issue.record("warm did not complete"); return }
        let session = SessionCache(warmRoot: store.reuse(forTokens: root),
                                   makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect(PromptCacheIO.tokenLength(session.cache) == 512)
    }

    // 2 — advance returns only the new turn's tokens.
    @Test func advanceReturnsOnlyDelta() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: twoLayerModel(), parameters: GenerateParameters())
        let session = SessionCache(warmRoot: store.reuse(forTokens: root),
                                   makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        let delta = session.advance(fullPromptTokens: Fixture.tokens(512 + 20))          // 512 resident + a 20-tok question
        #expect((delta.text.tokens.shape.last ?? 0) == 20)
    }

    // 3 — THE CRUX of approach B: the cache is HELD and extended, never reloaded.
    @Test func growsAcrossTurnsHeldNotReloaded() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: twoLayerModel(), parameters: GenerateParameters())
        let model = twoLayerModel()
        let session = SessionCache(warmRoot: store.reuse(forTokens: root),
                                   makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        let attnId = ObjectIdentifier(session.cache[1] as AnyObject)                     // the held attention layer
        var resident = 512
        for (qLen, aLen) in [(20, 10), (15, 8), (30, 12)] {
            let d = runTurn(session, model: model, fullPrompt: Fixture.tokens(resident + qLen), answerLen: aLen)
            #expect(d == qLen)                                                           // only the new turn prefilled
            resident += qLen + aLen
            #expect(PromptCacheIO.tokenLength(session.cache) == resident)                // grew by turn + answer
        }
        #expect(ObjectIdentifier(session.cache[1] as AnyObject) == attnId)              // SAME object — never store.reuse'd again
    }

    // 4 — hybrid seed: the recurrent (Mamba) layer is held live across turns, never serialised away.
    @Test func hybridSeedRidesRecurrentLive() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: hybridModel(), parameters: GenerateParameters())
        let model = hybridModel()
        let session = SessionCache(warmRoot: store.reuse(forTokens: root),
                                   makeCache: { [MambaCache(), KVCacheSimple()] as [KVCache] })
        #expect(session.cache.first is MambaCache)
        #expect(PromptCacheIO.tokenLength(session.cache) == 512)
        var resident = 512
        for (qLen, aLen) in [(20, 10), (15, 8)] {
            #expect(runTurn(session, model: model, fullPrompt: Fixture.tokens(resident + qLen), answerLen: aLen) == qLen)
            resident += qLen + aLen
        }
        #expect(session.cache.first is MambaCache)                                       // still held live
    }

    // 5 — no warm root: the first advance is the whole prompt.
    @Test func emptySeedFirstAdvanceIsFullPrompt() {
        let session = SessionCache(warmRoot: nil, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect((session.advance(fullPromptTokens: Fixture.tokens(30)).text.tokens.shape.last ?? 0) == 30)
    }

    // 6 — a prompt shorter than the resident cache clamps rather than underflowing.
    @Test func divergedPrefixClamps() {
        let model = twoLayerModel()
        let session = SessionCache(warmRoot: nil, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        _ = model.callAsFunction(LMInput(tokens: MLXArray(Fixture.tokens(40))).text, cache: session.cache, state: nil)
        #expect((session.advance(fullPromptTokens: Fixture.tokens(25)).text.tokens.shape.last ?? 0) == 0)
    }

    // 7 — release frees the cache.
    @Test func releaseEmptiesCache() {
        let session = SessionCache(warmRoot: nil, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        session.release()
        #expect(session.cache.isEmpty)
    }
}
