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
        #expect(cache.isEmpty == false)
        sessions.release(id)
        var reseeded = false
        let (_, cache2) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(10),
            warmRoot: { nil }, makeCache: { reseeded = true; return [KVCacheSimple()] as [KVCache] })
        #expect(reseeded)                                                                 // entry was dropped → makeCache ran
        #expect(PromptCacheIO.tokenLength(cache2) == 0)                                    // brand-new empty cache
        sessions.release(id); sessions.release(id)                                        // idempotent — no crash
    }

    // 8 — the PUBLIC coordinator seam: `advance` wires the disk root into the seed and `release` frees.
    //     The tests above drive `SessionStore.advance` directly; this exercises the consumer-facing door.
    @Test func coordinatorAdvanceSeedsFromRootAndReleaseFrees() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)                                                     // boundary 512
        guard case .complete = coord.warm(promptTokens: root, model: twoLayerModel(),
                                          parameters: GenerateParameters())
        else { Issue.record("warm did not complete"); return }
        let sessions = SessionStore()
        let id = UUID()
        let scope = PerformScope()

        let (delta, cache) = coord.advance(sessions, id: id,
                                           fullPromptTokens: Fixture.tokens(512 + 20),
                                           rootTokens: root, model: twoLayerModel(),
                                           parameters: GenerateParameters(), scope: scope)
        #expect(PromptCacheIO.tokenLength(cache) == 512)                                   // seeded from the durable root
        #expect((delta.text.tokens.shape.last ?? 0) == 20)                                // only the new turn

        coord.release(sessions, id: id, scope: scope)
        // After release the entry is gone, so the next advance re-seeds from the root rather than resuming.
        var reseeded = false
        let (_, cache2) = sessions.advance(id: id, fullPromptTokens: root,
            warmRoot: { reseeded = true; return store.reuse(forTokens: root) },
            makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect(reseeded)                                                                  // release dropped the live entry
        #expect(PromptCacheIO.tokenLength(cache2) == 512)
    }

    // MARK: - Eviction surface (app-budgeted session eviction, §3.1)

    /// Seed conversation `id` with a specific cache (no prefill, no root) so its footprint is known.
    @discardableResult
    private func seed(_ sessions: SessionStore, _ id: UUID, cache: [KVCache]) -> [KVCache] {
        sessions.advance(id: id, fullPromptTokens: [], warmRoot: { nil }, makeCache: { cache }).cache
    }

    private func snapshotCount(in dir: URL) -> Int {
        ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
            .filter { $0.pathExtension == "safetensors" }.count
    }

    // 9 — residentBytes is the sum of every held cache's live footprint.
    @Test func residentBytesSumsHeldCaches() {
        let sessions = SessionStore()
        let a = Fixture.syntheticCache(tokens: 256)
        let b = Fixture.syntheticCache(tokens: 512)
        seed(sessions, UUID(), cache: a)
        seed(sessions, UUID(), cache: b)
        #expect(sessions.residentBytes == WarmStore.footprint(a) + WarmStore.footprint(b))
    }

    // 10 — under budget (and the disabled 0-budget) evict nothing.
    @Test func victimsEmptyWhenUnderOrDisabled() {
        let sessions = SessionStore()
        seed(sessions, UUID(), cache: Fixture.syntheticCache(tokens: 256))
        let big = sessions.residentBytes * 4
        #expect(sessions.victimsOverBudget(big, excluding: UUID()).isEmpty)     // room to spare
        #expect(sessions.victimsOverBudget(0, excluding: UUID()).isEmpty)       // 0 = unbounded
    }

    // 11 — largest-first, accumulating only until back under budget: the smallest survives.
    @Test func evictsLargestFirstUntilUnderBudget() {
        let sessions = SessionStore()
        let small = UUID(), mid = UUID(), large = UUID()
        let fa = WarmStore.footprint(seed(sessions, small, cache: Fixture.syntheticCache(tokens: 256)))
        seed(sessions, mid, cache: Fixture.syntheticCache(tokens: 512))
        seed(sessions, large, cache: Fixture.syntheticCache(tokens: 1024))
        // Budget leaves room for only the smallest ⇒ drop large then mid, in that order, and stop.
        let victims = sessions.victimsOverBudget(fa, excluding: UUID())
        #expect(victims == [large, mid])                                        // biggest first, smallest spared
    }

    // 12 — `keep` is never a victim, even when it is the largest resident cache.
    @Test func keepIsNeverEvictedEvenIfLargest() {
        let sessions = SessionStore()
        let a = UUID(), b = UUID(), keep = UUID()
        seed(sessions, a, cache: Fixture.syntheticCache(tokens: 256))
        seed(sessions, b, cache: Fixture.syntheticCache(tokens: 512))
        seed(sessions, keep, cache: Fixture.syntheticCache(tokens: 1024))       // biggest — but kept
        let victims = sessions.victimsOverBudget(1, excluding: keep)            // budget 1 ⇒ evict all it can
        #expect(victims.contains(keep) == false)
        #expect(Set(victims) == [a, b])                                         // everything else goes
    }

    // 13 — THE PUBLIC SEAM: evictSessions drops the over-budget victims' RAM and writes NOTHING to disk
    //      (a session's durable source is the log, so eviction never persists — unlike the warm side).
    @Test func evictSessionsDropsRamWithoutPersisting() throws {
        let (dir, _, coord) = try makeStore()
        let sessions = SessionStore()
        let keep = UUID(), victim = UUID()
        let fKeep = WarmStore.footprint(seed(sessions, keep, cache: Fixture.syntheticCache(tokens: 256)))
        seed(sessions, victim, cache: Fixture.syntheticCache(tokens: 1024))     // the big one to shed

        coord.evictSessions(sessions, overBudget: fKeep, keep: keep, scope: PerformScope())

        #expect(sessions.residentBytes == fKeep)                                // only `keep` remains resident
        #expect(snapshotCount(in: dir) == 0)                                    // NO persist-before-release
        // `keep` is still live (advance does not re-seed it)…
        var keepReseeded = false
        _ = sessions.advance(id: keep, fullPromptTokens: [], warmRoot: { nil },
                             makeCache: { keepReseeded = true; return [] as [KVCache] })
        #expect(keepReseeded == false)
        // …while the evicted victim re-seeds on its next advance.
        var victimReseeded = false
        _ = sessions.advance(id: victim, fullPromptTokens: [], warmRoot: { nil },
                             makeCache: { victimReseeded = true; return Fixture.syntheticCache(tokens: 256) })
        #expect(victimReseeded)
    }

    // 14 — evictSessions under budget is a no-op, and it is idempotent (a second call drops nothing more).
    @Test func evictSessionsUnderBudgetIsNoOpAndIdempotent() throws {
        let (_, _, coord) = try makeStore()
        let sessions = SessionStore()
        let keep = UUID(), other = UUID()
        seed(sessions, keep, cache: Fixture.syntheticCache(tokens: 256))
        seed(sessions, other, cache: Fixture.syntheticCache(tokens: 512))
        let before = sessions.residentBytes
        let scope = PerformScope()

        coord.evictSessions(sessions, overBudget: before * 2, keep: keep, scope: scope)   // budget above resident
        #expect(sessions.residentBytes == before)                              // nothing dropped

        coord.evictSessions(sessions, overBudget: 1, keep: keep, scope: scope)            // now shed `other`
        let afterFirstEvict = sessions.residentBytes
        coord.evictSessions(sessions, overBudget: 1, keep: keep, scope: scope)            // idempotent
        #expect(sessions.residentBytes == afterFirstEvict)
    }
}
