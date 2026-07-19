import Foundation
import MLXLMCommon
import Synchronization
import Testing
@testable import MLXPromptCache

/// Residency: `warm(_ warms:id:…)` holds the live cache across pauses so a resume neither reloads
/// the prefix nor rewrites it. Headless (StubModel) — the byte-level proof against real models lives
/// in `MLXPromptCacheBench`.
@Suite struct WarmStoreTests {

    private func twoLayerModel() -> StubModel {
        StubModel { [KVCacheSimple(), KVCacheSimple()] as [KVCache] }
    }

    private func hybridModel() -> StubModel {
        StubModel { [MambaCache(), KVCacheSimple()] as [KVCache] }
    }

    /// blockSize 256 + step 256 ⇒ one block per chunk, so `shouldPause: { true }` advances exactly
    /// one block per call.
    private func blockStepParams() -> GenerateParameters {
        var p = GenerateParameters()
        p.prefillStepSize = 256
        return p
    }

    /// A store whose diagnostic sink we can interrogate — `reuse: loaded` is emitted once per
    /// snapshot read (PromptCacheStore.swift:75), which is exactly what residency must eliminate.
    private func makeStore(_ dir: URL, sink: @escaping @Sendable (String) -> Void = { _ in })
        throws -> PromptCacheStore {
        try PromptCacheStore(directory: dir, budgetBytes: 1 << 30,
                             signature: Fixture.signature, log: sink)
    }

    // MARK: - The core claim

    @Test func resumeFromResidentCacheReadsNothingFromDisk() throws {
        let dir = Fixture.tempDir()
        let lines = Mutex<[String]>([])
        let store = try makeStore(dir) { l in lines.withLock { $0.append(l) } }
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(2000)                       // boundary 1792
        let params = blockStepParams()

        var reached = 0
        for _ in 0 ..< 3 {
            let out = coord.warm(warms, id: id, promptTokens: tokens, model: twoLayerModel(),
                                 parameters: params, shouldPause: { true })
            guard case let .paused(c) = out else { Issue.record("expected .paused, got \(out)"); return }
            #expect(c == reached + 256)                         // one block per pause
            reached = c
        }
        #expect(reached == 768)

        let loads = lines.withLock { $0.filter { $0.hasPrefix("reuse: loaded") }.count }
        let saves = lines.withLock { $0.filter { $0.hasPrefix("record: saved") }.count }
        #expect(loads == 0)                                     // nothing reloaded…
        #expect(saves == 0)                                     // …and nothing written on a pause
        #expect(warms.residentBytes > 0)
    }

    @Test func heldCacheIsTheSameObjectAcrossResumes() throws {
        let dir = Fixture.tempDir()
        let coord = PromptCacheCoordinator(store: try makeStore(dir))
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(2000)
        let model = twoLayerModel()

        _ = coord.warm(warms, id: id, promptTokens: tokens, model: model,
                       parameters: blockStepParams(), shouldPause: { true })
        let first = coord.heldCache(warms, id: id, model: model)
        _ = coord.warm(warms, id: id, promptTokens: tokens, model: model,
                       parameters: blockStepParams(), shouldPause: { true })
        let second = coord.heldCache(warms, id: id, model: model)

        #expect(first?.count == 2)
        // Extended in place, never rebuilt — the same layer objects carry forward.
        #expect(ObjectIdentifier(first![1] as AnyObject) == ObjectIdentifier(second![1] as AnyObject))
        #expect(PromptCacheIO.tokenLength(second ?? []) == 512)
    }

    // MARK: - The divergence guard

    /// The hazard a UUID-keyed holder introduces: resuming the wrong id would extend the wrong cache
    /// and then record it under a chain hash that does not describe its contents, poisoning the
    /// catalog. The frontier hash must make that impossible.
    @Test func divergedPromptIsDeclinedNotExtended() throws {
        let dir = Fixture.tempDir()
        let coord = PromptCacheCoordinator(store: try makeStore(dir))
        let warms = WarmStore()
        let id = UUID()
        let a = Fixture.tokens(2000, seed: 0)
        let b = Fixture.tokens(2000, seed: 7)                   // different content, same length

        _ = coord.warm(warms, id: id, promptTokens: a, model: twoLayerModel(),
                       parameters: blockStepParams(), shouldPause: { true })
        #expect(warms.entry(id)?.prefix == Array(a[0 ..< 256]))

        // Same id, different tokens: must NOT continue a's cache.
        let out = coord.warm(warms, id: id, promptTokens: b, model: twoLayerModel(),
                             parameters: blockStepParams(), shouldPause: { true })
        guard case let .paused(c) = out else { Issue.record("expected .paused, got \(out)"); return }
        #expect(c == 256)                                       // restarted from 0, not 256 → 512
        #expect(warms.entry(id)?.prefix == Array(b[0 ..< 256]))
    }

    // MARK: - Persistence policy

    @Test func completionPersistsOnceAndReleases() throws {
        let dir = Fixture.tempDir()
        let lines = Mutex<[String]>([])
        let store = try makeStore(dir) { l in lines.withLock { $0.append(l) } }
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(600)                        // boundary 512

        let out = coord.warm(warms, id: id, promptTokens: tokens, model: twoLayerModel(),
                             parameters: blockStepParams())     // no pause → straight to boundary
        guard case let .complete(cached, _) = out else {
            Issue.record("expected .complete, got \(out)"); return
        }
        #expect(cached == 512)
        #expect(lines.withLock { $0.filter { $0.hasPrefix("record: saved") }.count } == 1)
        #expect(store.peek(forTokens: tokens) == 512)           // durable
        #expect(warms.isEmpty)                                  // and the RAM is given back
    }

    @Test func neverPersistLeavesDiskUntouched() throws {
        let dir = Fixture.tempDir()
        let store = try makeStore(dir)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(600)

        let out = coord.warm(warms, id: id, promptTokens: tokens, model: twoLayerModel(),
                             parameters: blockStepParams(), persist: .never)
        guard case .complete = out else { Issue.record("expected .complete, got \(out)"); return }
        #expect(snapshotCount(in: dir) == 0)                    // nothing written…
        #expect(store.peek(forTokens: tokens) == 0)             // …and nothing catalogued
    }

    /// Regression: `warm` used to release on `.complete` unconditionally, so `.never` finished the
    /// warm, wrote nothing, and silently discarded every token of it. Completion may only free the
    /// cache once the work is actually on disk.
    @Test func completionUnderNeverKeepsTheCacheResident() throws {
        let dir = Fixture.tempDir()
        let store = try makeStore(dir)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(600)
        let model = twoLayerModel()

        let out = coord.warm(warms, id: id, promptTokens: tokens, model: model,
                             parameters: blockStepParams(), persist: .never)
        guard case .complete = out else { Issue.record("expected .complete, got \(out)"); return }
        #expect(!warms.isEmpty)                                 // the work is still recoverable…
        #expect(PromptCacheIO.tokenLength(coord.heldCache(warms, id: id, model: model) ?? []) == 512)

        _ = coord.finishWarm(warms, id: id, model: model)       // …and finishWarm can still save it
        #expect(store.peek(forTokens: tokens) == 512)
        #expect(warms.isEmpty)
    }

    @Test func finishWarmPersistsAndReleases() throws {
        let dir = Fixture.tempDir()
        let store = try makeStore(dir)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(2000)
        let model = twoLayerModel()

        _ = coord.warm(warms, id: id, promptTokens: tokens, model: model,
                       parameters: blockStepParams(), shouldPause: { true })
        #expect(store.peek(forTokens: tokens) == 0)             // a pause wrote nothing…

        let out = coord.finishWarm(warms, id: id, model: model)
        guard case .complete = out else { Issue.record("expected .complete, got \(out)"); return }
        #expect(store.peek(forTokens: tokens) == 256)           // …abandonment keeps the work
        #expect(warms.isEmpty)
    }

    @Test func everyTokensPersistsOnCadence() throws {
        let dir = Fixture.tempDir()
        let store = try makeStore(dir)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(2000)

        for _ in 0 ..< 2 {
            _ = coord.warm(warms, id: id, promptTokens: tokens, model: twoLayerModel(),
                           parameters: blockStepParams(), persist: .everyTokens(512),
                           shouldPause: { true })
        }
        #expect(store.peek(forTokens: tokens) == 512)           // fired at the 512-token mark
    }

    // MARK: - Degrades losslessly to the disk path

    /// Residency is a strict accelerator. An id that is not resident — fresh process, released,
    /// evicted — must behave exactly as the original `warm` did: resume from disk.
    @Test func nonResidentIdFallsBackToDiskProgress() throws {
        let dir = Fixture.tempDir()
        let tokens = Fixture.tokens(2000)
        let params = blockStepParams()

        // Session one: warm a block and abandon it, so progress exists only on disk.
        do {
            let coord = PromptCacheCoordinator(store: try makeStore(dir))
            let warms = WarmStore()
            let id = UUID()
            _ = coord.warm(warms, id: id, promptTokens: tokens, model: twoLayerModel(),
                           parameters: params, shouldPause: { true })
            _ = coord.finishWarm(warms, id: id, model: twoLayerModel())
        }

        // Session two: brand new store AND new WarmStore — nothing resident anywhere.
        let store = try makeStore(dir)
        let coord = PromptCacheCoordinator(store: store)
        #expect(store.peek(forTokens: tokens) == 256)
        let out = coord.warm(WarmStore(), id: UUID(), promptTokens: tokens,
                             model: twoLayerModel(), parameters: params, shouldPause: { true })
        guard case let .paused(c) = out else { Issue.record("expected .paused, got \(out)"); return }
        #expect(c == 512)                                       // resumed from the disk block
    }

    // MARK: - Hybrid

    @Test func hybridWarmHoldsAcrossPausesAndCompletes() throws {
        let dir = Fixture.tempDir()
        let store = try makeStore(dir)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        let tokens = Fixture.tokens(600)
        let params = blockStepParams()

        let first = coord.warm(warms, id: id, promptTokens: tokens, model: hybridModel(),
                               parameters: params, shouldPause: { true })
        guard case let .paused(c) = first else { Issue.record("expected .paused, got \(first)"); return }
        #expect(c == 256)
        #expect(PromptCacheIO.isSliceable(coord.heldCache(warms, id: id, model: hybridModel()) ?? []) == false)

        let second = coord.warm(warms, id: id, promptTokens: tokens, model: hybridModel(),
                                parameters: params)
        guard case let .complete(total, _) = second else {
            Issue.record("expected .complete, got \(second)"); return
        }
        #expect(total == 512)
        #expect(store.peek(forTokens: tokens) == 512)
        #expect(warms.isEmpty)
    }

    // MARK: - Budget

    @Test func overBudgetVictimIsPersistedThenReleased() throws {
        let dir = Fixture.tempDir()
        let store = try makeStore(dir)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore(budgetBytes: 1)                   // anything resident is over budget
        let a = UUID(), b = UUID()
        let ta = Fixture.tokens(2000, seed: 0)
        let tb = Fixture.tokens(2000, seed: 11)

        _ = coord.warm(warms, id: a, promptTokens: ta, model: twoLayerModel(),
                       parameters: blockStepParams(), shouldPause: { true })
        #expect(warms.heldIds.count == 1)

        // Warming b evicts a — but persists it first, so the work is not lost.
        _ = coord.warm(warms, id: b, promptTokens: tb, model: twoLayerModel(),
                       parameters: blockStepParams(), shouldPause: { true })
        #expect(warms.heldIds == [b])
        #expect(store.peek(forTokens: ta) == 256)               // a's progress survived on disk
    }

    @Test func releaseAllDropsEverything() throws {
        let dir = Fixture.tempDir()
        let coord = PromptCacheCoordinator(store: try makeStore(dir))
        let warms = WarmStore()
        for _ in 0 ..< 3 {
            _ = coord.warm(warms, id: UUID(), promptTokens: Fixture.tokens(2000),
                           model: twoLayerModel(), parameters: blockStepParams(),
                           shouldPause: { true })
        }
        #expect(warms.heldIds.count == 3)
        warms.releaseAll()
        #expect(warms.isEmpty)
        #expect(warms.residentBytes == 0)
    }
}

private func snapshotCount(in dir: URL) -> Int {
    ((try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? [])
        .filter { $0.pathExtension == "safetensors" }.count
}
