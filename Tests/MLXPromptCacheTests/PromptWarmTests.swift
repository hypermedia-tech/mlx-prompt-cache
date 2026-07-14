import Foundation
import MLXLMCommon
import Testing
@testable import MLXPromptCache

/// K2a: `PromptCacheCoordinator.warm` — peek-first idempotence, resume-from-partial, chunked prefill
/// with a pause probe, block-aligned partial record. Headless (StubModel), mirroring
/// `PromptCacheCoordinatorTests`. The byte-identical paused-vs-uninterrupted proof needs a real model
/// and lives in the `MLXPromptCacheScratch` harness (GPU, manual).
@Suite struct PromptWarmTests {

    /// Two attention layers. StubModel advances each `KVCacheSimple` by the input length, so a prefill
    /// from `start` to `boundary` lands the cache at exactly `boundary`.
    private func twoLayerModel() -> StubModel { StubModel { [KVCacheSimple(), KVCacheSimple()] as [KVCache] } }

    private func makeCoord() throws -> (dir: URL, store: PromptCacheStore, coord: PromptCacheCoordinator) {
        let dir = Fixture.tempDir()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 30, signature: Fixture.signature)
        return (dir, store, PromptCacheCoordinator(store: store))   // default blockSize 256
    }

    /// blockSize 256 + step 256 ⇒ one 256-token block per chunk, so a 512-boundary prefill is exactly two
    /// chunks — a `shouldPause` that fires pauses after the first (the default step 512 would prefill 512 in
    /// a single chunk that is never eligible to pause).
    private func blockStepParams() -> GenerateParameters {
        var p = GenerateParameters()
        p.prefillStepSize = 256
        return p
    }

    private func deleteSnapshots(in dir: URL) throws {
        for f in try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
        where f.pathExtension == "safetensors" { try FileManager.default.removeItem(at: f) }
    }

    // MARK: - Cold capture

    @Test func coldWarmCapturesToBoundary() throws {
        let (_, store, coord) = try makeCoord()
        let tokens = Fixture.tokens(600)                    // boundary 512 at blockSize 256
        let out = coord.warm(promptTokens: tokens, model: twoLayerModel(), parameters: GenerateParameters())
        guard case let .complete(cached, prefilled) = out else {
            Issue.record("expected .complete, got \(out)"); return
        }
        #expect(cached == 512)
        #expect(prefilled == 512)                           // nothing reused — all prefilled cold
        #expect(store.peek(forTokens: tokens) == 512)       // the block-aligned prefix is on disk
    }

    // MARK: - Idempotence (the peek-only no-op — no snapshot IO, no GPU)

    @Test func repeatWarmIsPeekOnlyNoSnapshotIO() throws {
        let (dir, _, coord) = try makeCoord()
        let tokens = Fixture.tokens(600)
        _ = coord.warm(promptTokens: tokens, model: twoLayerModel(), parameters: GenerateParameters())
        // Delete the on-disk snapshot: a catalog-probe fast path must NOT need it (reuse would).
        try deleteSnapshots(in: dir)
        let again = coord.warm(promptTokens: tokens, model: twoLayerModel(), parameters: GenerateParameters())
        guard case let .complete(cached, prefilled) = again else {
            Issue.record("expected .complete, got \(again)"); return
        }
        #expect(cached == 512)
        #expect(prefilled == 0)                             // peek full-hit: no reuse load, no prefill
    }

    // MARK: - Below one block

    @Test func warmUnderOneBlockIsUncacheable() throws {
        let (_, _, coord) = try makeCoord()
        let out = coord.warm(promptTokens: Fixture.tokens(100), model: twoLayerModel(),
                             parameters: GenerateParameters())
        guard case .uncacheable = out else { Issue.record("expected .uncacheable, got \(out)"); return }
    }

    // MARK: - Pause → record partial → resume from partial

    @Test func pausesMidPrefillThenResumes() throws {
        let (_, store, coord) = try makeCoord()
        let tokens = Fixture.tokens(600)                    // boundary 512
        let params = blockStepParams()                      // 256-token chunks ⇒ two chunks to the boundary
        // Pause at the first between-chunk checkpoint: the first block lands, the second is deferred.
        let first = coord.warm(promptTokens: tokens, model: twoLayerModel(), parameters: params,
                               shouldPause: { true })
        guard case let .paused(cached) = first else { Issue.record("expected .paused, got \(first)"); return }
        #expect(cached == 256)                              // one block prefilled…
        #expect(store.peek(forTokens: tokens) == 256)       // …and recorded (progress persists on disk)
        // Resume: the next warm reuses the recorded block and prefills only the remainder.
        let second = coord.warm(promptTokens: tokens, model: twoLayerModel(), parameters: params)
        guard case let .complete(total, prefilled) = second else {
            Issue.record("expected .complete, got \(second)"); return
        }
        #expect(total == 512)
        #expect(prefilled == 256)                           // only the second block — the first was reused
        #expect(store.peek(forTokens: tokens) == 512)
    }

    // MARK: - The disk IS the resume token (survives a store reopen)

    @Test func pausedProgressSurvivesReopen() throws {
        let dir = Fixture.tempDir()
        let tokens = Fixture.tokens(600)
        let params = blockStepParams()
        do {
            let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 30, signature: Fixture.signature)
            let out = PromptCacheCoordinator(store: store)
                .warm(promptTokens: tokens, model: twoLayerModel(), parameters: params, shouldPause: { true })
            guard case .paused = out else { Issue.record("expected .paused, got \(out)"); return }
        }
        // A NEW store instance over the same directory — no in-memory resumption state exists anywhere.
        let reopened = try PromptCacheStore(directory: dir, budgetBytes: 1 << 30, signature: Fixture.signature)
        #expect(reopened.peek(forTokens: tokens) == 256)
        let resumed = PromptCacheCoordinator(store: reopened)
            .warm(promptTokens: tokens, model: twoLayerModel(), parameters: params)
        guard case let .complete(total, prefilled) = resumed else {
            Issue.record("expected .complete, got \(resumed)"); return
        }
        #expect(total == 512)
        #expect(prefilled == 256)
    }

    // MARK: - Longest-prefix reuse across different-length prompts

    @Test func longerWarmReusesShorterWarmsBlocks() throws {
        let (_, store, coord) = try makeCoord()
        let short = Fixture.tokens(600)                     // boundary 512
        let long  = Fixture.tokens(2000)                    // shares its first 600 tokens; boundary 1792
        _ = coord.warm(promptTokens: short, model: twoLayerModel(), parameters: GenerateParameters())
        #expect(store.peek(forTokens: short) == 512)
        let out = coord.warm(promptTokens: long, model: twoLayerModel(), parameters: GenerateParameters())
        guard case let .complete(total, prefilled) = out else {
            Issue.record("expected .complete, got \(out)"); return
        }
        let boundary = (long.count - 1) / store.blockSize * store.blockSize   // 1792
        #expect(total == boundary)
        #expect(prefilled == boundary - 512)                // reused the 512 blocks the short warm recorded
    }
}
