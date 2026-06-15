import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite struct PromptCacheIOHybridTests {
    let sig = Fixture.signature

    @Test func classifiesSliceableVsHybrid() {
        #expect(PromptCacheIO.isSliceable(Fixture.syntheticCache(tokens: 8)))   // all attention
        #expect(!PromptCacheIO.isSliceable(Fixture.hybridCache(slice: 8)))      // one MambaCache layer
        #expect(!PromptCacheIO.isSliceable(Fixture.pureSSMCache()))
    }

    @Test func tokenLengthReadsSliceableLayerNotFirst() {
        // The crash scenario: first layer is MambaCache (offset 0); tokenLength must skip it.
        #expect(PromptCacheIO.tokenLength(Fixture.hybridCache(slice: 8)) == 8)
        #expect(PromptCacheIO.tokenLength(Fixture.pureSSMCache()) == nil)
    }

    @Test func saveRejectsHybridOffBoundary() {
        let url = Fixture.tempDir().appendingPathComponent("snap.safetensors")
        #expect(throws: PromptCacheIO.Failure.hybridNotAtBoundary) {
            try PromptCacheIO.save(prefixTokenCount: 4, liveCache: Fixture.hybridCache(slice: 8),
                                   url: url, signature: sig)
        }
    }

    @Test func saveRejectsPureSSM() {
        let url = Fixture.tempDir().appendingPathComponent("snap.safetensors")
        #expect(throws: PromptCacheIO.Failure.noSliceableLayer) {
            try PromptCacheIO.save(prefixTokenCount: 8, liveCache: Fixture.pureSSMCache(),
                                   url: url, signature: sig)
        }
    }

    @Test func trimRejectsHybridSubPrefixAcceptsExact() {
        #expect(PromptCacheIO.trim(Fixture.hybridCache(slice: 8), toMatched: 4) == nil)       // can't slice a hybrid
        #expect(PromptCacheIO.trim(Fixture.hybridCache(slice: 8), toMatched: 8)?.count == 2)  // exact → whole, no trim
    }

    @Test func savesAndReloadsHybridAtBoundary() throws {
        let url = Fixture.tempDir().appendingPathComponent("snap.safetensors")
        try PromptCacheIO.save(prefixTokenCount: 8, liveCache: Fixture.hybridCache(slice: 8),
                               url: url, signature: sig)                        // 8 == 8 → no throw
        let loaded = PromptCacheIO.loadFull(url: url, signature: sig)
        #expect(loaded?.count == 2)
        #expect(PromptCacheIO.tokenLength(loaded ?? []) == 8)
    }
}
