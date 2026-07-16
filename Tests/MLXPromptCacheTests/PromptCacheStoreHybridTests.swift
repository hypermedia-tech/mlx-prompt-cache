import Foundation
import Testing
@testable import MLXPromptCache

@Suite struct PromptCacheStoreHybridTests {
    let sig = Fixture.signature
    
    @Test func recordSkipsOffBoundaryHybrid() throws {
        let store = try PromptCacheStore(directory: Fixture.tempDir(), budgetBytes: 1 << 30, signature: sig)
        let prompt = Fixture.tokens(512)
        try store.record(prefixTokens: prompt, cache: Fixture.hybridCache(slice: 300))
        #expect(store.reuse(forTokens: prompt) == nil)
    }
    
    @Test func recordSkipsPureSSM() throws {
        let store = try PromptCacheStore(directory: Fixture.tempDir(), budgetBytes: 1 << 30, signature: sig)
        let prompt = Fixture.tokens(512)
        try store.record(prefixTokens: prompt, cache: Fixture.pureSSMCache())
        #expect(store.reuse(forTokens: prompt) == nil)
    }
    
    @Test func reuseHitsHybridAtAlignedBoundary() throws {
        let store = try PromptCacheStore(directory: Fixture.tempDir(), budgetBytes: 1 << 30, signature: sig)
        let prompt = Fixture.tokens(512)                                        // block-aligned (2 × 256)
        try store.record(prefixTokens: prompt, cache: Fixture.hybridCache(slice: 512))   // captured AT the boundary
        let reused = store.reuse(forTokens: prompt)
        #expect(reused?.matchedTokens == 512)                                   // hybrid prefix HIT from disk
        #expect(PromptCacheIO.tokenLength(reused?.cache ?? []) == 512)
        #expect(PromptCacheIO.isSliceable(reused?.cache ?? []) == false)        // the recurrent layer came back too
    }
}
