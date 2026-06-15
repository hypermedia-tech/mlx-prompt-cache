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
}
