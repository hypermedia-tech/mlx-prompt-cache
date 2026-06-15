import Foundation
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite struct PromptCacheCoordinatorTests {
    let sig = Fixture.signature
    
    private func coordinator() throws -> PromptCacheCoordinator {
        let store = try PromptCacheStore(directory: Fixture.tempDir(), budgetBytes: 1 << 30, signature: sig)
        return PromptCacheCoordinator(store: store)
    }
    
    @Test func capturesOnMiss() throws {
        let coord = try coordinator()
        let model = StubModel { [KVCacheSimple(), KVCacheSimple()] as [KVCache] }
        let tokens = Fixture.tokens(600)
        let prepared = coord.prepare(promptTokens: tokens, model: model, parameters: GenerateParameters())
        guard case let .captured(boundary) = prepared.outcome else {
            Issue.record("expected .captured, got \(prepared.outcome)"); return
        }
        #expect(boundary == 512)
        #expect(prepared.suffixStart == 512)
        #expect(PromptCacheIO.tokenLength(prepared.cache) == 512)
    }
    
    @Test func hitsOnRepeat() throws {
        let coord = try coordinator()
        let model = StubModel { [KVCacheSimple(), KVCacheSimple()] as [KVCache] }
        let tokens = Fixture.tokens(600)
        _ = coord.prepare(promptTokens: tokens, model: model, parameters: GenerateParameters())
        let again = coord.prepare(promptTokens: tokens, model: model, parameters: GenerateParameters())
        guard case let .hit(matched) = again.outcome else {
            Issue.record("expected .hit, got \(again.outcome)"); return
        }
        #expect(matched == 512)
        #expect(again.suffixStart == 512)
    }
        
    @Test func uncacheableUnderOneBlock() throws {
        let coord = try coordinator()
        let model = StubModel { [KVCacheSimple()] as [KVCache] }
        let prepared = coord.prepare(promptTokens: Fixture.tokens(100), model: model, parameters: GenerateParameters())
        guard case .uncacheable = prepared.outcome else {
            Issue.record("expected .uncacheable, got \(prepared.outcome)"); return
        }
        #expect(prepared.suffixStart == 0)
    }
}
