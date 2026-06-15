import Foundation
import MLX
import MLXNN
import MLXLMCommon

/// Minimal `LanguageModel` for coordinator tests — no weights. `newCache` hands back caller-chosen layers;
/// `prepare` defers the whole prompt to one forward, and that forward advances every `KVCacheSimple` by the
/// input length — so `prefillOnly` lands the cache at exactly the prompt length. `MambaCache` layers are
/// left untouched (no token offset), matching a real hybrid.
final class StubModel: Module, LanguageModel {
    let kvHeads: Int
    let headDim: Int
    let makeLayers: () -> [KVCache]
    
    init(kvHeads: Int = 2, headDim: Int = 8, layers: @escaping () -> [KVCache]) {
        self.kvHeads = kvHeads
        self.headDim = headDim
        self.makeLayers = layers
        super.init()
    }
    
    func newCache(parameters: GenerateParameters?) -> [KVCache] { makeLayers() }
    
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult {
        .tokens(input.text)
    }
    
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?) -> LMOutput {
        let n = input.tokens.shape.last ?? 0
        for case let kv as KVCacheSimple in cache ?? [] {
            _ = kv.update(
                keys: MLXArray.zeros([1, kvHeads, n, headDim], dtype: .bfloat16),
                values: MLXArray.zeros([1, kvHeads, n, headDim], dtype: .bfloat16)
            )
        }
        return LMOutput(logits: MLXArray.zeros([1, max(1, n), 1]))
    }
}
