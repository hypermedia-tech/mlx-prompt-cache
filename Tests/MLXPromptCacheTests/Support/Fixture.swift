import Foundation
import MLX
import MLXLMCommon
@testable import MLXPromptCache

enum Fixture {
    static let signature = CacheSignature(
        modelId: "test-model",
        kvDType: "bf16",
        kvBits: nil,
        buildVersion: "t1"
    )
    
    static func tokens(_ n: Int, seed: Int = 0) -> [Int] {
        (0..<n).map { ($0 + seed) % 50_000 }
    }
    
    static func tempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxpc-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

extension Fixture {
    /// A synthetic, trimmable `[KVCache]` of `tokens` length — no model. One `KVCacheSimple` per
    /// "layer", shaped [B=1, kvHeads, tokens, headDim] like a real cache.
    static func syntheticCache(tokens: Int, layers: Int = 2, kvHeads: Int = 2, headDim: Int = 8) -> [KVCache] {
        (0..<layers).map { _ in
            let c = KVCacheSimple()
            let k = MLXArray.zeros([1, kvHeads, tokens, headDim], dtype: .bfloat16)
            let v = MLXArray.zeros([1, kvHeads, tokens, headDim], dtype: .bfloat16)
            _ = c.update(keys: k, values: v)
            return c
        }
    }
}
