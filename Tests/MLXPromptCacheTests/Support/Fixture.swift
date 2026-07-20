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

extension Fixture {
    /// Non-zero, position-dependent `[KVCache]` so a byte-layout / stride / dtype bug can't hide
    /// behind zeros. Each element is distinct within a layer (×coprime mod 251), and layers differ.
    static func patternedCache(tokens: Int, layers: Int = 2, kvHeads: Int = 2, headDim: Int = 8) -> [KVCache] {
        (0..<layers).map { layer in
            let count = kvHeads * tokens * headDim
            func pattern(_ mult: Int) -> MLXArray {
                let vals = (0..<count).map { Float(($0 * mult + layer) % 251) }   // ≤ 250 ⇒ bf16-exact
                return MLXArray(vals).reshaped([1, kvHeads, tokens, headDim]).asType(.bfloat16)
            }
            let c = KVCacheSimple()
            _ = c.update(keys: pattern(7 + 2 * layer), values: pattern(13 + 2 * layer))
            return c
        }
    }
}

extension Fixture{
    /// A real `QuantizedKVCache` with no model: prefill a simple cache, then `toQuantized`.
    static func quantizedCache(tokens: Int, layers: Int = 2, kvHeads: Int = 2, headDim: Int = 64, groupSize: Int = 64, bits: Int = 4) -> [KVCache] {
        patternedCache(tokens: tokens, layers: layers, kvHeads: kvHeads, headDim: headDim).map {
            ($0 as! KVCacheSimple).toQuantized(groupSize: groupSize, bits: bits)
        }
    }
}

extension Fixture {
    /// A hybrid `[KVCache]`: a non-sliceable `MambaCache` FIRST (offset 0 — the exact layout that made
    /// `cache.first?.offset` return 0 and throw `trimUnderflow`), then a sliceable `KVCacheSimple`
    /// prefilled to `slice` tokens.
    static func hybridCache(slice: Int, kvHeads: Int = 2, headDim: Int = 8) -> [KVCache] {
        let attn = KVCacheSimple()
        _ = attn.update(
            keys: MLXArray.zeros([1, kvHeads, slice, headDim], dtype: .bfloat16),
            values: MLXArray.zeros([1, kvHeads, slice, headDim], dtype: .bfloat16)
        )
        return [MambaCache(), attn]                       // non-sliceable layer first
    }

    /// No sliceable layer at all — a pure-SSM model.
    static func pureSSMCache() -> [KVCache] { [MambaCache()] }
}

extension Fixture {
    /// A `ChunkedKVCache` prefilled to `tokens`. It SUBCLASSES `KVCacheSimple` — the trap that made
    /// it slip past the attention-only gates — but carries a `startPosition` and chunk metaState, so
    /// it is not per-token sliceable. `chunkSize` set so it is a realistic chunked cache, not a plain
    /// one masquerading.
    static func chunkedCache(tokens: Int, chunkSize: Int = 4, kvHeads: Int = 2, headDim: Int = 8) -> ChunkedKVCache {
        let c = ChunkedKVCache(chunkSize: chunkSize)
        _ = c.update(
            keys: MLXArray.zeros([1, kvHeads, tokens, headDim], dtype: .bfloat16),
            values: MLXArray.zeros([1, kvHeads, tokens, headDim], dtype: .bfloat16)
        )
        return c
    }

    /// A hybrid attention + chunked cache — the shape a real sliding-window-plus-full-attention model
    /// produces. Sliceable `KVCacheSimple` first, non-sliceable `ChunkedKVCache` second, both to
    /// `slice` tokens.
    static func attentionPlusChunked(slice: Int, chunkSize: Int = 4, kvHeads: Int = 2, headDim: Int = 8) -> [KVCache] {
        let attn = KVCacheSimple()
        _ = attn.update(
            keys: MLXArray.zeros([1, kvHeads, slice, headDim], dtype: .bfloat16),
            values: MLXArray.zeros([1, kvHeads, slice, headDim], dtype: .bfloat16)
        )
        return [attn, chunkedCache(tokens: slice, chunkSize: chunkSize, kvHeads: kvHeads, headDim: headDim)]
    }
}
