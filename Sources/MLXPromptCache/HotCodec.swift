import Foundation
import MLX
import MLXLMCommon

/// One serialised tensor: contiguous little-endian bytes in `dtype`/`shape` (the `asData` contract).
struct TensorBytes: Sendable {
    let data: Data
    let dtype: DType
    let shape: [Int]
}

/// A whole cache serialised to bytes: its concrete type + `metaState` + its `state` tensors.
struct CacheBytes: Sendable {
    let className: String        // "KVCacheSimple" | "QuantizedKVCache"
    let metaState: [String]
    let tensors: [TensorBytes]
}

/// Extract `[KVCache]` ⇄ `Sendable` bytes — oMLX's `_extract_tensor_bytes` / restore, adapted to
/// mlx-swift-lm's public `state` / `metaState`. (We don't reuse `loadPromptCache`'s reconstruction:
/// it is `private` and URL-only, so the faithful port extracts per-tensor directly from the cache.)
enum HotCodec {

    /// Snapshot a cache to bytes, or `nil` if any layer is a type we don't hot-reconstruct (the
    /// snapshot then stays cold-only — a lossless degradation).
    static func extract(_ caches: [KVCache]) -> [CacheBytes]? {
        var out: [CacheBytes] = []
        out.reserveCapacity(caches.count)
        for cache in caches {
            let className: String
            if cache is QuantizedKVCache { className = "QuantizedKVCache" }   // subtype-first
            else if cache is KVCacheSimple { className = "KVCacheSimple" }
            else { return nil }                                              // unsupported → no hot entry
            let tensors = cache.state.map {
                TensorBytes(data: $0.asData(access: .copy).data, dtype: $0.dtype, shape: $0.shape)   // evals + copies
            }
            out.append(CacheBytes(className: className, metaState: cache.metaState, tensors: tensors))
        }
        return out
    }

    /// Rebuild a fresh, private `[KVCache]`. Each array owns a freshly-copied buffer the finalizer frees
    /// (ownership transfers to MLXArray) — no shared storage, no lazy graph.
    static func reconstruct(_ blobs: [CacheBytes]) -> [KVCache] {
        blobs.map { blob in
            let arrays: [MLXArray] = blob.tensors.map { t in
                let n = max(1, t.data.count)
                let buf = UnsafeMutableRawPointer.allocate(byteCount: n, alignment: 64)
                t.data.copyBytes(to: buf.assumingMemoryBound(to: UInt8.self), count: t.data.count)
                return MLXArray(rawPointer: buf, t.shape, dtype: t.dtype, finalizer: { buf.deallocate() })
            }
            var cache: KVCache = (blob.className == "QuantizedKVCache") ? QuantizedKVCache() : KVCacheSimple()
            cache.state = arrays                 // same order as restoreCacheFromMetaState
            cache.metaState = blob.metaState
            return cache
        }
    }

    static func footprint(_ blobs: [CacheBytes]) -> Int {
        blobs.reduce(0) { $0 + $1.tensors.reduce(0) { $0 + $1.data.count } }
    }
}
