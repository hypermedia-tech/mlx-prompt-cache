import Foundation
import MLX
import MLXLMCommon

/// The `KVCache`-touching half: load/trim/save via the mlx-swift-lm primitives. Plain helpers (the
/// store calls them on the caller's thread); kept separate so they're unit-testable in isolation.
enum PromptCacheIO {
    enum Failure: Error { case notTrimmable, trimUnderflow, hybridNotAtBoundary, noSliceableLayer }
    static let metaSignature = "mlxpc.signature"

    /// A layer whose KV is per-token sliceable (attention). `MambaCache`/`ArraysCache`/`RotatingKVCache`/
    /// `ChunkedKVCache`/`CacheList` hold recurrent or circular state that can't be trimmed to a token
    /// prefix — this mirrors `HotCodec.extract`'s gate and oMLX's `supports_block_slicing`. Subtype-first
    /// (`QuantizedKVCache` before `KVCacheSimple`), matching `HotCodec`.
    static func isSliceableLayer(_ c: KVCache) -> Bool { c is QuantizedKVCache || c is KVCacheSimple }

    /// True iff EVERY layer is sliceable — only then can a snapshot be trimmed to a sub-prefix. A single
    /// non-sliceable layer (i.e. a hybrid model) makes the whole snapshot reusable only at a *captured*
    /// boundary, never an arbitrary trim.
    static func isSliceable(_ cache: [KVCache]) -> Bool { cache.allSatisfy(isSliceableLayer) }

    /// Token length read from a sliceable (attention) layer's `offset` — NEVER `cache.first` (in a hybrid
    /// the first layer is a `MambaCache` reporting `0`, which was the cause of the `trimUnderflow` crash).
    /// `nil` = no attention layer at all (a pure-SSM model), which this prefix scheme can't key on.
    static func tokenLength(_ cache: [KVCache]) -> Int? { cache.first(where: isSliceableLayer)?.offset }

    /// Deserialise the whole snapshot (no trim). Signature re-checked; arrays `eval`'d so the result is
    /// materialised and detached from any lazy graph. nil on missing/corrupt/signature-mismatch.
    static func loadFull(url: URL, signature: CacheSignature) -> [KVCache]? {
        guard let (cache, meta) = try? loadPromptCache(url: url) else { return nil }
        guard meta[metaSignature] == signature.canonical else { return nil }
        eval(cache.flatMap { $0.state })
        return cache
    }

    /// Trim a **private** loaded cache to `matchedTokens` in place and verify it landed. A non-sliceable
    /// (hybrid) snapshot cannot be trimmed: it is usable only when already exactly `matchedTokens`
    /// (oMLX's "partial prefix match → reject"). Returns the cache, or nil if it can't satisfy the match.
    static func trim(_ cache: [KVCache], toMatched matchedTokens: Int) -> [KVCache]? {
        guard let loaded = tokenLength(cache) else { return nil }
        guard isSliceable(cache) else { return loaded == matchedTokens ? cache : nil }
        guard loaded >= matchedTokens else { return nil }
        let toTrim = loaded - matchedTokens
        if toTrim > 0 {
            guard canTrimPromptCache(cache) else { return nil }
            trimPromptCache(cache, numTokens: toTrim)
        }
        guard tokenLength(cache) == matchedTokens else { return nil }
        return cache
    }

    @discardableResult
    static func save(prefixTokenCount: Int, liveCache: [KVCache], url: URL, signature: CacheSignature) throws -> Int {
        let snapshot = liveCache.map { $0.copy() }
        guard let length = tokenLength(snapshot) else { throw Failure.noSliceableLayer }
        if isSliceable(snapshot) {
            if length > prefixTokenCount {
                guard canTrimPromptCache(snapshot) else { throw Failure.notTrimmable }
                trimPromptCache(snapshot, numTokens: length - prefixTokenCount)
            }
            guard tokenLength(snapshot) == prefixTokenCount else { throw Failure.trimUnderflow }
        } else {
            // Hybrid: recurrent (Mamba/SSM) layers can't be trimmed, so the snapshot is only valid when it
            // is ALREADY exactly the prefix — i.e. captured at the boundary (a prefill-only / preload
            // pass), never a post-generation cache (which carries generated tokens that can't be removed).
            guard length == prefixTokenCount else { throw Failure.hybridNotAtBoundary }
        }
        try savePromptCache(url: url, cache: snapshot, metadata: [metaSignature: signature.canonical])
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? Int) ?? 0
    }
}
