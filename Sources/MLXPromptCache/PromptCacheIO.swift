import Foundation
import MLX
import MLXLMCommon

/// The `KVCache`-touching half: load/trim/save via the mlx-swift-lm primitives. Plain helpers (the
/// store calls them on the caller's thread); kept separate so they're unit-testable in isolation.
enum PromptCacheIO {
    enum Failure: Error { case notTrimmable, trimUnderflow, hybridNotAtBoundary, noSliceableLayer }
    static let metaSignature = "mlxpc.signature"
    /// Present on a delta file: "`from`,`to`" token range. Its presence is not load-bearing (the
    /// catalog drives chaining), but it makes a file self-describing for debugging.
    static let metaDelta = "mlxpc.delta"

    /// A layer whose KV is per-token sliceable (attention). `MambaCache`/`ArraysCache`/`RotatingKVCache`/
    /// `ChunkedKVCache`/`CacheList` hold recurrent or circular state that can't be trimmed to a token
    /// prefix — this mirrors `HotCodec.extract`'s gate and oMLX's `supports_block_slicing`. Subtype-first
    /// (`QuantizedKVCache` before `KVCacheSimple`), matching `HotCodec`.
    /// `ChunkedKVCache` is excluded first: it subclasses `KVCacheSimple`, so the `is KVCacheSimple`
    /// test would otherwise report it sliceable — contradicting the line above and the truth, since it
    /// carries a `startPosition` and front-trims (`maybeTrimFront`), so slicing it by absolute offset
    /// mis-keys the prefix. No shipped model returns one today, so this changes no in-use behaviour; it
    /// aligns the code with its documented intent and makes a future chunked model degrade to a clean
    /// boundary-only capture instead of a silently wrong trim.
    static func isSliceableLayer(_ c: KVCache) -> Bool {
        if c is ChunkedKVCache { return false }
        return c is QuantizedKVCache || c is KVCacheSimple
    }

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

    // MARK: - Delta (v2) — write only the new token range; reuse chains the files

    /// True iff a delta write is safe for this cache. Every *sliceable* layer must be a plain
    /// `KVCacheSimple` (its KV slices per token along axis 2); every *non-sliceable* layer must be an
    /// `ArraysCache`/`MambaCache` (recurrent state, stored whole per delta file, deepest-wins on
    /// reuse). Anything else — `QuantizedKVCache`, `RotatingKVCache`, `ChunkedKVCache`, `CacheList` —
    /// makes this false and the store falls back to the legacy whole-snapshot path, unchanged.
    static func canDelta(_ cache: [KVCache]) -> Bool {
        guard tokenLength(cache) != nil else { return false }
        return cache.allSatisfy { layer in
            isSliceableLayer(layer) ? type(of: layer) == KVCacheSimple.self : layer is ArraysCache
        }
    }

    /// Save ONLY tokens `[from, to)`. Attention layers are sliced along the sequence axis; recurrent
    /// layers carry their FULL state as of `to` (so a hybrid delta file is valid only when the cache
    /// sits exactly at `to` — the same boundary rule `save` enforces, checked here explicitly).
    /// Precondition: `canDelta(liveCache)`.
    @discardableResult
    static func saveDelta(from: Int, to: Int, liveCache: [KVCache], url: URL,
                          signature: CacheSignature) throws -> Int {
        guard let offset = tokenLength(liveCache) else { throw Failure.noSliceableLayer }
        let hasRecurrent = liveCache.contains { !isSliceableLayer($0) }
        // A recurrent layer summarises the whole prefix, so its state is only the prefix's state when
        // the cache is exactly at `to`. Past `to` (e.g. a post-generation cache) it is untrimmable
        // garbage — refuse, exactly as the whole-snapshot path does.
        if hasRecurrent, offset != to { throw Failure.hybridNotAtBoundary }
        guard offset >= to else { throw Failure.trimUnderflow }

        let delta: [KVCache] = liveCache.map { layer in
            guard isSliceableLayer(layer) else { return layer.copy() }   // full recurrent state @ to
            let s = layer.state                                          // [keys, values] to `offset`
            let sliced = s.map { $0[.ellipsis, from ..< to, 0...] }      // just [from, to)
            let c = KVCacheSimple()
            c.state = sliced                                            // setter ⇒ offset == to-from
            return c
        }
        try savePromptCache(url: url, cache: delta,
                            metadata: [metaSignature: signature.canonical, metaDelta: "\(from),\(to)"])
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? Int) ?? 0
    }

    /// Load and reassemble a reuse chain. One url ⇒ the file as-is (legacy whole snapshot or a single
    /// delta). N urls ⇒ per layer, concatenate the attention slices along axis 2 across files and take
    /// the DEEPEST file's recurrent layer (its state summarises the whole chain). `nil` on any missing/
    /// corrupt/signature-mismatched file, or a layer-count mismatch (⇒ caller self-heals to a miss).
    static func reassemble(_ urls: [URL], signature: CacheSignature) -> [KVCache]? {
        guard !urls.isEmpty else { return nil }
        var loaded: [[KVCache]] = []
        loaded.reserveCapacity(urls.count)
        for url in urls {
            guard let (cache, meta) = try? loadPromptCache(url: url),
                  meta[metaSignature] == signature.canonical else { return nil }
            loaded.append(cache)
        }
        if loaded.count == 1 {
            eval(loaded[0].flatMap { $0.state })
            return loaded[0]
        }
        let layerCount = loaded[0].count
        guard loaded.allSatisfy({ $0.count == layerCount }) else { return nil }
        var result: [KVCache] = []
        result.reserveCapacity(layerCount)
        for i in 0 ..< layerCount {
            if isSliceableLayer(loaded[0][i]) {
                let keys = loaded.map { $0[i].state[0] }
                let vals = loaded.map { $0[i].state[1] }
                let c = KVCacheSimple()
                c.state = [concatenated(keys, axis: 2), concatenated(vals, axis: 2)]
                result.append(c)
            } else {
                result.append(loaded.last![i])                         // deepest recurrent state wins
            }
        }
        eval(result.flatMap { $0.state })
        return result
    }
}
