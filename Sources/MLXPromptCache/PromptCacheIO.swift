import Foundation
import MLX
import MLXLMCommon

/// The `KVCache`-touching half: load/trim/save via the mlx-swift-lm primitives. Plain helpers (the
/// store calls them on the caller's thread); kept separate so they're unit-testable in isolation.
enum PromptCacheIO {
    enum Failure: Error { case notTrimmable, trimUnderflow }
    static let metaSignature = "mlxpc.signature"

    /// Deserialise the whole snapshot (no trim). Signature re-checked; arrays `eval`'d so the result is
    /// materialised and detached from any lazy graph. nil on missing/corrupt/signature-mismatch.
    static func loadFull(url: URL, signature: CacheSignature) -> [KVCache]? {
        guard let (cache, meta) = try? loadPromptCache(url: url) else { return nil }
        guard meta[metaSignature] == signature.canonical else { return nil }
        eval(cache.flatMap { $0.state })
        return cache
    }

    /// Trim a **private** loaded cache to `matchedTokens` in place and verify it landed. Returns the same
    /// array, or nil if the trim under-delivers (e.g. a wrapped RotatingKVCache).
    static func trim(_ cache: [KVCache], toMatched matchedTokens: Int) -> [KVCache]? {
        guard let loaded = cache.first?.offset, loaded >= matchedTokens else { return nil }
        let toTrim = loaded - matchedTokens
        if toTrim > 0 {
            guard canTrimPromptCache(cache) else { return nil }
            trimPromptCache(cache, numTokens: toTrim)
        }
        guard cache.first?.offset == matchedTokens else { return nil }
        return cache
    }

    @discardableResult
    static func save(prefixTokenCount: Int, liveCache: [KVCache], url: URL, signature: CacheSignature) throws -> Int {
        let snapshot = liveCache.map { $0.copy() }
        if let offset = snapshot.first?.offset, offset > prefixTokenCount {
            guard canTrimPromptCache(snapshot) else { throw Failure.notTrimmable }
            trimPromptCache(snapshot, numTokens: offset - prefixTokenCount)
        }
        guard snapshot.first?.offset == prefixTokenCount else { throw Failure.trimUnderflow }
        try savePromptCache(url: url, cache: snapshot, metadata: [metaSignature: signature.canonical])
        let attrs = try? FileManager.default.attributesOfItem(atPath: url.path)
        return (attrs?[.size] as? Int) ?? 0
    }
}
