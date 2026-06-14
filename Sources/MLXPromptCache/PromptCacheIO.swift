import Foundation
import MLXLMCommon

/// The `KVCache`-touching half: load/save/trim via the mlx-swift-lm primitives. Plain helpers (the
/// store calls them on the caller's thread); kept separate so they're unit-testable in isolation.
enum PromptCacheIO {
    enum Failure: Error { case notTrimmable, trimUnderflow }
    static let metaSignature = "mlxpc.signature"
    
    static func load(url: URL, matchedTokens: Int, signature: CacheSignature) -> [KVCache]? {
        guard let (cache, meta) = try? loadPromptCache(url: url) else { return nil }
        guard meta[metaSignature] == signature.canonical else { return nil }
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
