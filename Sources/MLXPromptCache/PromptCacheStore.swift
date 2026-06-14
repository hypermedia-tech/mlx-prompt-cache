import Foundation
import Synchronization
import MLXLMCommon

/// Reuse result. Plain struct — holds non-`Sendable` `[KVCache]`, returned synchronously to the
/// caller, so it never crosses an isolation boundary
public struct Reused { public let cache: [KVCache]; public let matchedTokens: Int }

/// Persistent prompt/prefix KV cache for mlx-swift.
/// Design + full API: `mlx-prompt-cache-module-brief.md`. This is the compiling shell;
/// `reuse` / `record` / the on-disk index land in the implementation pass (where the
/// `MLXLMCommon` import + KVCache handling come in).
public final class PromptCacheStore: Sendable {
    private let directory: URL
    private let budgetBytes: Int
    private let blockSize: Int
    private let signature: CacheSignature
    private let catalog: Mutex<Catalog>
    
    public init(directory: URL, budgetBytes: Int, signature: CacheSignature, blockSize: Int = 256) throws {
        self.directory = directory
        self.budgetBytes = budgetBytes
        self.blockSize = blockSize
        self.signature = signature
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.catalog = Mutex(Catalog.loadOrReset(directory: directory, signature: signature, blockSize: blockSize))
    }
    
    /// Longest cached prefix for `tokens`, loaded + trimmed to the match. Runs on the caller's thread.
    public func reuse(forTokens tokens: [Int]) -> Reused? {
        let hashes = BlockHasher.boundaries(for: tokens, blockSize: blockSize, signature: signature)
        guard let hit = catalog.withLock({ $0.lookup(hashes) }) else { return nil }
        let url = directory.appendingPathComponent(hit.fileName)
        guard let cache = PromptCacheIO.load(url: url, matchedTokens: hit.matchedTokens,
                                             signature: signature) else { return nil }
        return Reused(cache: cache, matchedTokens: hit.matchedTokens)
    }
    
    /// Snapshot the stable prefix of a freshly-prefilled cache for future reuse.
    public func record(prefixTokens: [Int], cache: [KVCache]) throws {
        let hashes = BlockHasher.boundaries(for: prefixTokens, blockSize: blockSize, signature: signature)
        guard let plan = catalog.withLock({ $0.planRecord(hashes, blockSize: blockSize) }) else { return }
        let url = directory.appendingPathComponent(plan.fileName)
        let bytes = try PromptCacheIO.save(prefixTokenCount: prefixTokens.count, liveCache: cache, url: url, signature: signature)
        let idxURL = directory.appendingPathComponent("index.json")
        let toDelete: [String] = catalog.withLock { cat in
            let deleted = cat.commit(plan, byteSize: bytes, budgetBytes: budgetBytes)
            Self.writeIndex(cat, to: idxURL)
            return deleted
        }
        for name in toDelete {
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
    }
    
    private static func writeIndex(_ catalog: Catalog, to url: URL) {
        guard let data = try? JSONEncoder().encode(catalog) else { return }
        try? data.write(to: url, options: .atomic)
    }
}
