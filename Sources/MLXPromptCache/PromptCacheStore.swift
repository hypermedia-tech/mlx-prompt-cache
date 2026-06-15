import Foundation
import Synchronization
import MLXLMCommon

/// Reuse result. Plain struct — holds non-`Sendable` `[KVCache]`, returned synchronously to the
/// caller, so it never crosses an isolation boundary
public struct Reused { public let cache: [KVCache]; public let matchedTokens: Int }

/// Persistent prompt/prefix KV cache for mlx-swift: longest-prefix reuse across prompts, a byte-budget
/// disk tier, and an optional in-RAM hot tier (raw KV bytes; enabled by `hotBudgetBytes > 0`). Both the
/// catalog and the hot tier are `Sendable` value types behind a `Mutex`. Design:
/// `mlx-prompt-cache-module-brief.md` + `-phase2-hot-tier.md`.
public final class PromptCacheStore: Sendable {
    private let directory: URL
    private let budgetBytes: Int
    private let blockSize: Int
    private let signature: CacheSignature
    private let catalog: Mutex<Catalog>
    private let hot: Mutex<HotCache>

    /// `hotBudgetBytes: 0` disables the RAM tier — byte-for-byte the disk-only behaviour.
    public init(directory: URL, budgetBytes: Int, signature: CacheSignature,
                blockSize: Int = 256, hotBudgetBytes: Int = 0) throws {
        self.directory = directory
        self.budgetBytes = budgetBytes
        self.blockSize = blockSize
        self.signature = signature
        self.hot = Mutex(HotCache(budgetBytes: hotBudgetBytes))
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.catalog = Mutex(Catalog.loadOrReset(directory: directory, signature: signature, blockSize: blockSize))
    }

    /// Longest cached prefix for `tokens`, served from RAM if resident else loaded from disk, trimmed to
    /// the match. Runs on the caller's thread.
    public func reuse(forTokens tokens: [Int]) -> Reused? {
        let hashes = BlockHasher.boundaries(for: tokens, blockSize: blockSize, signature: signature)
        guard let hit = catalog.withLock({ $0.lookup(hashes) }) else { return nil }

        // Hot, full-length match: take Sendable bytes out of the lock, reconstruct a private cache outside it.
        if let entry = hot.withLock({ $0.resident(hit.fileName) }), entry.fullTokens == hit.matchedTokens {
            return Reused(cache: HotCodec.reconstruct(entry.blobs), matchedTokens: hit.matchedTokens)
        }

        // Cold: load the whole snapshot (private buffers).
        let url = directory.appendingPathComponent(hit.fileName)
        guard let full = PromptCacheIO.loadFull(url: url, signature: signature),
              let fullTokens = full.first?.offset else { return nil }

        if fullTokens == hit.matchedTokens {                       // full match → promote bytes + vend the load
            if let blobs = HotCodec.extract(full) {
                hot.withLock { $0.insert(hit.fileName, blobs: blobs, fullTokens: fullTokens) }
            }
            return Reused(cache: full, matchedTokens: hit.matchedTokens)
        } else {                                                   // partial → trim private cache, no hot
            guard let trimmed = PromptCacheIO.trim(full, toMatched: hit.matchedTokens) else { return nil }
            return Reused(cache: trimmed, matchedTokens: hit.matchedTokens)
        }
    }

    /// Snapshot the stable prefix of a freshly-prefilled cache for future reuse (disk only; the hot tier
    /// warms on the next cold reuse).
    public func record(prefixTokens: [Int], cache: [KVCache]) throws {
        try write(prefixTokens: prefixTokens, cache: cache, warmHot: false)
    }

    /// Like `record` but also pre-warms the RAM tier — for launch-time warming of a known working set.
    public func preload(prefixTokens: [Int], cache: [KVCache]) throws {
        try write(prefixTokens: prefixTokens, cache: cache, warmHot: true)
    }

    /// Drop all residents (model swap / admin clear). Disk is untouched.
    public func clearHot() { hot.withLock { $0.clear() } }

    private func write(prefixTokens: [Int], cache: [KVCache], warmHot: Bool) throws {
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
            hot.withLock { $0.drop(name) }                         // keep hot coherent with disk
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
        if warmHot,
           let full = PromptCacheIO.loadFull(url: url, signature: signature),
           let fullTokens = full.first?.offset,
           let blobs = HotCodec.extract(full) {
            hot.withLock { $0.insert(plan.fileName, blobs: blobs, fullTokens: fullTokens) }
        }
    }

    private static func writeIndex(_ catalog: Catalog, to url: URL) {
        guard let data = try? JSONEncoder().encode(catalog) else { return }
        try? data.write(to: url, options: .atomic)
    }
}
