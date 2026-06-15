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
    let blockSize: Int
    private let directory: URL
    private let budgetBytes: Int
    private let signature: CacheSignature
    private let catalog: Mutex<Catalog>
    private let hot: Mutex<HotCache>
    /// Diagnostic sink. Default no-op; inject `{ logger.info($0) }` to trace every step. (Demo aid.)
    private let log: @Sendable (String) -> Void

    /// `hotBudgetBytes: 0` disables the RAM tier — byte-for-byte the disk-only behaviour.
    public init(directory: URL, budgetBytes: Int, signature: CacheSignature,
                blockSize: Int = 256, hotBudgetBytes: Int = 0,
                log: @escaping @Sendable (String) -> Void = { _ in }) throws {
        self.directory = directory
        self.budgetBytes = budgetBytes
        self.blockSize = blockSize
        self.signature = signature
        self.hot = Mutex(HotCache(budgetBytes: hotBudgetBytes))
        self.log = log
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let loaded = Catalog.loadOrReset(directory: directory, signature: signature, blockSize: blockSize)
        self.catalog = Mutex(loaded)
        log("init: dir=\(directory.path)")
        log("init: budget=\(budgetBytes) block=\(blockSize) hotBudget=\(hotBudgetBytes) sig=[\(signature.canonical)]")
        log("init: catalog loaded \(loaded.files.count) files / \(loaded.byHash.count) boundaries / \(loaded.totalBytes) bytes")
    }

    /// Longest cached prefix for `tokens`, served from RAM if resident else loaded from disk, trimmed to
    /// the match. Runs on the caller's thread.
    public func reuse(forTokens tokens: [Int]) -> Reused? {
        let hashes = BlockHasher.boundaries(for: tokens, blockSize: blockSize, signature: signature)
        let (boundaries, files) = catalog.withLock { ($0.byHash.count, $0.files.count) }
        log("reuse: \(tokens.count) tokens → \(hashes.count) full \(blockSize)-blocks; catalog has \(boundaries) boundaries / \(files) files")

        guard let hit = catalog.withLock({ $0.lookup(hashes) }) else {
            log("reuse: MISS — no matching prefix in catalog (\(hashes.isEmpty ? "prompt < one block" : "no shared leading blocks"))")
            return nil
        }
        log("reuse: catalog matched \(hit.matchedTokens) tokens in \(hit.fileName)")

        // Hot, full-length match: take Sendable bytes out of the lock, reconstruct a private cache outside it.
        if let entry = hot.withLock({ $0.resident(hit.fileName) }), entry.fullTokens == hit.matchedTokens {
            log("reuse: HIT (hot/RAM) \(hit.matchedTokens) tokens")
            return Reused(cache: HotCodec.reconstruct(entry.blobs), matchedTokens: hit.matchedTokens)
        }

        // Cold: load the whole snapshot (private buffers).
        let url = directory.appendingPathComponent(hit.fileName)
        guard let full = PromptCacheIO.loadFull(url: url, signature: signature),
              let fullTokens = PromptCacheIO.tokenLength(full) else {
            log("reuse: MISS — loadFull failed for \(hit.fileName) (missing file / corrupt / signature mismatch / no sliceable layer)")
            return nil
        }
        log("reuse: loaded \(hit.fileName) — snapshot offset \(fullTokens)")

        if fullTokens == hit.matchedTokens {                       // full match → promote bytes + vend the load
            if let blobs = HotCodec.extract(full) {
                hot.withLock { $0.insert(hit.fileName, blobs: blobs, fullTokens: fullTokens) }
            }
            log("reuse: HIT (cold/disk, full) \(hit.matchedTokens) tokens")
            return Reused(cache: full, matchedTokens: hit.matchedTokens)
        } else {                                                   // partial → trim private cache, no hot
            guard let trimmed = PromptCacheIO.trim(full, toMatched: hit.matchedTokens) else {
                log("reuse: MISS — trim to \(hit.matchedTokens) failed (snapshot offset \(fullTokens), not trimmable?)")
                return nil
            }
            log("reuse: HIT (cold/disk, partial) \(hit.matchedTokens)/\(fullTokens) tokens")
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
        let types = Set(cache.map { String(describing: type(of: $0)) }).sorted().joined(separator: ", ")
        log("record: \(prefixTokens.count) tokens → \(hashes.count) full \(blockSize)-blocks; live cache = \(cache.count) layers, attnLen=\(PromptCacheIO.tokenLength(cache).map { "\($0)" } ?? "none"), sliceable=\(PromptCacheIO.isSliceable(cache)), types=[\(types)]")

        guard let plan = catalog.withLock({ $0.planRecord(hashes, blockSize: blockSize) }) else {
            log("record: NOTHING STORED — planRecord returned nil (\(hashes.isEmpty ? "0 full blocks: prompt < \(blockSize) tokens" : "all \(hashes.count) blocks already in catalog"))")
            return
        }
        log("record: plan \(plan.fileName) — \(plan.boundaries.count) boundaries, up to \(plan.boundaries.last?.tokenCount ?? 0) tokens")

        let url = directory.appendingPathComponent(plan.fileName)
        let bytes: Int
        do {
            bytes = try PromptCacheIO.save(prefixTokenCount: prefixTokens.count, liveCache: cache, url: url, signature: signature)
        } catch PromptCacheIO.Failure.hybridNotAtBoundary {
            // Hybrid model (recurrent Mamba/SSM layers): the post-generation cache carries the generated
            // tokens and can't be trimmed back to the prompt. Caching it needs a boundary capture (preload),
            // not record-after-generate — see mlx-prompt-cache-hybrid-models.md. Skip cleanly, not an error.
            log("record: SKIP — hybrid cache, can't record after generation (recurrent state untrimmable); capture at a boundary via preload. types=[\(types)]")
            return
        } catch PromptCacheIO.Failure.noSliceableLayer {
            log("record: SKIP — no attention layer to key on (pure-SSM model). types=[\(types)]")
            return
        } catch {
            log("record: ❌ SAVE FAILED — \(error)  (prefixTokenCount=\(prefixTokens.count), attnLen=\(PromptCacheIO.tokenLength(cache).map { "\($0)" } ?? "none"), types=[\(types)])")
            throw error
        }
        log("record: saved \(bytes) bytes → \(plan.fileName)")

        let idxURL = directory.appendingPathComponent("index.json")
        let toDelete: [String] = catalog.withLock { cat in
            let deleted = cat.commit(plan, byteSize: bytes, budgetBytes: budgetBytes)
            Self.writeIndex(cat, to: idxURL)
            return deleted
        }
        let (b, f, tb) = catalog.withLock { ($0.byHash.count, $0.files.count, $0.totalBytes) }
        log("record: committed → catalog now \(b) boundaries / \(f) files / \(tb) bytes; evicted \(toDelete.count); index.json \(FileManager.default.fileExists(atPath: idxURL.path) ? "WRITTEN" : "❌ MISSING")")

        for name in toDelete {
            hot.withLock { $0.drop(name) }                         // keep hot coherent with disk
            try? FileManager.default.removeItem(at: directory.appendingPathComponent(name))
        }
        if warmHot,
           let full = PromptCacheIO.loadFull(url: url, signature: signature),
           let fullTokens = PromptCacheIO.tokenLength(full),
           let blobs = HotCodec.extract(full) {
            hot.withLock { $0.insert(plan.fileName, blobs: blobs, fullTokens: fullTokens) }
            log("record: warmed RAM hot tier (\(fullTokens) tokens)")
        }
    }

    private static func writeIndex(_ catalog: Catalog, to url: URL) {
        guard let data = try? JSONEncoder().encode(catalog) else { return }
        try? data.write(to: url, options: .atomic)
    }
}
