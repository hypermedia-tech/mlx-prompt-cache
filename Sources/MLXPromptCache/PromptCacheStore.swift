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
    public let blockSize: Int
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

        // Cold: chain the delta files (or the single legacy snapshot) and reassemble private buffers.
        // `chain` stops at the first hole, so `effectiveMatch` may be shorter than `hit.matchedTokens`
        // if a mid-chain file was budget-evicted. Touch the WHOLE chain (not just the deepest file) so
        // the next eviction can't drop an early file and orphan everything after it.
        let links = catalog.withLock { cat -> [Catalog.ChainLink] in
            let l = cat.chain(hashes); cat.touch(l.map { $0.fileName }); return l
        }
        let effectiveMatch = links.last?.tokenCount ?? 0
        let urls = links.map { directory.appendingPathComponent($0.fileName) }
        guard effectiveMatch > 0,
              let full = PromptCacheIO.reassemble(urls, signature: signature),
              let fullTokens = PromptCacheIO.tokenLength(full) else {
            // Self-heal: a file in the chain is gone/corrupt. Evict the deepest matched entry and drop
            // it from hot, so the next `record` re-stores. `chain` recomputes from what survives.
            catalog.withLock { cat in
                cat.evict(hit.fileName)
                Self.writeIndex(cat, to: directory.appendingPathComponent("index.json"))
            }
            hot.withLock { $0.drop(hit.fileName) }
            log("reuse: MISS — chain for \(hit.fileName) (\(links.count) files) missing/corrupt; evicted (will re-record)")
            return nil
        }
        log("reuse: reassembled \(links.count) file\(links.count == 1 ? "" : "s") → \(fullTokens) tokens"
            + (effectiveMatch < hit.matchedTokens ? " (chain HOLE — capped from \(hit.matchedTokens))" : ""))

        if fullTokens == effectiveMatch {                          // full match (always so for a delta chain)
            // Hot promote only for a SINGLE file — a multi-file chain has no one file that is the whole
            // prefix, so it cannot be keyed into the file-name-addressed hot tier.
            if links.count == 1, let blobs = HotCodec.extract(full) {
                hot.withLock { $0.insert(links[0].fileName, blobs: blobs, fullTokens: fullTokens) }
            }
            log("reuse: HIT (cold/disk, full) \(effectiveMatch) tokens")
            return Reused(cache: full, matchedTokens: effectiveMatch)
        } else {                                                   // partial → trim (single legacy file only)
            guard let trimmed = PromptCacheIO.trim(full, toMatched: effectiveMatch) else {
                log("reuse: MISS — trim to \(effectiveMatch) failed (offset \(fullTokens), not trimmable?)")
                return nil
            }
            log("reuse: HIT (cold/disk, partial) \(effectiveMatch)/\(fullTokens) tokens")
            return Reused(cache: trimmed, matchedTokens: effectiveMatch)
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
    
    /// Matched token count for the longest cached prefix of `tokens` — catalog-only: no snapshot IO,
    /// no MLX, no GPU, and NO LRU touch (see `Catalog.probe`). The idempotence probe for background
    /// warming: `peek(...) >= alignedLength` means "already warm, skip".
    ///
    /// Honesty note: the probe trusts the in-memory catalog. A snapshot deleted out-of-band while the
    /// store is live can make `peek` over-report until the next `reuse` self-heals the dead entry.
    /// Consequence: a warm pass skips a file it should have warmed, and the next ask pays one cold
    /// prefill — never a wrong answer.
    public func peek(forTokens tokens: [Int]) -> Int {
        let hashes = BlockHasher.boundaries(for: tokens, blockSize: blockSize, signature: signature)
        guard !hashes.isEmpty else { return 0 }
        let matched = catalog.withLock { $0.probe(hashes)?.matchedTokens ?? 0 }
        log("peek: \(tokens.count) tokens → \(hashes.count) full \(blockSize)-blocks; matched \(matched)")
        return matched
    }

    /// Chain hash of the first `n` tokens' last full block — the key `WarmStore` uses to verify that
    /// a held cache actually covers the tokens a resume is asking about. Keeps `signature` private
    /// while letting the coordinator build the same hash the catalog keys on.
    package func frontierHash(forTokens tokens: [Int], upTo n: Int) -> BlockHash? {
        guard n > 0, n <= tokens.count else { return nil }
        return BlockHasher.boundaries(for: Array(tokens[0 ..< n]),
                                      blockSize: blockSize, signature: signature).last
    }

    private func write(prefixTokens: [Int], cache: [KVCache], warmHot: Bool) throws {
        let hashes = BlockHasher.boundaries(for: prefixTokens, blockSize: blockSize, signature: signature)
        let types = Set(cache.map { String(describing: type(of: $0)) }).sorted().joined(separator: ", ")
        log("record: \(prefixTokens.count) tokens → \(hashes.count) full \(blockSize)-blocks; live cache = \(cache.count) layers, attnLen=\(PromptCacheIO.tokenLength(cache).map { "\($0)" } ?? "none"), sliceable=\(PromptCacheIO.isSliceable(cache)), types=[\(types)]")

        let delta = PromptCacheIO.canDelta(cache)
        guard let plan = catalog.withLock({ $0.planRecord(hashes, blockSize: blockSize, delta: delta) }) else {
            log("record: NOTHING STORED — planRecord returned nil (\(hashes.isEmpty ? "0 full blocks: prompt < \(blockSize) tokens" : "all \(hashes.count) blocks already in catalog"))")
            return
        }
        log("record: plan \(plan.delta ? "DELTA range [\(plan.fromToken),\(plan.toToken))" : "WHOLE up to \(plan.boundaries.last?.tokenCount ?? 0)") — \(plan.boundaries.count) new boundaries → \(plan.fileName)")

        // Recreate the directory if it was deleted out-of-band (e.g. a live `rm -rf`) — otherwise
        // `savePromptCache` fails to open the file. Completes the self-heal alongside the catalog evict.
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(plan.fileName)
        let bytes: Int
        do {
            if plan.delta {
                bytes = try PromptCacheIO.saveDelta(from: plan.fromToken, to: plan.toToken,
                                                    liveCache: cache, url: url, signature: signature)
            } else {
                bytes = try PromptCacheIO.save(prefixTokenCount: prefixTokens.count, liveCache: cache, url: url, signature: signature)
            }
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
