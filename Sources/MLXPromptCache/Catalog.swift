import Foundation

/// Persistent catalogue of what's on disk. Pure value type — `Codable` for restart. Holds no `KVCache`.
struct Catalog: Codable {
    /// `indexLayout` tags the on-disk `index.json` shape — internal to this cache, NOT the package
    /// version. `1` is the first explicitly-tagged layout (delta files: per-persist ranges chained on
    /// reuse). A pre-tag `index.json` has no such field, so it fails to decode below and `loadOrReset`
    /// wipes it — the migration is the decode failure itself (greenfield, no in-place upgrade).
    struct Header: Codable { var signature: CacheSignature; var blockSize: Int; var indexLayout: Int }
    struct FileRecord: Codable { var fileName: String; var byteSize: Int; var lastAccess: Int
        var ownedBoundaries: Set<String> }
    struct Boundary: Codable { let fileName: String; let tokenCount: Int }

    var header: Header
    var byHash: [String: Boundary] = [:]
    var files: [String: FileRecord] = [:]
    var totalBytes = 0
    var clock = 0

    static let currentIndexLayout = 1

    struct Hit: Sendable { let fileName: String; let matchedTokens: Int }
    /// A planned write. `delta == true` ⇒ `boundaries` holds only the NEW blocks and the file covers
    /// the token range `[fromToken, toToken)`; the previous files keep their blocks (no orphaning).
    /// `delta == false` ⇒ every block (the legacy whole-snapshot write; supersedes the previous file).
    struct Plan: Sendable {
        let fileName: String
        let boundaries: [BoundaryPlan]
        let delta: Bool
        let fromToken: Int
        let toToken: Int
    }
    struct BoundaryPlan: Sendable { let hash: String; let tokenCount: Int }
    /// One resolvable link of a reuse chain: a file plus the tokenCount it reaches.
    struct ChainLink: Sendable { let fileName: String; let tokenCount: Int }
    
    /// Longest matched prefix; touches LRU. Mutating because it bumps `lastAccess`.
    mutating func lookup(_ hashes: [BlockHash]) -> Hit? {
        var deepest: Boundary?
        for h in hashes { guard let b = byHash[h.hex] else { break }; deepest = b }
        guard let hit = deepest else { return nil }
        clock += 1; files[hit.fileName]?.lastAccess = clock
        return Hit(fileName: hit.fileName, matchedTokens: hit.tokenCount)
    }
    
    /// Plan a record, or nil if there's nothing new to cache.
    ///
    /// `delta == true` (the default for sliceable caches): plan ONLY the blocks not already stored, so
    /// the write covers just the new token range. The file is named by its own last block, and the
    /// previous files are left untouched — `reuse` chains them. This is what kills the write
    /// amplification.
    ///
    /// `delta == false` (a `QuantizedKVCache` or any layer we can't slice per token): plan EVERY block
    /// into one file, the legacy whole-snapshot behaviour, which supersedes the previous file in
    /// `commit`.
    func planRecord(_ hashes: [BlockHash], blockSize: Int, delta: Bool) -> Plan? {
        guard !hashes.isEmpty, hashes.contains(where: { byHash[$0.hex] == nil }) else { return nil }
        if delta {
            let fresh = hashes.enumerated().filter { byHash[$0.element.hex] == nil }
            guard let last = fresh.last, let first = fresh.first else { return nil }
            let name = "snap-\(last.element.hex.prefix(32)).safetensors"
            let bounds = fresh.map {
                BoundaryPlan(hash: $0.element.hex, tokenCount: ($0.offset + 1) * blockSize)
            }
            return Plan(fileName: name, boundaries: bounds, delta: true,
                        fromToken: first.offset * blockSize,
                        toToken: (last.offset + 1) * blockSize)
        } else {
            let name = "snap-\(hashes.last!.hex.prefix(32)).safetensors"
            let bounds = hashes.enumerated().map {
                BoundaryPlan(hash: $0.element.hex, tokenCount: ($0.offset + 1) * blockSize)
            }
            return Plan(fileName: name, boundaries: bounds, delta: false,
                        fromToken: 0, toToken: hashes.count * blockSize)
        }
    }

    /// The ordered, CONTIGUOUS run of distinct files covering the longest matched prefix of `hashes`,
    /// with the token count each reaches. Stops at the first block that isn't catalogued OR whose file
    /// was budget-evicted from under a deeper block — so a hole in the chain caps the match there
    /// (never a wrong reassembly). One file for a legacy whole snapshot; N files for a delta chain.
    func chain(_ hashes: [BlockHash]) -> [ChainLink] {
        var links: [ChainLink] = []
        var lastFile: String?
        for h in hashes {
            guard let b = byHash[h.hex], files[b.fileName] != nil else { break }   // hole ⇒ stop
            if b.fileName != lastFile {
                links.append(ChainLink(fileName: b.fileName, tokenCount: b.tokenCount))
                lastFile = b.fileName
            } else {
                links[links.count - 1] = ChainLink(fileName: b.fileName, tokenCount: b.tokenCount)
            }
        }
        return links
    }
    
    /// Longest matched prefix WITHOUT touching LRU state. `lookup` is the mutating twin — it
    /// bumps `clock`/`lastAccess` because its caller (`reuse`) is about to use the snapshot,
    /// so the touch is honest recency. A background probe (`peek` — K1 warm idempotence, K2
    /// custodian sweeps) must not freshen entries it merely checks, or probe frequency would
    /// masquerade as heat and rot the eviction order.
    func probe(_ hashes: [BlockHash]) -> Hit? {
        var deepest: Boundary?
        for h in hashes {
            guard let b = byHash[h.hex] else { break }
            deepest = b
        }
        guard let hit = deepest else { return nil }
        return Hit(fileName: hit.fileName, matchedTokens: hit.tokenCount)
    }
    
    /// Install a written snapshot, enforce budget. Returns fileNames whose files the caller must delete.
    mutating func commit(_ plan: Plan, byteSize: Int, budgetBytes: Int) -> [String] {
        clock += 1
        var owned = Set<String>()
        for b in plan.boundaries {
            // A delta plan only ever holds NEW blocks (not in `byHash`), so the steal-from-previous
            // path is dead for it — the previous files keep their blocks. It stays for the legacy
            // whole-snapshot plan, which does re-point every block and thereby supersedes.
            if !plan.delta, let prev = byHash[b.hash], prev.fileName != plan.fileName {
                files[prev.fileName]?.ownedBoundaries.remove(b.hash)
            }
            byHash[b.hash] = Boundary(fileName: plan.fileName, tokenCount: b.tokenCount)
            owned.insert(b.hash)
        }
        files[plan.fileName] = FileRecord(
            fileName: plan.fileName,
            byteSize: byteSize,
            lastAccess: clock,
            ownedBoundaries: owned
        )
        totalBytes += byteSize
        var deleted: [String] = []
        // Orphan sweep only matters for whole-snapshot supersession; a delta write never empties a
        // previous file's boundaries. Guarding it keeps delta files (which are NOT supersets) intact.
        if !plan.delta {
            let orphans = files.filter { $0.value.ownedBoundaries.isEmpty }.map { $0.key }
            for name in orphans { deleted.append(drop(name)) }
        }
        while totalBytes > budgetBytes,
              let victim = files.values.min(by: { $0.lastAccess < $1.lastAccess }) {
            deleted.append(drop(victim.fileName))
        }

        return deleted
    }

    /// Bump LRU recency for every file in a reuse chain, so budget eviction can't drop an early delta
    /// file while a deeper one stays warm (which would punch a hole in the chain).
    mutating func touch(_ fileNames: [String]) {
        clock += 1
        for n in fileNames { files[n]?.lastAccess = clock }
    }

    /// Drop one file's entry — used by the store when a snapshot is found missing/corrupt on load
    /// (self-heal after an out-of-band delete), so the next `record` can replace it rather than being
    /// blocked by `planRecord` seeing the blocks "already in catalog".
    mutating func evict(_ name: String) { _ = drop(name) }

    private mutating func drop(_ name: String) -> String {
        if let r = files.removeValue(forKey: name) {
            for h in r.ownedBoundaries { byHash[h] = nil }
            totalBytes -= r.byteSize
        }
        return name
    }
    
    /// Load the persisted catalog, or start fresh. Wipes on signature/blockSize change or a
    /// missing/corrupt index (greenfield — no migration); drops entries whose snapshot file is gone.
    static func loadOrReset(directory: URL, signature: CacheSignature, blockSize: Int) -> Catalog {
        let url = directory.appendingPathComponent("index.json")
        // A pre-tag index.json has no `indexLayout` key, fails to decode, and drops to the wipe
        // branch — that decode failure IS the migration to the delta layout.
        guard let data = try? Data(contentsOf: url),
              var loaded = try? JSONDecoder().decode(Catalog.self, from: data),
              loaded.header.signature == signature,
              loaded.header.blockSize == blockSize,
              loaded.header.indexLayout == currentIndexLayout
        else {
            try? FileManager.default.removeItem(at: directory)                       // wipe + restart (no migration)
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            return Catalog(header: .init(signature: signature, blockSize: blockSize,
                                         indexLayout: currentIndexLayout))
        }
        let vanished = loaded.files.keys.filter {                                    // drop entries whose file is gone
            !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path)
        }
        for name in vanished { _ = loaded.drop(name) }
        return loaded
    }
}
