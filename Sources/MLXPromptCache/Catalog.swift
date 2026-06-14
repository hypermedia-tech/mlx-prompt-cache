import Foundation

/// Persistent catalogue of what's on disk. Pure value type — `Codable` for restart. Holds no `KVCache`.
struct Catalog: Codable {
    struct Header: Codable { var signature: CacheSignature; var blockSize: Int }
    struct FileRecord: Codable { var fileName: String; var byteSize: Int; var lastAccess: Int
        var ownedBoundaries: Set<String> }
    struct Boundary: Codable { let fileName: String; let tokenCount: Int }
    
    var header: Header
    var byHash: [String: Boundary] = [:]
    var files: [String: FileRecord] = [:]
    var totalBytes = 0
    var clock = 0
    
    struct Hit: Sendable { let fileName: String; let matchedTokens: Int }
    struct Plan: Sendable { let fileName: String; let boundaries: [BoundaryPlan] }
    struct BoundaryPlan: Sendable { let hash: String; let tokenCount: Int }
    
    /// Longest matched prefix; touches LRU. Mutating because it bumps `lastAccess`.
    mutating func lookup(_ hashes: [BlockHash]) -> Hit? {
        var deepest: Boundary?
        for h in hashes { guard let b = byHash[h.hex] else { break }; deepest = b }
        guard let hit = deepest else { return nil }
        clock += 1; files[hit.fileName]?.lastAccess = clock
        return Hit(fileName: hit.fileName, matchedTokens: hit.tokenCount)
    }
    
    /// Plan a record, or nil if there's nothing new to cache.
    func planRecord(_ hashes: [BlockHash], blockSize: Int) -> Plan? {
        guard !hashes.isEmpty, hashes.contains(where: { byHash[$0.hex] == nil }) else { return nil }
        let name = "snap-\(hashes.last!.hex.prefix(32)).safetensors"
        let bounds = hashes.enumerated().map {
            BoundaryPlan(hash: $0.element.hex, tokenCount: ($0.offset + 1) * blockSize)
        }
        return Plan(fileName: name, boundaries: bounds)
    }
    
    /// Install a written snapshot, enforce budget. Returns fileNames whose files the caller must delete.
    mutating func commit(_ plan: Plan, byteSize: Int, budgetBytes: Int) -> [String] {
        clock += 1
        var owned = Set<String>()
        for b in plan.boundaries {
            if let prev = byHash[b.hash], prev.fileName != plan.fileName {
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
        let orphans = files.filter { $0.value.ownedBoundaries.isEmpty }.map { $0.key }
        for name in orphans { deleted.append(drop(name)) }
        while totalBytes > budgetBytes,
              let victim = files.values.min(by: { $0.lastAccess < $1.lastAccess }) {
            deleted.append(drop(victim.fileName))
        }
        
        return deleted
    }

    private mutating func drop(_ name: String) -> String {
        if let r = files.removeValue(forKey: name) {
            for h in r.ownedBoundaries { byHash[h] = nil }
            totalBytes -= r.byteSize
        }
        return name
    }
}
