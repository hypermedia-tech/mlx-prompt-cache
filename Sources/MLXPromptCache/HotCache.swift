import Foundation

/// In-RAM LRU index of snapshot bytes, keyed by snapshot file name. A pure `Sendable` value type —
/// held behind `Mutex<HotCache>` in the store, exactly like `Catalog`. Because every field is
/// `Sendable` (`Data`/`DType`/`[Int]`/`String`), there is no `@unchecked` and no non-`Sendable` state.
struct HotCache: Sendable {
    struct Entry: Sendable {
        let blobs: [CacheBytes]
        let fullTokens: Int
        let bytes: Int
        var lastAccess: Int
    }

    let budgetBytes: Int
    private(set) var totalBytes = 0
    private var entries: [String: Entry] = [:]
    private var clock = 0

    init(budgetBytes: Int) { self.budgetBytes = budgetBytes }

    var isEnabled: Bool { budgetBytes > 0 }

    /// Resident entry (bytes + full length) for `fileName`, touching LRU. The returned `Entry` is
    /// `Sendable`, so the caller reconstructs from it **outside** the lock.
    mutating func resident(_ fileName: String) -> Entry? {
        guard var e = entries[fileName] else { return nil }
        clock += 1; e.lastAccess = clock; entries[fileName] = e
        return e
    }

    /// Install bytes for a full-length snapshot. No-op if disabled, or if it can't fit the whole budget.
    mutating func insert(_ fileName: String, blobs: [CacheBytes], fullTokens: Int) {
        let size = HotCodec.footprint(blobs)
        guard isEnabled, size <= budgetBytes else { return }
        if let old = entries.removeValue(forKey: fileName) { totalBytes -= old.bytes }
        clock += 1
        entries[fileName] = Entry(blobs: blobs, fullTokens: fullTokens, bytes: size, lastAccess: clock)
        totalBytes += size
        while totalBytes > budgetBytes,
              let lru = entries.min(by: { $0.value.lastAccess < $1.value.lastAccess })?.key {
            totalBytes -= entries.removeValue(forKey: lru)!.bytes
        }
    }

    mutating func drop(_ fileName: String) {
        if let e = entries.removeValue(forKey: fileName) { totalBytes -= e.bytes }
    }

    mutating func clear() { entries.removeAll(); totalBytes = 0 }
}
