import Foundation
import MLX
import MLXLMCommon

/// Owns the live KV caches for in-flight background *warms*, keyed by id — the sibling of
/// `SessionStore`, which does the same for conversations.
///
/// Without this, `PromptCacheCoordinator.warm` builds its `[KVCache]` as a local and drops it on
/// return, so every resume must reload the whole prefix from disk and write it back. Holding the
/// cache across pauses turns both of those from O(prefix × resumes) into O(prefix).
///
/// `@unchecked Sendable` invariant — identical to `SessionStore`'s (see that type for the full
/// argument): `live` and every `[KVCache]` in it are only ever read or mutated inside
/// `ModelContainer.perform`, which serialises all model access, so the data race `Sendable` guards
/// against cannot occur. A `Mutex` is deliberately NOT used: it would add a second access path
/// reachable off `perform` and defeat the single-serialised-domain guarantee.
///
/// Residency is a **strict accelerator**. Every held prefix is either already on disk or will be
/// persisted before release, and an id that is absent — fresh process, released, evicted — makes
/// `warm` fall back to exactly its pre-existing disk behaviour. Losing a resident cache costs one
/// re-prefill, never a wrong answer.
public final class WarmStore: @unchecked Sendable {

    package struct Entry {
        package var cache: [KVCache]
        /// The block-aligned prefix this cache covers. Held so `finishWarm` can record without the
        /// caller re-supplying it.
        package var prefix: [Int]
        /// Chain hash of `prefix` — the divergence guard. A resume whose tokens do not hash to this
        /// is declined rather than extended.
        package var frontier: BlockHash?
        /// Tokens covered at the last successful persist, for `.everyTokens` cadence.
        package var persistedTokens: Int
        package var bytes: Int
    }

    private var live: [UUID: Entry] = [:]

    /// Soft cap on resident KV bytes. `0` disables the cap. Exceeding it does not drop work —
    /// `PromptCacheCoordinator` persists a victim before releasing it.
    public let budgetBytes: Int

    /// `budgetBytes: 0` means unbounded — appropriate only when the consumer bounds concurrency
    /// itself. A warm holds roughly `perTokenKVBytes × tokens`; for a 35B-class hybrid that is
    /// ~20 KiB/token, so a 183k-token document is ~3.7 GB.
    public init(budgetBytes: Int = 0) {
        self.budgetBytes = budgetBytes
    }

    // MARK: - Package surface (the coordinator is the public door)

    /// The held cache for `id`, but only if `expecting` matches the frontier hash recorded when it
    /// was held. A mismatch means the caller's tokens are not the tokens this cache covers; we
    /// decline so the caller falls back to disk, rather than extending the wrong cache and then
    /// recording it under a chain hash that does not describe its contents.
    package func resume(_ id: UUID, expecting: BlockHash?) -> (cache: [KVCache], tokens: Int)? {
        guard let e = live[id], e.frontier == expecting else { return nil }
        return (e.cache, e.prefix.count)
    }

    /// Token count and frontier hash held for `id` — lets the coordinator recompute the expected
    /// hash over the caller's own tokens before committing to a resume.
    package func heldTokenCount(_ id: UUID) -> Int? { live[id]?.prefix.count }

    package func hold(_ id: UUID, cache: [KVCache], prefix: [Int], frontier: BlockHash?) {
        let persisted = live[id]?.persistedTokens ?? 0
        live[id] = Entry(cache: cache, prefix: prefix, frontier: frontier,
                         persistedTokens: persisted, bytes: Self.footprint(cache))
    }

    package func entry(_ id: UUID) -> Entry? { live[id] }

    package func markPersisted(_ id: UUID, tokens: Int) {
        live[id]?.persistedTokens = tokens
    }

    /// Ids to evict when resident bytes exceed `budgetBytes`, **largest first**. `WarmStore` tracks
    /// no recency, so this is a size-based policy, not an LRU — it frees the most memory in the
    /// fewest evictions. Empty when unbounded or under budget. Nothing is discarded: the coordinator
    /// persists each victim before releasing it.
    package func victimsOverBudget(excluding keep: UUID) -> [UUID] {
        guard budgetBytes > 0, residentBytesUnchecked > budgetBytes else { return [] }
        var over = residentBytesUnchecked - budgetBytes
        var out: [UUID] = []
        for (id, e) in live.sorted(by: { $0.value.bytes > $1.value.bytes }) where id != keep {
            out.append(id)
            over -= e.bytes
            if over <= 0 { break }
        }
        return out
    }

    // MARK: - Public lifecycle

    /// Free one warm's live cache. Idempotent. **Does not persist** — use
    /// `PromptCacheCoordinator.finishWarm` to keep the work. The `PerformScope` gates this to inside `perform`.
    public func release(_ id: UUID, scope: borrowing PerformScope) { live[id] = nil }

    /// Free every held cache (model swap, memory pressure, shutdown). The `PerformScope` gates this to inside
    /// `perform` — the "memory pressure" caller must route through the model queue, not a bare handler.
    public func releaseAll(scope: borrowing PerformScope) { live.removeAll() }

    public func heldIds(scope: borrowing PerformScope) -> [UUID] { Array(live.keys) }
    public func isEmpty(scope: borrowing PerformScope) -> Bool { live.isEmpty }
    public func residentBytes(scope: borrowing PerformScope) -> Int { residentBytesUnchecked }

    /// Internal, ungated byte sum for the type's OWN `perform`-confined methods (e.g. `victimsOverBudget`),
    /// which already run inside the serialised domain and so need no witness. The public gate is
    /// `residentBytes(scope:)`.
    private var residentBytesUnchecked: Int { live.values.reduce(0) { $0 + $1.bytes } }

    /// Bytes of KV state a cache holds. Reads `state`, which for an attention layer is already
    /// sliced to `offset`, so this is the live footprint rather than the allocated capacity.
    package static func footprint(_ cache: [KVCache]) -> Int {
        cache.reduce(0) { total, layer in
            total + layer.state.reduce(0) { $0 + $1.nbytes }
        }
    }
}

/// When a warm writes a snapshot to disk.
///
/// The distinction matters because a persist holds a **process-global** MLX evaluation lock for the
/// whole write — measured 55–331 ms for 0.4–1.9 GB snapshots — during which no tensor evaluation
/// anywhere in the process can proceed. Doing that on every pause interleaves the stall with exactly
/// the interactive work the pause exists to let through.
public enum WarmPersistence: Sendable, Equatable {
    /// Never write. The held cache is the only copy, so a completed warm stays **resident** and the
    /// caller must call `finishWarm` (persist and free) or `WarmStore.release` (discard) — otherwise
    /// completion would silently throw the whole warm away. Progress is lost if the process dies.
    case never
    /// Write when the warm reaches its boundary, and on `finishWarm`. The default.
    case onCompletion
    /// Also write once at least this many tokens have accrued since the last write. Keyed on
    /// **tokens**, never on pause count — keying on pauses re-couples durability to interactive
    /// activity, which is the behaviour this replaces.
    case everyTokens(Int)
}
