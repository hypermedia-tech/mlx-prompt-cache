import Foundation
import MLX
import MLXLMCommon

/// Owns the live KV caches for in-flight conversations, keyed by id. The live `[KVCache]` for a
/// conversation is created, grown, and freed entirely inside this type — nothing non-`Sendable` is
/// stored by, or handed for retention to, the consumer.
///
/// `@unchecked Sendable` invariant: `live` (and every `[KVCache]` in it) is only ever read or mutated
/// inside `ModelContainer.perform`, which serialises all model access (via `SerialAccessContainer` /
/// `AsyncMutex`). There is never concurrent access to the map, so the data race `Sendable` guards against
/// cannot occur. This mirrors mlx-swift-lm's own `SerialAccessContainer<T>: @unchecked Sendable`, which
/// wraps the non-`Sendable` `ModelContext` the same way. A `Mutex` is deliberately NOT used: it would add
/// a second access path reachable off `perform` and defeat the single-serialised-domain guarantee.
///
/// The raw entry points are `package`: reachable by the coordinator seam and the package's own tests,
/// never by external dependents (who use the `public` `PromptCacheCoordinator` doors).
public final class SessionStore: @unchecked Sendable {
    private var live: [UUID: [KVCache]] = [:]

    public init() {}

    /// Advance conversation `id` by one turn. Seeds on the FIRST call for `id` — from the durable disk
    /// root (`warmRoot`) if present, else a fresh (hybrid-correct) empty cache from `makeCache`. Returns
    /// ONLY the tokens beyond the cache's resident offset, plus the live cache to generate over.
    /// `warmRoot`/`makeCache` are evaluated at most once (seed only) and never on a resumed turn.
    /// Call only inside `ModelContainer.perform` (see the type invariant).
    package func advance(
        id: UUID,
        fullPromptTokens: [Int],
        warmRoot: () -> Reused?,
        makeCache: () -> [KVCache]
    ) -> (input: LMInput, cache: [KVCache]) {
        let cache: [KVCache]
        if let existing = live[id] {
            cache = existing
        } else {
            cache = warmRoot()?.cache ?? makeCache()
            live[id] = cache
        }
        let resident = PromptCacheIO.tokenLength(cache) ?? 0
        let start = min(resident, fullPromptTokens.count)   // clamp; a diverged prefix yields an empty delta
        return (LMInput(tokens: MLXArray(Array(fullPromptTokens[start...]))), cache)
    }

    /// Free the GPU/RAM for one conversation. Idempotent. Dropping the store's only long-lived reference
    /// to the `[KVCache]` releases the Metal buffers via ARC. Call inside `perform` (same discipline).
    package func release(_ id: UUID) { live[id] = nil }
    
    package var residentBytes: Int {
        live.values.reduce(0) {
            $0 + WarmStore.footprint($1)
        }
    }
    /// Ids to drop when resident bytes exceed the budget, largest-first (size policy, not LRU).
    package func victimsOverBudget(
        _ budgetBytes: Int,
        excluding keep: UUID
    ) -> [UUID] {
        guard budgetBytes > 0, residentBytes > budgetBytes else { return [] }
        var over = residentBytes - budgetBytes;
        var out: [UUID] = []
        for (id, cache) in live.sorted(
            by: {
                WarmStore.footprint($0.value) > WarmStore.footprint($1.value)
            }
        )
        where id != keep {
            out.append(id);
            over -= WarmStore.footprint(cache);
            if over <= 0 { break }
        }
        return out
    }
}
