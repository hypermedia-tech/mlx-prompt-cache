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
    /// The prompt each live session was last handed. Kept so the next turn can be checked against
    /// what the cache was actually built from, instead of being trusted on length alone.
    ///
    /// One `[Int]` per live conversation — a few hundred KB against a cache measured in gigabytes.
    private var presented: [UUID: [Int]] = [:]

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
    ) -> (input: LMInput, cache: [KVCache], divergedAt: Int?) {
        // CONGRUENCE GUARD. A held cache is only usable for a prompt that still carries the prompt
        // it was built from as a leading run. Without this check the delta below is cut by LENGTH
        // alone, so a prompt that has drifted from the cache produces exactly the numbers a correct
        // one does — same reused count, same delta size, no error — and the model reads the rest of
        // the conversation shifted by however far the drift went.
        //
        // Drift is not hypothetical and not the consumer's fault. A chat template decides how to
        // re-render a past assistant turn, and two templates in the same model family have been
        // observed disagreeing about it: one keeps a `<think></think>` marker on a previous answer,
        // the other drops it, four tokens. The cache was built with the marker; the next prompt
        // arrives without it. Nothing above this layer can see that.
        var divergedAt: Int?
        if let held = live[id], let last = presented[id] {
            let resident = PromptCacheIO.tokenLength(held) ?? 0
            // Only the span this store actually SAW can be compared. Between the last presented
            // prompt and the resident length sit the tokens the model generated, and nothing here
            // has those ids — that span is unverifiable by construction, not by omission.
            let checkable = min(last.count, resident, fullPromptTokens.count)
            var index = 0
            while index < checkable, last[index] == fullPromptTokens[index] { index += 1 }
            if index < checkable {
                divergedAt = index
            } else if fullPromptTokens.count < resident {
                // The prompt ran out before the cache did. The old behaviour clamped this to an
                // empty delta, and generating on an empty input traps inside MLX's C++ reshape —
                // a process kill, not a throw.
                divergedAt = fullPromptTokens.count
            }
            if divergedAt != nil {
                live[id] = nil
                presented[id] = nil
            }
        }
        let cache: [KVCache]
        if let existing = live[id] {
            cache = existing
        } else {
            cache = warmRoot()?.cache ?? makeCache()
            live[id] = cache
        }
        presented[id] = fullPromptTokens
        let resident = PromptCacheIO.tokenLength(cache) ?? 0
        let start = min(resident, fullPromptTokens.count)
        return (LMInput(tokens: MLXArray(Array(fullPromptTokens[start...]))), cache, divergedAt)
    }

    /// Free the GPU/RAM for one conversation. Idempotent. Dropping the store's only long-lived reference
    /// to the `[KVCache]` releases the Metal buffers via ARC. Call inside `perform` (same discipline).
    package func release(_ id: UUID) { live[id] = nil; presented[id] = nil }
    
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
