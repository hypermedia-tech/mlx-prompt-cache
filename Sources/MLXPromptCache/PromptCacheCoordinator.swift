import Foundation
import MLX
import MLXLMCommon

/// What `prepare` hands back. Plain struct (holds non-`Sendable` `[KVCache]`); built and consumed on the
/// caller's thread inside the model's `perform`, never sent across an isolation boundary — same rule as `Reused`.
public struct PreparedCache {
    /// A generation-ready cache: a restored prefix (hit) or a freshly-captured one (miss).
    public let cache: [KVCache]
    /// The caller generates `promptTokens[suffixStart...]`. `0` means generate the whole prompt.
    public let suffixStart: Int
    public let outcome: PromptCacheOutcome
}

/// Why `prepare` returned what it did — for the consumer's logs/metrics. `Sendable` (pure values).
public enum PromptCacheOutcome: Sendable {
    case hit(matched: Int)              // restored a cached prefix of this many tokens
    case captured(boundary: Int)        // snapshotted a new prefix at this block boundary
    case uncacheable(reason: String)    // prompt < one block, pure-SSM, or the prefill didn't land
}

/// Why `warm` returned what it did. `Sendable` (pure values).
public enum PromptWarmOutcome: Sendable {
    /// The prefix is cached to its last full block. `prefilled == 0` means the catalog already
    /// held it (a peek-only no-op — no snapshot IO, no GPU).
    case complete(cachedTokens: Int, prefilled: Int)
    /// `shouldPause` fired mid-prefill. The completed full blocks are recorded; calling `warm`
    /// again with the same tokens resumes from them (longest-prefix reuse IS the resume token).
    case paused(cachedTokens: Int)
    case uncacheable(reason: String)
}

/// Drives a `PromptCacheStore` against a live model: reuse-vs-capture, prefill, snapshot. This is the whole
/// prompt-cache capability — a consumer calls `prepare` then generates the returned suffix; attention vs
/// hybrid (Mamba/SSM) is handled here, never by the consumer.
///
/// `Sendable` (holds only the `Sendable` store), so it can be captured into a model `perform` closure.
/// `prepare` runs synchronously on the caller's thread, so the `[KVCache]` it returns never leaves it.
/// Proof that the holder is executing inside `ModelContainer.perform`. Every `PromptCacheCoordinator`
/// door that mutates the live-cache stores (`SessionStore`/`WarmStore`) requires one. External callers
/// cannot construct it — the initialiser is `package` — so the ONLY way to obtain one is
/// `PromptCacheCoordinator.scope(_:)`, which needs a `ModelContext` and therefore can only be called from
/// inside a `perform` block. A cache-mutating call from off the model queue is a compile error, not a
/// runtime race. `~Escapable` so it cannot be stashed and reused after the `perform` returns.
public struct PerformScope: ~Escapable {
    @lifetime(immortal)
    package init() {}
}

public final class PromptCacheCoordinator: Sendable {
    private let store: PromptCacheStore

    public init(store: PromptCacheStore) { self.store = store }

    /// Mint a `PerformScope`. Requires the `ModelContext` handed to a `ModelContainer.perform` closure —
    /// which is non-`Sendable` and cannot escape that closure — so a scope can only be created inside
    /// `perform`. This is the single door to the cache-mutating API's proof-of-context.
    @lifetime(borrow context)
    public func scope(_ context: borrowing ModelContext) -> PerformScope { PerformScope() }

    /// Reuse the longest cached prefix for `promptTokens`, or capture a fresh snapshot at the last full
    /// block boundary. Always returns a usable cache; `suffixStart` is where the caller begins generation.
    public func prepare(
        promptTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> PreparedCache {
        // 1. Reuse — works for attention AND hybrid (the snapshot is restored whole, never sliced).
        if let reused = store.reuse(forTokens: promptTokens), reused.matchedTokens > 0 {
            return PreparedCache(
                cache: reused.cache,
                suffixStart: reused.matchedTokens,
                outcome: .hit(matched: reused.matchedTokens)
            )
        }

        // 2. Miss → fresh cache. Capture at the last FULL block (rounded down; the -1 keeps ≥1 suffix token).
        let cache = makePromptCache(model: model, parameters: parameters)
        let boundary = max(0, (promptTokens.count - 1) / store.blockSize) * store.blockSize
        guard boundary > 0 else {
            return PreparedCache(
                cache: cache,
                suffixStart: 0,
                outcome: .uncacheable(reason: "prompt < one \(store.blockSize)-block")
            )
        }

        // 3. Prefill to the boundary (no sampling); verify it landed before trusting `suffixStart`.
        prefillOnly(Array(promptTokens[0 ..< boundary]), into: cache,
                    model: model, stepSize: parameters.prefillStepSize)
        guard PromptCacheIO.tokenLength(cache) == boundary else {
            return PreparedCache(
                cache: makePromptCache(model: model, parameters: parameters),
                suffixStart: 0,
                outcome: .uncacheable(reason: "prefill did not reach boundary")
            )
        }

        // 4. Snapshot at the boundary. `record` logs its own SKIP for a non-cacheable cache (e.g. pure-SSM).
        try? store.record(prefixTokens: Array(promptTokens[0 ..< boundary]), cache: cache)
        return PreparedCache(cache: cache, suffixStart: boundary, outcome: .captured(boundary: boundary))
    }

    /// Prefill `tokens` into `cache` WITHOUT sampling, leaving it at `offset == tokens.count`. Mirrors
    /// `TokenIterator`'s prefill (`model.prepare` + the trailing chunk) minus the sampled token, so it works
    /// for every model — including ones whose `callAsFunction(_:MLXArray,cache:)` is unimplemented.
    private func prefillOnly(
        _ tokens: [Int],
        into cache: [KVCache],
        model: any LanguageModel,
        stepSize: Int
    ) {
        let input = LMInput(tokens: MLXArray(tokens))
        guard let result = try? model.prepare(input, cache: cache, windowSize: stepSize) else { return }
        if case let .tokens(remaining) = result {
            _ = model(remaining[text: .newAxis], cache: cache, state: nil)
        }
        eval(cache.flatMap { $0.state })
    }
}

extension PromptCacheCoordinator {
    /// Warm a prefix with no generation: probe the catalog (free), resume from any partial
    /// match, prefill the remainder in block-aligned chunks — checking `shouldPause` between
    /// chunks — and record. The pause probe is how a consumer keeps a long background prefill
    /// from holding the GPU against interactive work: pause, let the interactive turn run,
    /// call `warm` again.
    public func warm(
        promptTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters,
    shouldPause: () -> Bool = { false }
    ) -> PromptWarmOutcome {
        let boundary = max(0, (promptTokens.count - 1 ) / store.blockSize) * store.blockSize
        guard boundary > 0 else {
            return .uncacheable(reason: "prompt < one \(store.blockSize)-block")
        }
        
        // 1. Catalog-only idempotence probe, the already-warm path costs no snapshot IO (peek).
        if store.peek(forTokens: promptTokens) >= boundary {
            return .complete(cachedTokens: boundary, prefilled: 0)
        }
        
        // 2. Resume from the longest cached prefix, or start cold.
        let reused = store.reuse(forTokens: promptTokens)
        let start = reused?.matchedTokens ?? 0
        let cache = reused?.cache ?? makePromptCache(model: model, parameters: parameters)
        
        // 3. Chunked prefill start → boundary; pause points land on block boundaries so the
        //    paused prefix is recordable (and hybrid/Mamba caches sit exactly at a boundary,
        //    never needing the trim that record-after-generate can't do).
        let reached = prefillChunked(
            Array(promptTokens[0 ..< boundary]),
            into: cache,
            from: start,
            model: model,
            stepSize: parameters.prefillStepSize,
            shouldPause: shouldPause
        )
        guard reached > start, PromptCacheIO.tokenLength(cache) == reached else {
            return .uncacheable(reason: "prefill did not advance")
        }
        
        // 4. Record what we have. `record` skips already-catalogued blocks itself.
        try? store.record(prefixTokens: Array(promptTokens[0 ..< reached]), cache: cache)
        return reached == boundary
        ? .complete(cachedTokens: boundary, prefilled: boundary - start)
        : .paused(cachedTokens: reached)
    }
    
    /// Prefill `tokens[from...]` into `cache` chunk by chunk (each chunk a block-multiple of
    /// roughly `stepSize`), checking `shouldPause` between chunks. Per chunk this is exactly
    /// `prefillOnly`'s model-agnostic mechanic (`model.prepare` + trailing piece + eval) —
    /// continuing a cache mid-prefix is the same operation as the suffix prefill the reuse
    /// path already proves. Returns the offset reached.
    private func prefillChunked(
        _ tokens: [Int],
        into cache: [KVCache],
        from start: Int,
        model: any LanguageModel,
        stepSize: Int,
        shouldPause: () -> Bool
    ) -> Int {
        let chunk = max(store.blockSize, (stepSize / store.blockSize) * store.blockSize)
        var offset = start
        while offset < tokens.count {
            if offset > start, shouldPause() { return offset }
            let end = min(offset + chunk, tokens.count)
            let piece = LMInput(tokens: MLXArray(Array(tokens[offset ..< end])))
            guard let result = try? model.prepare(piece, cache: cache, windowSize: stepSize) else {
                return offset
            }
            if case let .tokens(remaining) = result {
                _ = model(remaining[text: .newAxis], cache: cache, state: nil)
            }
            eval(cache.flatMap { $0.state })
            offset = end
        }
        return offset
    }
}

extension PromptCacheCoordinator {

    /// Warm a prefix while **holding the live cache across pauses**, so a resume neither reloads the
    /// prefix nor rewrites it. The residency twin of `warm(promptTokens:model:parameters:shouldPause:)`.
    ///
    /// `id` names this warm; pass the same `id` after a `.paused` to resume from the held cache with
    /// no snapshot IO at all. If `id` is not resident — first call, released, evicted, fresh process —
    /// this falls back to **exactly** the disk behaviour of the original `warm`: peek, reuse the
    /// longest cached prefix, continue. Residency is an accelerator, never a source of truth.
    ///
    /// A held cache is only reused when the caller's tokens hash to the frontier recorded when it was
    /// held. A mismatch is declined, not extended — extending a diverged cache and then recording it
    /// would poison the catalog with a snapshot whose contents do not match its chain hash.
    ///
    /// Call inside `ModelContainer.perform` (see `WarmStore`'s type invariant).
    public func warm(
        _ warms: WarmStore,
        id: UUID,
        promptTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters,
        scope: borrowing PerformScope,
        persist: WarmPersistence = .onCompletion,
        shouldPause: () -> Bool = { false }
    ) -> PromptWarmOutcome {
        let boundary = max(0, (promptTokens.count - 1) / store.blockSize) * store.blockSize
        guard boundary > 0 else {
            return .uncacheable(reason: "prompt < one \(store.blockSize)-block")
        }

        // 1. Resident? Verify the held cache covers THESE tokens before trusting it.
        var cache: [KVCache]
        var start: Int
        if let heldTokens = warms.heldTokenCount(id),
           heldTokens <= promptTokens.count,
           let resumed = warms.resume(id, expecting: store.frontierHash(forTokens: promptTokens,
                                                                       upTo: heldTokens)) {
            cache = resumed.cache
            start = resumed.tokens
        } else {
            // 2. Not resident → the original path, unchanged.
            if store.peek(forTokens: promptTokens) >= boundary {
                return .complete(cachedTokens: boundary, prefilled: 0)
            }
            let reused = store.reuse(forTokens: promptTokens)
            start = reused?.matchedTokens ?? 0
            cache = reused?.cache ?? makePromptCache(model: model, parameters: parameters)
        }

        guard start < boundary else {
            // Already at the boundary: persist if we are holding unpersisted work, then finish.
            return finishWarm(warms, id: id, model: model, scope: scope)
        }

        // 3. Chunked prefill, pausing on block boundaries so a hybrid's recurrent state always sits
        //    exactly at a boundary — the only shape its snapshot is valid in.
        let reached = prefillChunked(
            Array(promptTokens[0 ..< boundary]),
            into: cache,
            from: start,
            model: model,
            stepSize: parameters.prefillStepSize,
            shouldPause: shouldPause
        )
        guard reached > start, PromptCacheIO.tokenLength(cache) == reached else {
            return .uncacheable(reason: "prefill did not advance")
        }

        let prefix = Array(promptTokens[0 ..< reached])
        warms.hold(id, cache: cache, prefix: prefix,
                   frontier: store.frontierHash(forTokens: promptTokens, upTo: reached))

        // 4. Persist per policy — NOT on every pause. A save holds a process-global MLX lock for its
        //    whole duration, so doing it per pause stalls the interactive work the pause exists for.
        let done = reached == boundary
        let sinceLast = reached - (warms.entry(id)?.persistedTokens ?? 0)
        let shouldWrite: Bool
        switch persist {
        case .never: shouldWrite = false
        case .onCompletion: shouldWrite = done
        case let .everyTokens(k): shouldWrite = done || (k > 0 && sinceLast >= k)
        }
        if shouldWrite { persistHeld(warms, id: id) }

        // 5. Budget: persist-then-release the largest other warms if we are over.
        for victim in warms.victimsOverBudget(excluding: id) {
            persistHeld(warms, id: victim)
            warms.release(victim, scope: scope)
        }

        if done {
            // Give the memory back ONLY when the work is safely on disk. Releasing unconditionally
            // would mean `.never` completes, writes nothing, and silently discards the whole warm —
            // so under `.never` the cache stays resident and the caller must `finishWarm` (persist
            // and free) or `release` (discard deliberately).
            if shouldWrite { warms.release(id, scope: scope) }
            return .complete(cachedTokens: boundary, prefilled: boundary - start)
        }
        return .paused(cachedTokens: reached)
    }

    /// Persist whatever `id` has reached and release it. Call when abandoning a warm that will not be
    /// resumed — tab closed, file deselected, memory pressure — so the work is not thrown away.
    /// Idempotent; a `nil`/absent id is a no-op.
    ///
    /// Takes `model` it does not use, deliberately: `model` is only reachable via `context.model`
    /// inside `ModelContainer.perform`, which is where this must be called. Same nudge as `advance`.
    @discardableResult
    public func finishWarm(
        _ warms: WarmStore,
        id: UUID,
        model: any LanguageModel,
        scope: borrowing PerformScope
    ) -> PromptWarmOutcome {
        guard let e = warms.entry(id) else { return .uncacheable(reason: "no warm held for id") }
        persistHeld(warms, id: id)
        warms.release(id, scope: scope)
        return .complete(cachedTokens: e.prefix.count, prefilled: 0)
    }

    /// The live cache held for `id`, for generating directly off a warm without a disk round trip.
    /// `nil` if nothing is held. Requires `model` for the same reason as `finishWarm`: this hands
    /// back a non-`Sendable` `[KVCache]` that must not leave the `perform` it was obtained in.
    public func heldCache(
        _ warms: WarmStore,
        id: UUID,
        model: any LanguageModel,
        scope: borrowing PerformScope
    ) -> [KVCache]? {
        warms.entry(id)?.cache
    }

    /// Write the held prefix for `id` to disk, if there is unpersisted work. `record` refuses cleanly
    /// (logging its own SKIP) for a hybrid off a boundary or a pure-SSM model, so this is best-effort
    /// by design — the same contract the non-resident `warm` already has.
    private func persistHeld(_ warms: WarmStore, id: UUID) {
        guard let e = warms.entry(id), e.prefix.count > e.persistedTokens else { return }
        try? store.record(prefixTokens: e.prefix, cache: e.cache)
        warms.markPersisted(id, tokens: e.prefix.count)
    }
}

extension PromptCacheCoordinator {
    /// Consumer-facing turn driver — the only public door to the live caches. Requires a `PerformScope`
    /// (obtainable ONLY inside `perform`, see `scope(_:)`), so a call from off the model queue does not
    /// compile — the enforced form of the old "pass `model` as a nudge". Seeds conversation `id` from the
    /// durable disk root (`store.reuse(forTokens: rootTokens)`) on the first turn; thereafter the held
    /// cache is extended in place, never reloaded.
    public func advance(
        _ sessions: SessionStore,
        id: UUID,
        fullPromptTokens: [Int],
        rootTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters,
        scope: borrowing PerformScope
    ) -> (input: LMInput, cache: [KVCache]) {
        sessions.advance(
            id: id,
            fullPromptTokens: fullPromptTokens,
            warmRoot: { store.reuse(forTokens: rootTokens) },
            makeCache: { makePromptCache(model: model, parameters: parameters) }
        )
    }

    /// Free conversation `id`'s live cache. Idempotent. The `PerformScope` gates this to inside `perform`.
    public func release(_ sessions: SessionStore, id: UUID, scope: borrowing PerformScope) {
        sessions.release(id)
    }

    /// Evict the largest held sessions over an APP-SUPPLIED byte budget, keeping `keep`. Unlike the warm-side
    /// budget (which `WarmStore` stores at init), the session budget is passed in — the app owns it, resolving
    /// live system RAM. NO persist-before-release: a session's durable source is the day-chunked log
    /// (reassemble on next resume), so eviction just drops RAM. Idempotent. The `PerformScope` gates this to
    /// inside `perform` — the same domain as `advance`/`release`, now enforced by the type system.
    public func evictSessions(
        _ sessions: SessionStore,
        overBudget budgetBytes: Int,
        keep: UUID,
        scope: borrowing PerformScope
    ) {
        for id in sessions.victimsOverBudget(budgetBytes, excluding: keep) { sessions.release(id) }
    }
}
