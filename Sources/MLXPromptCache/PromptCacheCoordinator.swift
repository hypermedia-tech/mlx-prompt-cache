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
public final class PromptCacheCoordinator: Sendable {
    private let store: PromptCacheStore

    public init(store: PromptCacheStore) { self.store = store }

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
