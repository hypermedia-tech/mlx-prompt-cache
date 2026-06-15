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
