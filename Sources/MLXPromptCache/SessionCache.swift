import Foundation
import MLX
import MLXLMCommon

/// A held KV cache for the turns of ONE conversation. Holds a live `[KVCache]` in RAM and grows it
/// as turns are added — the last-turn state is never re-prefilled. Non-persisting (no disk, no
/// catalog): a conversation's suffix is private and transient (the eviction asymmetry). Hybrid-
/// native — the live cache carries attention AND recurrent (Mamba) layers; nothing is serialised.
/// Not `Sendable` (holds `[KVCache]`): create and use it on the model's thread, like `PreparedCache`.
public final class SessionCache {
    /// The live cache handed to `MLXLMCommon.generate(cache:)`. After each `generate` it holds the
    /// prompt + generated answer, so the next turn prefills only the new user text.
    public private(set) var cache: [KVCache]
    
    /// Seed the session. `warmRoot` is a `store.reuse(forTokens: rootTokens)` result — the durable
    /// `[preamble][file]` cache loaded from disk — so turn 1 skips re-prefilling the file. `nil` ⇒ start
    /// from a fresh (hybrid-correct) empty cache the first `advance` fills.
    public init(warmRoot: Reused?, makeCache: () -> [KVCache]) {
        self.cache = warmRoot?.cache ?? makeCache()
    }
    
    /// Prepare the next turn. `fullPromptTokens` is the whole templated conversation for this turn.
    /// Returns ONLY the tokens beyond what the held cache already contains; the caller passes them to
    /// `MLXLMCommon.generate(input:, cache: self.cache)`, which extends the held cache in place.
    public func advance(fullPromptTokens: [Int]) -> LMInput {
        let resident = PromptCacheIO.tokenLength(cache) ?? 0
        // clamp. a diverged prefix re-seeds via the store.
        let start = min(resident, fullPromptTokens.count)
        return LMInput(tokens: MLXArray(Array(fullPromptTokens[start...])))
    }
    
    /// Free the GPU/RAM when the conversation ends. Idempotent
    public func release() { cache = [] }
}
