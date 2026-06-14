import Foundation

/// Persistent prompt/prefix KV cache for mlx-swift.
/// Design + full API: `mlx-prompt-cache-module-brief.md`. This is the compiling shell;
/// `reuse` / `record` / the on-disk index land in the implementation pass (where the
/// `MLXLMCommon` import + KVCache handling come in).
public actor PromptCacheStore {
    private let directory: URL
    private let budgetBytes: Int
    private let signature: CacheSignature
    
    public init(directory: URL, budgetBytes: Int, signature: CacheSignature) {
        self.directory = directory
        self.budgetBytes = budgetBytes
        self.signature = signature
    }
}
