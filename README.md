# MLXPromptCache

**Persistent prefix KV-cache for [mlx-swift](https://github.com/ml-explore/mlx-swift). Run the same context through many prompts and pay the prefill cost once — across a RAM hot tier and an SSD cold tier, even after a restart.**

![Swift Package Manager](https://img.shields.io/badge/SPM-compatible-brightgreen.svg)
![Swift](https://img.shields.io/badge/Swift-6.3-orange.svg)
![Platforms](https://img.shields.io/badge/platforms-macOS%2026%20·%20iOS%2026%20·%20tvOS%2026%20·%20visionOS%2026-blue.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey.svg)

---

When you prompt a local LLM, the model first **prefills** your prompt into a key/value attention cache before it generates a single token. For long prompts that prefill dominates time-to-first-token. If your prompts share a long prefix — the same document, system prompt, codebase, or log dump followed by a *different* question — `MLXPromptCache` lets you reuse the cached prefill and only process the new suffix.

mlx-swift-lm already gives you single-snapshot `savePromptCache` / `loadPromptCache` / `trimPromptCache` primitives. This package is the layer on top: a persistent, content-addressed catalog that finds the **longest cached prefix across many different prompts**, gates reuse on exact model identity, bounds disk with LRU eviction, and — when enabled — keeps the hottest snapshots in RAM so a repeat query skips the disk read entirely.

## When to use it

- **Document / log Q&A** — one big context, many questions. (The original target: a security workbench re-querying the same report.)
- **RAG or agent loops** with a large, stable system prompt.
- **Few-shot prompting** with a fixed exemplar block.
- Anything **long-prompt + short-output + repeated-prefix**.

If every prompt is unique end to end, there's no shared prefix to reuse and this won't help.

## How it works (30 seconds)

- **Block hashing.** Tokens are split into fixed-size blocks (default 256). Each block gets a *chained* SHA-256 digest: `hash(blockₙ) = SHA256(signature ++ hash(blockₙ₋₁) ++ tokens)`. A shared prefix produces identical leading hashes; any divergence only affects the blocks after it.
- **Signature gating.** A `CacheSignature` (model id + KV dtype + KV bits + build version) is folded into every hash *and* re-checked against snapshot metadata on load — so a cache is only ever reused for the exact model and quantization that produced it.
- **Two tiers, both LRU.** A `Codable` catalog maps block hashes → on-disk snapshots (the **cold/SSD tier**, bounded by `budgetBytes`). With `hotBudgetBytes > 0`, recently used snapshots are also held as raw bytes in an in-RAM **hot tier**; a repeat hit is reconstructed from RAM with no disk read. The hot tier is a strict accelerator — every resident snapshot also exists on disk, so RAM eviction is always lossless.

Reuse is **block-aligned**: only whole blocks are reusable, and the trailing partial block is always re-prefilled.

## Requirements

- **macOS 26 / iOS 26 / tvOS 26 / visionOS 26** or newer
- **Swift 6.3+** (Swift 6 language mode)
- Apple Silicon

These track [mlx-swift](https://github.com/ml-explore/mlx-swift)'s own baseline — MLX is not available on earlier OS versions.

## Installation

Add the package to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/hypermedia-tech/mlx-prompt-cache", from: "0.1.0"),
],
targets: [
    .target(
        name: "YourApp",
        dependencies: [
            .product(name: "MLXPromptCache", package: "mlx-prompt-cache"),
        ]
    ),
]
```

## Usage

The flow is two calls: **`reuse`** before generation to recover a cached prefix, and **`record`** after a fresh prefill to snapshot the reusable part for next time. The hot tier is transparent — `reuse` serves from RAM automatically when a snapshot is resident.

```swift
import MLX
import MLXLMCommon
import MLXPromptCache

// 1. Pin exactly which model + quantization this cache belongs to.
let signature = CacheSignature(
    modelId: "mlx-community/Qwen3-1.7B-4bit",
    kvDType: "bf16",
    kvBits: nil,
    buildVersion: "1"          // bump to invalidate every cache after a behaviour change
)

// 2. Open a store: an SSD budget, plus an optional RAM hot-tier budget (0 = disk only).
let store = try PromptCacheStore(
    directory: cacheDir,
    budgetBytes: 4_000_000_000,      // 4 GB of snapshots on disk
    signature: signature,
    blockSize: 256,
    hotBudgetBytes: 2_000_000_000    // 2 GB of snapshots kept hot in RAM
)

// `fullTokens` is your stable context + the new question.
let fullTokens = await modelContainer.encode(document + question)

try await modelContainer.perform { context in
    var inputTokens = fullTokens
    let cache: [KVCache]
    var wasHit = false

    // 3. Reuse the longest cached prefix, if any (RAM if resident, else disk). Prefill only the suffix.
    if let reused = store.reuse(forTokens: fullTokens) {
        inputTokens = Array(fullTokens[reused.matchedTokens...])
        cache = reused.cache
        wasHit = true
    } else {
        cache = makePromptCache(model: context.model, parameters: params)
    }

    // 4. Generate as usual against that cache.
    let stream = try MLXLMCommon.generate(
        input: LMInput(tokens: MLXArray(inputTokens)),
        cache: cache, parameters: params, context: context)
    for await _ in stream { /* collect chunks */ }

    // 5. On a cold prefill, snapshot the *stable* prefix (the document — not the question).
    if !wasHit {
        let prefixLen = document.tokenCount.blockAligned(to: 256)   // your block-aligned stable prefix
        try store.record(prefixTokens: Array(fullTokens.prefix(prefixLen)), cache: cache)
    }
}
```

To pre-warm a known working set at launch (skip even the first disk read), use `preload` instead of `record` — it writes the snapshot and pulls it straight into the RAM tier.

A complete, runnable end-to-end example — load a model, cold vs. warm, verify the outputs are identical, and print the speedup — lives in [`Sources/MLXPromptCacheScratch/main.swift`](Sources/MLXPromptCacheScratch/main.swift).

### API surface

| Symbol | Purpose |
| --- | --- |
| `PromptCacheStore(directory:budgetBytes:signature:blockSize:hotBudgetBytes:)` | Open/create a store. `hotBudgetBytes: 0` (default) = SSD only; `> 0` enables the RAM hot tier. |
| `store.reuse(forTokens:) -> Reused?` | Longest cached prefix, served from RAM if resident else disk, trimmed to the match. |
| `store.record(prefixTokens:cache:)` | Snapshot a freshly-prefilled prefix for future reuse (disk; warms RAM on the next reuse). |
| `store.preload(prefixTokens:cache:)` | Like `record`, but also pre-warms the RAM tier now — for launch-time warming. |
| `store.clearHot()` | Drop all RAM residents (e.g. on a model swap). Disk is untouched. |
| `Reused { cache: [KVCache]; matchedTokens: Int }` | The recovered cache and how many leading tokens it covers. |
| `CacheSignature(modelId:kvDType:kvBits:buildVersion:)` | Invalidation key — reuse is gated on an exact match. |

### Threading

`reuse`/`record`/`preload` are designed to run on the caller's thread inside `modelContainer.perform { … }`. The returned `[KVCache]` is **not** `Sendable` (it is GPU-backed); use it on the thread that received it and don't pass it across an isolation boundary. The store itself is `Sendable` and safe to share — its catalog and hot tier are `Sendable` value types behind mutexes, and all MLX/disk work runs outside the locks.

## Correctness

Reuse must never change what the model produces. The integration harness verifies this directly: it runs the same prompt **cold** (fresh prefill) and **warm** (reused prefix) under greedy decoding and asserts the outputs are byte-for-byte identical before reporting the speedup.

## Limitations

- **Block-aligned reuse** — the trailing partial block is always re-prefilled (default block size 256 tokens).
- **Hot tier serves full-prefix matches.** A repeat query over the same recorded context is served from RAM; a *partial* cross-prompt match (sharing only a leading run of a longer snapshot) is served from disk. A hot hit reconstructs a fresh private cache from the resident bytes.
- **Snapshots are large.** KV-cache for a long prefix can be hundreds of MB; both budgets are in bytes — size them accordingly. The RAM tier holds the same bytes the snapshot occupies on disk.
- **Sliding-window (rotating) caches are cold-only.** Models whose cache isn't uniformly trimmable (e.g. some sliding-window attention) fall back to a clean miss rather than incorrect reuse.

## Acknowledgements

MLXPromptCache is an independent Swift implementation and includes no source code from the
projects below — it exists because of their work:

- **[oMLX](https://github.com/jundot/omlx)** (Apache-2.0, by Jun Kim) — the persistent
  prompt/prefix KV-cache design this package ports to Swift: tiered hot/cold caching,
  block-chained prefix hashing, and signature-gated reuse all follow oMLX's approach.
- **[vllm-mlx](https://github.com/waybarrios/vllm-mlx)** — oMLX's own stated origin
  ("oMLX started from vllm-mlx v0.1.0"); the MLX port of vLLM's cache machinery.
- **[vLLM](https://github.com/vllm-project/vllm)** — the origin of PagedAttention and
  block-hash prefix KV caching (Kwon et al., 2023) that the whole lineage builds on.
- **[MLX](https://github.com/ml-explore/mlx) · [mlx-swift](https://github.com/ml-explore/mlx-swift)
  · [mlx-swift-lm](https://github.com/ml-explore/mlx-swift-lm)** by Apple — the array framework
  and Swift bindings this package builds on; the `savePromptCache` / `loadPromptCache` /
  `trimPromptCache` primitives it orchestrates come from mlx-swift-lm.

## License

[Apache-2.0](LICENSE) © 2026 Hypermedia Tech Pty Ltd
