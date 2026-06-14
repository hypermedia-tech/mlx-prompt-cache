

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
