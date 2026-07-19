# Warm-loop residency — measured proposal

**Status:** proposal, measured. The harness (`MLXPromptCacheBench`) is written, builds clean, and has
run on all three models. No library source has been changed.

Supersedes the analysis half of `hybrid-block-storage-and-warm-io.md`; that document's answer on
block storage still stands (§8 below).

---

## 0. Provenance

- Measurements: `MLXPromptCacheBench` run on 137 GB M-series, 190 GB free disk, thermal nominal,
  T=16,384 tokens, R=8 resumes, block 256, all three models in one invocation.
- `A` and `M` are **predicted from each model's `config.json` before any weights load**, then
  compared against measurement. A fit over file sizes checked against a fit over the same file sizes
  would be an arithmetic identity; this is not that.
- Swift and mlx-swift claims are read directly with `file:line`. oMLX claims are marked where they
  appear and were gathered by subagents against `~/workspace/learning/omlx`.

---

## 1. What the harness proved

**H1 — the byte model, confirmed.** `bytes(N) = A·N + M`.

- `Qwen3-1.7B` measured **114,689 B/token** at two independent prefix lengths against **114,688**
  predicted from config (28 layers × 2 × 8 kvHeads × 128 headDim × 2 B). One part in 10⁵; the odd
  byte is the safetensors header amortised.
- `Qwen3.6-35B-A3B` predicted A = 20,480 (10 attention layers × 2 × 2 × 256 × 2). Measured M =
  **66.1 MB** against a **62.9 MB** SSM-only prediction (30 linear × 32×128×128 × 4 B fp32); the
  3.2 MB residual is ≈107 kB/layer of conv state, exactly the term the prediction omits.
- `Qwen3.5-9B` measured M = 54.3 MB against 50.3 MB predicted. Same shape.
- The reported `G1.A` failure (+2.08% on all three, identically) is a **harness defect, not a
  refutation**: the largest fit point asked for `prefix(n+1)` where the corpus held only `n`, so
  `warm` recorded 16,128 tokens while the point was labelled 16,384. `(16128−4096)/(16384−4096)` =
  0.979 → +2.13%. Fixed; re-run will show it green.

So the two constants that govern everything are architectural, not empirical, and can be computed
for any future model from its config before it is ever loaded.

**H2 — the disease, confirmed as counts, not curves.** On every model, every resume after the first
performed exactly one whole-prefix load, and every resume performed exactly one whole-prefix save.
The 35B write series is exactly linear in the prefix: 106.3 → 148.3 → 190.2 → 232.2 → 274.1 → 316.1
→ 358.0 → 394.7 MB, each step +41.9 MB = 2048 tokens × 20,480 B. The mechanism is
`PromptCacheIO.save` writing unconditionally (PromptCacheIO.swift:68) plus `Catalog.planRecord`
re-planning **every** boundary rather than only new ones (Catalog.swift:33-35).

**H3 — residency, confirmed.** Zero snapshot reads across all prefill rounds, exactly one snapshot
written, on all three models. 35B: 2.02 GB → 394.7 MB. 9B: 2.82 GB → 580 MB. 1.7B: 8.43 GB → 1.85 GB.

**H4 — lossless, confirmed.** Generating **directly off the held cache**, with no persist and no
reload, is token-identical to a cold no-cache prefill on all three models. This is the arm that
de-risks the actual change; the persist-and-reload arms only exercise the snapshot codec, which
already has tests behind it.

A bonus finding from the control: prefilling in 256-token versus 512-token chunks produces
**bit-identical state — maxΔ = 0 across all 60 recurrent and 20 attention tensors** on the 35B, and
the same on the 9B. The GatedDeltaNet recurrent scan is invariant to chunk boundary. That is a
genuine result and it materially de-risks residency, because it says the thing we were most afraid
of perturbing does not perturb.

**H5 — the persist cost, measured.**

- 35B (395 MB snapshot): `record` 54 ms, then `F_FULLFSYNC` 108 ms.
- 9B (580 MB): 87 ms + 147 ms.
- 1.7B (1.85 GB): 288 ms + 77 ms. That is ~6.4 GB/s — memcpy speed, confirming the write never
  touches the device synchronously.
- Process-global eval-lock canary max stall: **260 / 341 / 285 ms**. The lock is free at the median
  (0.00 ms) but a background warm can stall *all other MLX work in the process* for a third of a
  second at a stroke.

---

## 2. Three corrections to the earlier proposal

**(a) The read half of the disease is largely not real, on this machine.** Device reads were
208.9 kB against 1.63 GB of logical reads on the 35B; 5.7 MB against 6.58 GB on the 1.7B. Nothing in
mlx-swift's writer fsyncs (`io/load.h` has no `fsync`, `F_FULLFSYNC`, `F_NOCACHE` or `rename`), so a
snapshot written seconds earlier is still entirely resident and the re-read is served from RAM.

My earlier "~17× round-trip" figure counted those reads as saved cost. They were not being paid.
The defensible headline is the **write** half: **2.02 GB → 394.7 MB, 5.1× at R=8** on the 35B, in
real device writes. Write volume scales as (R+1)/2, so at the production R≈17 it is ~9×, not 17×.

Two caveats in the other direction, both real: a page-cache read still costs `loadFull` plus `eval`
in wall clock (the 35B's per-resume wall grew 1.79 s → 2.29 s as the snapshot grew 106 → 395 MB), and
a machine with less free RAM, or a warm interleaved with real interactive work, will not hold a
multi-GB snapshot resident. The harness now prints wall clock for both arms side by side so the next
run settles this rather than arguing it.

**(b) `copy()` is not a deep copy.** `PromptCacheIO.save:54` does `liveCache.map { $0.copy() }`, and
every reachable `copy()` funnels through `$0[.ellipsis]`, which mlx's `slice()` short-circuits to
return the *identical* array when the shape is unchanged (`ops.cpp:768`). The "snapshot" aliases the
live cache's buffers. The transient doubling, where it happens, comes later — from `contiguous()`
inside the safetensors writer, and only when the state getter returns a strided partial slice. The
`KVCache` protocol doc claiming "an independent deep copy" (KVCache.swift:86) is wrong for
`ArraysCache`/`MambaCache`.

**(c) My "0.6–2.0 s process-wide freeze" was extrapolated, not measured, and is too high at these
sizes.** Measured 54–288 ms for the record itself. It will scale with snapshot size — the production
183k-token snapshot is ~10× the 35B's test snapshot — but state it as ~0.5–3 s projected, not
measured, until someone runs `--tokens 183296`.

---

## 3. The decision

**Do residency. Do not do block storage yet.**

Residency removes both quadratics with no storage-format change, is measured lossless on all three
models, and reuses a pattern already shipping in this package. Block storage fixes the write half
only, adds reassembly cost to the reuse path (which is where the README's headline warm-TTFT numbers
come from), and makes LRU eviction genuinely harder — evicting a middle block orphans every deeper
block in the chain.

---

## 4. API shape, and the coupling question answered

The design space collapses on one dependency fact: `ModelContainer.perform` is
`perform<R: Sendable>(...) -> sending R` (ModelContainer.swift:90-92), so **a value containing
`[KVCache]` cannot be returned out of `perform` at all**. That rules out returning an opaque handle
to the caller — it would need the `perform(nonSendable:)` escape hatch, which
`session-store-reshape.md:558` makes a standing check to keep gone.

It also rules out the coordinator holding warms in a private map: `PromptCacheCoordinator` is
constructed ad hoc all over this repo (main.swift:191, :227), so those call sites would silently lose
residency with no compile error, and the type would have to become `@unchecked Sendable`, violating
the standing checklist item that `SessionStore` is the only such conformance in the module
(session-store-reshape.md:553).

That leaves a caller-owned holder, and the recommendation is a **sibling type to `SessionStore`**:

- `SessionStore`'s divergence semantics are wrong for warms. It clamps (`SessionStore.swift:42`),
  which is bounded and self-correcting for a conversation. For a warm it would extend the wrong cache
  and then record it under a chain hash that does not describe its contents — poisoning the catalog
  with a snapshot that is later served as a valid prefix.
- Warms need a byte budget and an eviction policy that conversations do not.

**Is the coupling justified?** Yes, on one condition: residency must be a **strict accelerator that
degrades losslessly**. If the id is not resident — fresh process, evicted, released — `warm` must do
exactly what it does today: peek, reuse from disk, continue. That is not a new contract; it is the
one the README already claims for the hot tier ("a strict accelerator — every resident snapshot also
exists on disk, so RAM eviction is always lossless").

The honest residual cost is **durability granularity**: today a crash loses at most one chunk; with
residency plus persist-on-completion it loses the whole warm. That is the decision the cadence knob
exists to make, and it is yours (§7).

---

## 5. Implementation plan

**New — `Sources/MLXPromptCache/WarmStore.swift`**

- `public final class WarmStore: @unchecked Sendable`, holding `[UUID: Entry]` where
  `Entry = (cache: [KVCache], frontier: BlockHash)`.
- Invariant comment copied verbatim in structure from `SessionStore.swift:9-17`: touched only inside
  `ModelContainer.perform`. **Do not use a `Mutex`** — it adds a second access path reachable off
  `perform` (session-store-reshape.md:108-115).
- `package func resume(_ id:expecting: BlockHash?) -> [KVCache]?` — returns `nil` on hash mismatch,
  never extends a diverged cache.
- `package func hold(_ id:cache:frontier:)`, `package func release(_ id:)`, `public func releaseAll()`.
- Byte accounting plus a `budgetBytes` cap with an evict→persist→release victim path, so eviction
  costs a snapshot rather than losing work.
- Update the checklist item in `session-store-reshape.md:553` to name both types.

**Changed — `PromptCacheCoordinator.swift`**

- Add an overload: `warm(_ warms: WarmStore, id: UUID, promptTokens:model:parameters:persist:shouldPause:)`.
  Distinct argument labels, so the existing `warm(promptTokens:...)` is untouched and nothing at
  0.4.1 breaks. Every current call site labels `shouldPause:` explicitly, so there is no
  trailing-closure ambiguity.
- Resume path: `warms.resume(id, expecting: frontierHash(promptTokens, upTo: resident))` first;
  on `nil`, fall through to today's `store.peek` / `store.reuse` path unchanged.
- On `.paused`, hold instead of record (subject to `persist`).
- Add `finishWarm(_ warms:id:model:)` — persist and release. **Take `model:` even though it is
  unused**, to inherit the existing type-level "inside `perform` only" nudge
  (PromptCacheCoordinator.swift:186-190); this call touches `[KVCache]` *and* does MLX work, so it is
  more dangerous off-domain than `SessionStore.release`.
- `public enum WarmPersistence: Sendable { case never, onCompletion, everyTokens(Int) }`.
  `everyTokens` must be keyed on tokens accrued, **not** pause count — keying on pauses re-couples
  durability to interactive activity, which is the bug.

**Unchanged — `Catalog.swift`, `PromptCacheStore.swift`, `PromptCacheIO.swift`, `HotCodec.swift`.**
Not one line. Fewer, later `record` calls exercise exactly the paths that exist today.

**Worth fixing while nearby (documentation defects found in this review)**

- `PromptCacheCoordinator.swift:147` claims "`record` skips already-catalogued blocks itself". It
  does not — `planRecord` gates on *at least one* block being new, then re-plans all of them.
- `PromptCacheStore.swift:144` references `mlx-prompt-cache-hybrid-models.md`. That file has never
  existed in any commit reachable from any ref. It was a forward reference, not a lost document.
- `Sources/MLXPromptCacheScratch/main.swift` leaks every store directory it creates (:153, :204,
  :256 — no `defer`, no `removeItem`).

---

## 6. Regression risks, named

- **`PromptWarmTests.pausedProgressSurvivesReopen` (:99) changes meaning.** It asserts that a paused
  warm's progress survives a full store reopen with no in-memory state anywhere — "the disk IS the
  resume token". Under residency with `persist: .onCompletion`, a paused warm writes nothing and the
  reopen finds nothing. Keep the test green by making it exercise the *non-resident* path explicitly
  (which must still behave exactly as today), and add a new test for the resident path.
- **`repeatedPausesAdvanceBlockByBlock` (:160)** asserts `store.peek` advances one block per pause.
  Under residency the catalog does not advance until persist. Same treatment.
- **`SessionStoreTests.growsAcrossTurnsHeldNotReloaded` (:64)** is the template to copy for the new
  residency tests — it proves residency by `ObjectIdentifier` equality plus a `reuseCalls == 1`
  counter, which transfers unchanged.
- **A latent hazard worth a rule, independent of this change:** `generateTokens` does *not* run in
  your `perform`. It wraps the `[KVCache]` in a `SendableBox` and launches an unstructured detached
  `Task` (Evaluate.swift:1825-1829), and `AsyncStream<TokenGeneration>` **is** `Sendable`, so it
  satisfies `perform`'s `R: Sendable` and can be returned out and drained outside — putting two live
  caches on MLX concurrently with zero compiler diagnostics. The existing harness obeys the rule by
  convention only (main.swift:192-199, :290). State it in the README's Threading section: every
  stream is drained inside the `perform` that created it.

---

## 7. Persist cadence, and the abandonment answer

The cost of persisting an abandoned warm, beyond the bytes, is a **process-global MLX stall**:
`save(arrays:metadata:url:)` holds `evalLock` (`IO.swift:76-80`), a module-global `NSRecursiveLock`
(`Transforms+Eval.swift:9`) taken by every `eval`, `asData`, save and load in the process. Measured
54–288 ms for these snapshots; canary max stall 260–341 ms.

That reframes the question you asked. Persisting an abandoned warm **once** is cheap — a third of a
second, at a moment the user has already walked away. Persisting at **every pause** is the same stall
× R, interleaved with exactly the interactive work the yield existed to protect. The current design's
remedy triggers the disease it is treating.

Recommendation: `persist: .onCompletion`, plus an explicit `finishWarm` on abandonment (tab closed,
file deselected, memory-pressure eviction). Two writes per file in the normal case. Add
`.everyTokens(K)` only if you want to bound crash loss below a whole file.

---

## 8. Block storage — still the answer to a different question

Unchanged from the prior document, and the measurements reinforce it:

- A hybrid **can** be block-stored. oMLX does it, and the mlx-swift-lm primitives support it
  (`loadPromptCache` reconstructs `MambaCache`, KVCache.swift:1723).
- But oMLX's exact shape — a full recurrent-state copy in every block — costs 66.1 MB per 256-token
  block on the 35B, i.e. **258 kB/token against 20 kB/token for the attention KV it accompanies**.
  That is worse than the status quo. oMLX hit the same wall for sliding-window caches and its answer
  is a two-rung ladder (`prefix_cache.py:728-737`), which is what we would copy.
- Revisit it when many *different* prompts start sharing partial prefixes, because then the
  catalog's existing block granularity finally has something to exploit. Warming one file per
  document does not have that property.

---

## 9. Open items

- Re-run the bench after the three probe fixes to confirm `G1.A`, `G4.sensitivity` and `G6.hotTier`
  go green, and to capture the residency arm's wall clock (newly printed) — that is the number that
  survives the page-cache caveat.
- `--tokens 183296 --model 35B` measures the production point directly. The 35B's context is 262,144,
  so no extrapolation is needed; I previously assumed 32k and was wrong.
- Decide `persist` default and whether abandonment persists or discards. This document assumes
  persist.
- Residency memory: predicted to hold A·T, but `KVCacheSimple.update` grows by
  `concatenated([current, new], axis: 2)` (KVCache.swift:410-411) with both buffers live, so the
  transient peak is nearer 2·A·T. The bench resets `Memory.peakMemory` per phase and records it; read
  it off the next run before sizing the budget. At 137 GB with a 20 GB model this is not binding, but
  it is the number the `budgetBytes` cap must be set from.
