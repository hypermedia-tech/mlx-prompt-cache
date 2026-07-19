# Hybrid block storage vs. warm-loop I/O — proposal

**Status:** proposal. Nothing applied. No source file changed.

Answers the four questions posed: (1) can a hybrid KV cache be stored per-block, (2) if yes the design,
(3) if no what oMLX does instead and whether snapshot cadence is the fallback, (4) can the catalog stay.

Short version: **yes, a hybrid can be block-stored — but block storage is not the fix for the measured
problem, and on this model oMLX's exact shape would be four times worse than what we do today.** The
measured pathology is a *warm-loop* pathology, not a storage-format one. Recommendation in §8.

---

## 0. Provenance — what is verified and what is not

Rule 3 compliance. Two different standards of evidence in this document:

- **MLXPromptCache and mlx-swift-lm claims:** read directly, this session, with `file:line`. Treat as fact.
- **oMLX claims:** read by subagents against `~/workspace/learning/omlx`, with `file:line` cited. I
  personally re-read and confirmed only the four load-bearing ones, marked ✅ below. The rest are
  agent-sourced and worth spot-checking before you rely on them.

Verified by me in oMLX:

- ✅ `type_handlers.py:772-779` — `ArraysCacheHandler.concatenate_states` returns `states[-1]` with the
  comment `# Use latest state`. Recurrent state is **not** concatenated across blocks on restore.
- ✅ `prefix_cache.py:1270-1311` — each block stores a *full copy* of the recurrent state (`conv_state`,
  `ssm_state`) when a boundary snapshot covers it, else a `mx.zeros((1,))` placeholder.
- ✅ `scheduler.py:1855-1859` — `_ARRAYS_CACHE_BLOCK_SIZE = 2048`, to "reduce the number of boundary
  snapshot stops during prefill while still storing valid per-block recurrent state."
- ✅ `prefix_cache.py:728-737` — supersede-on-extend, quoted in §4.

One naming correction that matters when reading oMLX: **there is no `MambaCache` type in oMLX or in
Python mlx-lm.** Recurrent models there use `ArraysCache(size=2)`. mlx-swift-lm *does* have a real
`MambaCache`, and it is exactly that — `public class MambaCache: ArraysCache` with
`super.init(size: 2)` (`KVCache.swift:1399-1409`). Same thing, different name. Wherever oMLX says
ArraysCache, read MambaCache.

---

## 1. The measurement, decomposed

The five resume sizes fit a straight line to within 0.001%:

- `bytes(N) = 20,480·N + 64,397,371`
- Marginal cost: **20,480 B/token — exactly 20 KiB/token.** This is the attention KV.
- Fixed cost: **~64.4 MB, independent of prefix length.** This is the recurrent (Mamba/SSM) state.

Cross-checked against two independent datasets in the repo's own README, which the fit was not derived
from:

- README:162 claims `51 → 23 KiB` per token across a 2k→24k sweep. Model predicts 50.7 KiB at 2,048
  tokens and 22.6 KiB at 24,576. Match.
- README:162 claims `~0.5 GiB` snapshot at 24k. Model predicts 0.53 GiB. Match.

So the brief's "consistently ~21.2 KB/token, whole prefix each time" is very slightly off in a way that
matters: it is **not** constant, it is *falling* — 22.6 → 21.2 KB/token across the five samples — because
a fixed 64.4 MB is being amortised over a growing N. That falling curve is the single most important fact
in this document, because it says the two halves of the snapshot have completely different economics and
must be costed separately.

Same decomposition for the other two models in README:163-164, for calibration:

- `Qwen3.5-9B` (hybrid): ~32 KiB/token attention + ~53 MB fixed recurrent.
- `Qwen3-1.7B` (pure attention): 112 KiB/token, zero fixed. The brief's warning is correct — attention is
  not the cheap path; it is 5.6× the per-token cost of the production hybrid.

---

## 2. Why it is quadratic — and one correction to the brief

Both leads about `PromptCacheIO` are **confirmed**:

- `PromptCacheIO.swift:54` — `let snapshot = liveCache.map { $0.copy() }`, the whole live cache.
- `PromptCacheIO.swift:68` — `try savePromptCache(url:cache:metadata:)`, unconditional, whole snapshot.
  There is no delta path, no append path, for any model type.
- `PromptCacheIO.swift:56-67` — `isSliceable` chooses only *how the snapshot is validated before the
  write*: trim down to the prefix (attention), or require it already sits exactly on the prefix
  (hybrid, else `Failure.hybridNotAtBoundary`). Both branches fall through to the same line 68. Exactly
  as the brief suspected.

The third lead is **refuted**, and this is worth correcting because the wrong mental model leads to the
wrong fix. The brief says `catalog.planRecord(hashes, blockSize:)` "already skips already-catalogued
blocks — so the indexing is block-granular while the storage is not."

It does not skip them. `Catalog.swift:30-37`:

- The `guard` only asks whether *at least one* block is new (`hashes.contains(where: { byHash[$0.hex] == nil })`).
- If so, `hashes.enumerated().map` plans **every** boundary, including all the already-catalogued ones.

So the write plan is whole-prefix too. What is block-granular is only the *lookup key space*. This also
explains the `evicted 1` on every round, which is not LRU and not budget pressure — it is orphan
collection in `Catalog.commit` (`Catalog.swift:55-81`):

- Each new plan re-points every boundary at the new file, removing it from the previous file's
  `ownedBoundaries` (lines 59-61).
- The previous snapshot is a strict prefix of the new one, so it loses *all* its boundaries.
- `files.filter { $0.value.ownedBoundaries.isEmpty }` (line 73) then finds it orphaned and drops it.

That behaviour is *correct* given whole-prefix snapshots — the old file genuinely is redundant. It is not
a bug to fix; it is a symptom to stop causing.

Two consequences worth noting while you are in there:

- `PromptCacheCoordinator.swift:147` says "`record` skips already-catalogued blocks itself." That comment
  is wrong, and is where the brief's misreading came from.
- `PromptCacheStore.swift:144` references `mlx-prompt-cache-hybrid-models.md`. I searched every commit
  reachable from all refs; that file has never existed in this repository. `README.md` and
  `.cat/docs/session-store-reshape.md` are the only docs that have ever been tracked. Nothing was lost —
  it was a forward reference that was never written.

### The read side is quadratic for a different reason, and it is an API fact

`PromptCacheCoordinator.warm` (`PromptCacheCoordinator.swift:111-152`) returns `PromptWarmOutcome` — a
plain enum of `Int`s. The live `[KVCache]` it builds at line 130 is a local, and it is dropped when
`warm` returns. **There is no way for a caller to hold it.** So every resume *must* begin with
`store.reuse(...)` at line 128, which calls `PromptCacheIO.loadFull` and reads the entire previous
snapshot back. The harness confirms this is the real usage: `main.swift:215-217` pauses, then calls
`warm` a second time as a separate call.

That is the whole read quadratic, and no storage-format change can remove it.

### A third finding: the RAM tier is inert for the production model

`HotCodec.extract` (`HotCodec.swift:26-40`) returns `nil` if any layer is not `KVCacheSimple` or
`QuantizedKVCache`. A hybrid has a `MambaCache`, so it returns `nil`, so `hot.insert` is never reached
from either `reuse` (`PromptCacheStore.swift:78-79`) or `preload` (`:169-175`).

`hotBudgetBytes` therefore does nothing whatsoever on `unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit`. This is
known and deliberate — `main.swift:19` sets `testHot: false` with the comment "hybrid → cold / disk only
(RAM tier is attention-only)" — but it means the one tier that could have made resume reads cheap is
switched off precisely for the model that needs it.

---

## 3. Q1 — Can a hybrid KV cache be stored per-block?

**Yes.** Both halves of the evidence:

**From oMLX.** It already does it. A block file holds, for the same block: sliced KV for the attention
layers, and a *full copy* of the recurrent state as of that block's boundary (`prefix_cache.py:1270-1311` ✅).
The recurrent state comes from a "boundary snapshot" captured during prefill and keyed by token count
(`prefix_cache.py:614-621`, `boundary_snapshot_store.py`). Where no snapshot covers a block, the
recurrent layers store a `mx.zeros((1,))` placeholder and a partial prefix match landing there is
rejected outright (`prefix_cache.py:2221-2235`) or walked back to the last block with real state
(`_find_walk_back_truncation_point`, `prefix_cache.py:1339`).

**From this package's own dependency.** The Swift primitives already support it, and nothing needs to be
hand-rolled:

- `savePromptCache` (`KVCache.swift:1592-1630`) serialises whatever `[KVCache]` you hand it, flattening
  `state` to `"i.j"` safetensors entries plus `metaState` and class names. A synthetic per-block cache
  array is a legal input.
- `loadPromptCache` reconstructs a `MambaCache` (`KVCache.swift:1723-1726`) via the module-internal
  `restoreFromMetaState`. So the round trip exists through the public API.
- `KVCacheSimple.state`'s getter slices to `offset` and its setter derives `offset = keys.dim(2)`
  (`KVCache.swift:430-449`). A per-block KV slice is therefore a well-formed cache on its own.
- `MambaCache` is `ArraysCache(size: 2)` (`KVCache.swift:1399-1409`) — two arrays, fixed shape,
  **independent of prefix length**. Confirmed empirically by the 64.4 MB constant in §1.
- Our `BlockHasher.hash` (`BlockHasher.swift:15-30`) is already the same construction as oMLX's
  `compute_block_hash` — a chain digest over signature/model ++ parent ++ this block's tokens. Keys are
  prefix-dependent, which is exactly what block-level KV storage requires for correctness.

One real constraint to design around: `ArraysCache.metaState`'s **setter traps** —
`assertionFailure("ArraysCache.metaState should not be set directly")` (`KVCache.swift:1360-1364`) — and
`restoreFromMetaState` is `internal`. So a recurrent layer can only be rebuilt by going through
`loadPromptCache`, never by assembling one from raw bytes ourselves. That is precisely why `HotCodec`
bails on non-attention layers rather than reconstructing them. Any block format we invent must therefore
be written by `savePromptCache` and read by `loadPromptCache`, not by a bespoke safetensors reader.

So the answer to the question as posed is unambiguous. The interesting question is the next one.

---

## 4. Why oMLX's exact shape is wrong for this model

oMLX snapshots the recurrent state at **every** block boundary, unconditionally — there is no cadence
heuristic, no growth threshold, no debounce (`scheduler.py:2447-2449`, `:2580-2590`). Priced against our
measured 64.4 MB recurrent state:

- At our current `blockSize: 256` — 64.4 MB per block is **246 KiB/token**, against 20 KiB/token for the
  attention KV it accompanies. Twelve times the cost of the thing it is riding along with.
- Warming the full 183k-token file that way: **~49.7 GB**, versus ~21–35 GB for the status quo. It would
  make the problem *worse*.
- At oMLX's enlarged `_ARRAYS_CACHE_BLOCK_SIZE = 2048` ✅ — 30.7 KiB/token, still above the attention cost.
  Full file ~9.5 GB. Better than today, but the recurrent state is still the majority of the bytes.

This is not a criticism of oMLX; it is a difference in constants. But oMLX itself hit exactly this wall
for the sibling case — sliding-window caches, whose state is also large and non-sliceable — and its
answer is the one we should copy. `prefix_cache.py:728-737` ✅, verbatim:

> Supersede-on-extend: on rotating (sliding-window) models every store of a growing conversation writes
> one tip block carrying the full sliding-window state of all rotating layers (hundreds of MB fp16 on a
> gemma3-class model). Restore only ever consumes the newest such block, and the immediate previous tip
> is kept intact as the walk-back fallback — so the tip two generations back is dead weight. Without
> stripping it, those blocks fill the hot cache after ~10-20 turns and LRU eviction breaks the prefix
> chain (multi-turn cache hit collapses to 0%). Steady state after stripping: two heavy blocks per chain.

Three transferable lessons, and the third is the one that decides the design:

- Heavy non-sliceable state stored per block does not just cost bytes — it **breaks LRU and collapses the
  hit rate**, because evicting one of those fat blocks severs the chain.
- The fix is a *sparse ladder*: keep the newest heavy state plus one fallback, strip the rest.
- And the reason a sparse ladder is sufficient: on restore, recurrent state is **not** accumulated across
  blocks — `concatenate_states` returns `states[-1]` ✅. Only the state at the match point is ever read.
  Every other per-block copy is pure redundancy that exists solely to *permit* a match at that point.

So the correct hybrid block design is not oMLX's Mamba path. It is oMLX's rotating path: per-block
attention KV, plus recurrent state at only the few boundaries you actually want to resume from.

---

## 5. Q2 — The block-storage design, if you choose it

Costed so you can compare, then set aside in favour of §6.

**Storage.** One file per block, written by `savePromptCache` over a synthetic `[KVCache]`:

- Attention layers: that block's KV slice only (`state = [k[..., r, ...], v[..., r, ...]]`).
- Recurrent layers: the boundary state, on ladder blocks; on non-ladder blocks a 1-element placeholder,
  mirroring `mx.zeros((1,))`.
- Filename from the block's chain hash, which `BlockHasher.boundaries` already produces. A one-hex-char
  subdirectory fan-out if directory size becomes an issue; at 714 blocks per file it will not.

**Ladder policy.** Recurrent state at the frontier of each warm, plus the previous one as the walk-back
fallback — oMLX's "two heavy blocks per chain". Cost becomes `64.4 MB × 2` retained per prefix rather
than `64.4 MB × 714`.

**Reassembly on reuse.** `concatenated(_:axis: 2)` per layer per tensor across the matched blocks, then
`state =` on a fresh `KVCacheSimple`; recurrent layers taken from the deepest ladder block at or before
the match, via `loadPromptCache`. Note this is real work on the reuse path: ~714 blocks × ~50 layers × 2
tensors for a full file.

**Every call site that changes:**

- `PromptCacheIO.save` — replaced by a block-slicing writer. `trim`/`loadFull` gain block-assembly twins.
- `PromptCacheIO.isSliceable` keeps its meaning but gains a per-*layer* use (slice attention, snapshot
  recurrent), which `isSliceableLayer` already supports.
- `Catalog.planRecord` — must return only the boundaries not already in `byHash`. Currently returns all
  (`Catalog.swift:33-35`).
- `Catalog.commit` — the boundary-stealing and orphan sweep (`:59-74`) must go; with one file per block
  there is nothing to steal.
- `Catalog.Boundary`/`FileRecord` — collapse toward 1:1. `ownedBoundaries` becomes a singleton.
- **`Catalog.commit`'s LRU loop (`:75-78`) becomes unsafe as written** and this is the sharpest new
  hazard: evicting the least-recently-used *block* can punch a hole in the middle of a chain, leaving
  every deeper block unreachable but still counted against the budget. Eviction has to become
  chain-aware — deepest-block-first within the coldest chain. oMLX does not solve this (it has no
  refcounting at the SSD tier and simply takes the miss); we would have to, or accept silent capacity
  rot.
- `PromptCacheStore.reuse` — one `loadFull` becomes an N-block assembly.
- `PromptCacheStore.write` — one save becomes N saves plus ladder bookkeeping.
- `HotCodec`/`HotCache` — keyed by snapshot filename today (`HotCache.swift:16`); would need block keys.
- `PromptCacheCoordinator.warm`/`prepare` — unchanged in shape.

**Migration.** Trivial, and already the house idiom. `Catalog.loadOrReset` (`Catalog.swift:98-108`) wipes
the directory outright on any signature or `blockSize` mismatch — "greenfield — no migration", restated
in `.cat/docs/session-store-reshape.md:719`. Add a `formatVersion` to `Catalog.Header` and bump it; every
existing catalog self-wipes on first open and re-warms. CyberBench is the only consumer and it is at
0.4.1, so there is no compatibility burden. Nothing else is needed.

**What it actually buys.** Warming the full file at 17 resumes:

- Status quo: ~34.8 GB written, ~31.0 GB read back — ~66 GB round trip. (At 10 resumes, ~21/~17/~39 GB.
  The brief's ~34 GB projection implies ~17 resumes, so the two agree.)
- Block storage with a two-rung ladder: ~4.8 GB written — a 7× cut — but **~31.0 GB still read**, because
  each resume must still reconstruct the full prefix from disk. ~36 GB round trip.

That is the crux. Block storage fixes writes and leaves reads untouched. oMLX has the identical property
and says so: reload is per-block and O(prefix) every time, and its hot tier "changes the constant, not
the complexity."

---

## 6. The change that dominates it — hold the cache across the yield

The measured problem is that a *background warm yields and resumes*. Nothing about that requires a round
trip to disk. The live cache is dropped only because `warm`'s signature gives the caller no way to keep it.

Hold it, and both quadratics disappear at once:

- Nothing is written on a pause, so the write quadratic is gone.
- Nothing is read on a resume, so the read quadratic is gone.
- One snapshot at completion: **3.81 GB total, no reads.** Against ~66 GB round-trip today, **~17×**.
  Against block storage's ~36 GB, **~9×**.

This is not a new architecture. It is the pattern this package already ships, applied to the warm path:
`SessionStore` (`SessionStore.swift:18-48`) already holds `[UUID: [KVCache]]` across calls with a written,
reviewed `@unchecked Sendable` invariant, and `PromptCacheCoordinator.advance` already drives it. The
conversation gate in `main.swift:271-315` already proves the held-cache path is byte-exact on this very
hybrid model — it asserts `held.a2Held == a2Cold`, token-identical, having extended a held `[KVCache]`
across turns. Holding a warm is the same operation with a different driver.

It is also *less* risk to the recurrent state than what we do today, not more: the cache simply never gets
serialised mid-warm, so the untrimmable-state round trip that `Failure.hybridNotAtBoundary` exists to
police stops happening on every pause.

### Concurrency review of the proposed shape

Run per rule 4, against `swiftLanguageModes: [.v6]` with no main-actor-by-default opt-in (`Package.swift:58`)
— so the package is `nonisolated` by default, which is correct for code that runs inside
`ModelContainer.perform`. Four findings:

- **`@unchecked Sendable` on a warm holder is justified on identical grounds to `SessionStore`, and the
  pause does not weaken it.** The worry would be that interactive work runs between a pause and a resume —
  but that work also goes through `perform`, which serialises it, and `SessionStore` already spans
  multiple `perform` blocks between conversation turns. Same invariant, already exercised. Do not reach
  for a `Mutex`: as `session-store-reshape.md:108-115` argues, it would add a second access path to the
  caches reachable off `perform` and defeat the single-serialised-domain guarantee.
- **Reuse `SessionStore` rather than adding a parallel type — but the value must carry a validating
  hash.** A held cache keyed only by `UUID` has no content gate. If a caller resumes the wrong id, we
  extend the wrong cache and then `record` it under a chain hash that does not describe its contents —
  **poisoning the catalog with a snapshot that will be served as a valid prefix**. Today that cannot
  happen, because the disk path is gated by the block chain. `SessionStore.advance` tolerates the
  analogous divergence for conversations by clamping to an empty delta (`SessionStore.swift:42`), which is
  bounded and self-correcting; for a warm it is neither. Mitigation is cheap: hold the frontier
  `BlockHash` alongside the cache and, on resume, recompute it from `promptTokens[0..<resident]` and
  compare. Pure CPU, no GPU, no IO. **This is the one genuinely new correctness hazard the change
  introduces, and it must be closed in the same commit.**
- **A `finishWarm`-style call must take `model:` even if it does not need it.** The codebase already
  enforces "inside `perform` only" at the type level by requiring `model`, which is reachable only via
  `context.model` (`PromptCacheCoordinator.swift:186-190`). A persist-and-release call touches `[KVCache]`
  *and* does MLX work, so it is more dangerous off-domain than `SessionStore.release` is, and should
  inherit the same nudge.
- **The non-escaping `shouldPause: () -> Bool` is fine** — non-`Sendable` but non-escaping and called
  synchronously, never crossing an isolation boundary. Unchanged from today's signature
  (`PromptCacheCoordinator.swift:115`).

And one non-concurrency risk that is the most likely thing to actually bite:

- **Residency must be bounded, and this is mandatory rather than nice-to-have.** Holding a full-file warm
  costs 20 KiB/token — 3.74 GB for the 183k-token file — in unified memory for the duration. A
  conversation cache is user-scoped and small; a warm cache is GB-scale and spawned by background code,
  so a leaked or forgotten id is a multi-gigabyte leak. This needs a byte budget, an eviction policy
  (evict → persist → release, so eviction costs a snapshot rather than losing work), and ideally a
  fallback to today's reload-from-disk behaviour when over budget. On the 128 GB M4 Max in the README,
  next to a ~20 GB model, 3.74 GB is comfortable; on a 36 GB machine warming several files at once it is
  not.

---

## 7. Q3 — Is snapshot cadence the right fallback?

The question assumed cadence would be a consolation prize if hybrids could not be block-stored. It is
better than that: **with residency, cadence stops being the resume mechanism and becomes purely the
durability policy** — which is what it should have been all along, and what oMLX does.

oMLX persists **once per finished request**, on a single background worker
(`scheduler.py:1644`, dispatched from `_cleanup_finished`), not per generation step. Its only per-block
cadence is boundary snapshotting during prefill, and that is gated off entirely for models that do not
need it.

So the cadence question reduces to: *what is the acceptable loss if the process dies mid-warm?*

- Persist on completion and on abandonment (tab closed, file deselected, memory-pressure eviction). Two
  writes per file in the normal case, ~3.8 GB.
- Optionally a token-count ladder — persist when ≥ K tokens have accrued since the last write — for very
  long warms. Note this must be keyed on *tokens accrued*, not on pause count, or it re-couples to
  interactive activity and we are back where we started.
- Do **not** persist on every pause. That is the current behaviour and the entire bug.

Cadence alone, without residency, is worth having but much weaker: every snapshot is still whole-prefix
and every resume still reloads, so both quadratics survive with a smaller constant. Persisting every 4th
pause instead of every pause is still ~9.6 GB of writes plus the full read load.

---

## 8. Recommendation

In order. Each step is independently shippable.

1. **Hold the warm cache across yields; persist on completion and abandonment.** Fixes both quadratics,
   ~17× less I/O, no storage-format change, reuses a pattern already proven byte-exact on this model.
   Ship with the chain-hash guard from §6 and a residency budget.
2. **Fix the two documentation defects** found on the way — the wrong comment at
   `PromptCacheCoordinator.swift:147`, and the dangling reference to a doc that never existed at
   `PromptCacheStore.swift:144`.
3. **Do not do block storage yet.** It solves half the problem, costs a format change, makes LRU eviction
   genuinely harder (§5), and adds reassembly work to the reuse path — which is the path the README's
   headline warm-TTFT numbers come from. The single-file snapshot is close to optimal for *serving*; the
   bug is entirely in *warming*.
4. **Revisit block storage when — and only when — the workload changes shape.** It becomes the right
   answer if many *different* prompts start sharing partial prefixes, because then the catalog's existing
   block granularity finally has something to exploit and whole-prefix snapshots start duplicating each
   other. Warming one file per document does not have that property. If that day comes, §5 is the design
   and §4 says use a two-rung recurrent ladder, not oMLX's per-block copies.

If residency turns out to be unacceptable for a memory reason I have not anticipated, the fallback order
is: cadence alone (§7, weak but trivial), then block storage with a two-rung ladder (§5, ~2× better than
today, much bigger change).

---

## 9. Q4 — Can `PromptCacheStore`'s catalog stay as-is?

- **Under the recommendation (§8 steps 1–2): yes, entirely unchanged.** Not one line of `Catalog.swift`
  needs to move. Fewer, later `record` calls exercise exactly the paths that exist today. Worth doing
  anyway while nearby: correct the comment at `PromptCacheCoordinator.swift:147`, and consider making
  `planRecord`'s all-boundaries behaviour explicit in a comment, since it has now misled one reader.
- **Under block storage (§5): the shape survives, three behaviours do not.** `byHash → Boundary` and the
  `files` LRU map are the right structures and would carry over. But `planRecord` must plan only new
  boundaries; `commit` must drop the boundary-stealing and orphan sweep; and the LRU victim loop must
  become chain-aware or it will punch holes in chains. The persisted `index.json` is, incidentally, a
  strict improvement on oMLX here — oMLX persists no index at all and rebuilds by scanning every file at
  startup (`paged_ssd_cache.py:1290-1336`). Keep ours.

---

## 10. Open questions for you

- What is the memory headroom on the target machine while a warm is in flight? That is the only input
  that could change the §8 ordering. 3.74 GB per concurrently-warmed file, held for the duration.
- Should an abandoned warm persist or discard? Persisting costs one 3.8 GB write and keeps the work;
  discarding is free and throws it away. My assumption above is persist.
- Is anything other than CyberBench's per-file warmer driving `warm`? The recommendation is tuned to that
  one caller, and 0.4.1 with a single consumer is a free hand.
- Do you want the `Qwen3.5-9B` numbers in §1 re-measured rather than derived from the README's rounded
  2-point sweep? The 35B fit is solid — five points, sub-0.001% residuals — but the 9B split
  (~32 KiB/token + ~53 MB) is inferred from two rounded figures and should not be quoted as measured.
