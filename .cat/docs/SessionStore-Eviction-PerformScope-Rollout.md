# SessionStore Eviction + `PerformScope` — rollout & caller migration

**Status:** change draft, ready to execute. Module side is applied on `feature/add-eviction-surface`
(`PerformScope` gate + app-budgeted session eviction); the **caller migration, tests, and release** are
what this doc drives.
**Design rationale:** `SessionStore-Eviction-Concurrency-Invariant.md` (the ticket) — read it for the *why*.
This doc is the *how to land it*.

Verify every line reference against the file in your session before relying on it — the numbers here are
guides, the patterns are the contract.

---

## 1. What this change is (one paragraph)

Every `PromptCacheCoordinator` door that mutates the live-cache stores (`SessionStore`/`WarmStore`) now
requires a `PerformScope` — a witness obtainable **only** inside `ModelContainer.perform`. An off-`perform`
cache mutation becomes a **compile error** instead of a latent heap-corruption race. The same change adds
`evictSessions(overBudget:keep:scope:)`, an app-budgeted way to shed the largest resident conversation
caches under memory pressure. The cache codec (`PromptCacheStore`/`PromptCacheIO`/`Catalog`/`HotCache`/
`HotCodec`/`BlockHasher`) is **untouched** — this is purely the RAM-residency lifecycle plus a compile gate.

**Confirmed on the applied branch:** the witness is the strong form —
`public struct PerformScope: ~Escapable` with `@lifetime(immortal)` init and
`@lifetime(borrow context)` on `scope(_:)` — and it builds. So a token cannot be stashed and reused after
`perform` returns; the guarantee is real, not the escapable fallback.

---

## 2. Module change (applied — reference only, do not re-apply)

- `PromptCacheCoordinator.swift`: new `PerformScope` type + `public func scope(_ context: borrowing ModelContext) -> PerformScope`; `advance`/`release`/`evictSessions`/`warm`/`finishWarm`/`heldCache` each gain `scope: borrowing PerformScope` (bodies unchanged).
- `SessionStore.swift`: `package` eviction surface — `residentBytes`, `victimsOverBudget(_:excluding:)`.
- `WarmStore.swift`: `release(_:scope:)` / `releaseAll(scope:)` gain the witness (bodies unchanged).
- `Package.swift`: `.enableExperimentalFeature("LifetimeDependence")` on the `MLXPromptCache` target.

### Open decision before you cut the release — gate the reads too?

The applied change gates **mutators only**. `WarmStore`'s three **public** reads — `heldIds`, `isEmpty`,
`residentBytes` — remain ungated (ticket §3.5); a read racing an in-`perform` write is still UB. Today it's
latent (the one caller reads them inside `perform`), but the public getters still invite the off-`perform`
access this whole change exists to kill.

**Recommendation: gate them now.** It's ~an hour of mostly-test edits, and folding it into this *already
breaking* release is far cheaper than a second break later. Since a property can't take a parameter, convert
each to a method: `func heldIds(scope:)`, `func isEmpty(scope:)`, `func residentBytes(scope:)`. Deltas in
Appendix A. If you'd rather ship the narrower gate, keep the §3.5 note as the explicit boundary — but decide
it here, don't let it default.

---

## 3. Caller migration walkthrough — CyberBench (`Infrastructure` adapters)

Two files call the gated doors, both already inside `container.perform`: `MLXConversationEngine.swift` and
`MLXContextWarmer.swift`. The migration is mechanical (**mint one scope per `perform`, thread it**) plus one
genuinely new piece — wiring `evictSessions` with an app budget (§3.3).

### 3.0 Prerequisite — the module release

`evictSessions` and the `scope:` params only exist in the new module version. Land the module release first
(§5), bump CyberBench's pin, resolve. Do **not** start the caller edits against the old pin — nothing will
compile and you'll chase phantom errors.

### 3.1 `MLXConversationEngine.conversationTurnStream`

The `coordinator` is built **inside** the `perform` closure, so the scope sits right beside it.

**Step 1 — mint the scope once, at the top of the `perform` body**, immediately after
`let coordinator = PromptCacheCoordinator(store: store)`:

```swift
let coordinator = PromptCacheCoordinator(store: store)
let scope = coordinator.scope(context)   // proof-of-perform; one per block, threaded everywhere below
```

**Step 2 — thread `scope:` into every gated call in the closure.** These are the sites (verify against the
file; the bodies otherwise don't change):

- the bank warm (turn-1 block): add `scope: scope` — it sits **before** the defaulted `persist:`/`shouldPause:`:
  ```swift
  let banked = coordinator.warm(
      warms, id: warmID,
      promptTokens: Array(rootTokens.prefix(bankBoundary + 1)),
      model: context.model, parameters: params,
      scope: scope,
      persist: .onCompletion,
      shouldPause: { Task.isCancelled })
  ```
- the finish-on-cancel: `coordinator.finishWarm(warms, id: warmID, model: context.model, scope: scope)`
- the seed and the diverged-reseed `advance` (both call sites): append `scope: scope` to the argument list.
- the diverged-prefix and degrade `release`: `coordinator.release(sessions, id: conversationId, scope: scope)`

**Step 3 — the miss-safety check.** After threading, search the closure for any un-migrated door
(`grep -n 'coordinator\.\(advance\|release\|warm\|finishWarm\|heldCache\|evictSessions\)(' MLXConversationEngine.swift`)
and confirm each carries `scope:`. A missed one is a compile error, which is the point, but catch it here
rather than in a confusing build log.

### 3.2 `endConversation`

Currently `await container.perform { _ in coordinator.release(sessions, id: conversationId) }`. The `_` has
to become `context` to mint a scope:

```swift
await container.perform { context in
    let scope = coordinator.scope(context)
    coordinator.release(sessions, id: conversationId, scope: scope)
}
```

### 3.3 Wire `evictSessions` — the new behaviour (needs a budget policy decision)

`evictSessions` is a *door*, not yet a *call*. The point of the ticket is that the app sheds resident session
caches under a budget it owns. Add the call inside the same `perform`, **after the final resolved `advance`** —
i.e. past the diverged-prefix reseed, not right after the first seed `advance` (which in the diverged path runs
*before* the reseed). `keep` is `conversationId` either way, so a misplacement is harmless, but after the
resolved advance is the clean spot:

```swift
coordinator.evictSessions(sessions, overBudget: sessionBudgetBytes(), keep: conversationId, scope: scope)
```

**`sessionBudgetBytes()` is the one real decision here.** It is an app-owned byte cap on total resident
session KV. A defensible v1: sample available unified memory at call time and subtract headroom for weights +
the warm budget, e.g.

```swift
// v1 — conservative fixed fraction; replace with the real RAM-aware policy once measured.
private func sessionBudgetBytes() -> Int { 8 << 30 }   // 8 GiB of session caches, tune per target machine
```

Flagged, not decided: pick the number (or the RAM-sampling formula) deliberately — too low and every
non-active conversation re-prefills its whole history on its next turn (correct, but a felt cost); too high
and it never sheds. Eviction is lossless (drops RAM only; the next turn re-seeds from the disk root and
re-prefills), so erring conservative is safe, just slower on resume.

### 3.4 `MLXContextWarmer.warmContext`

One gated call (`coordinator.warm`), already inside `container.perform`. It does **not** call
`warms.release`/`releaseAll`, so nothing else changes here.

```swift
let scope = coordinator.scope(context)
let outcome = coordinator.warm(
    warms, id: warmID, promptTokens: capped,
    model: context.model, parameters: GenerateParameters(),
    scope: scope,
    persist: .everyTokens(Self.bankInterval),
    shouldPause: { shouldYield() || Task.isCancelled })
```

### 3.5 If you took Appendix A (gate the reads)

Also convert any `warms.heldIds` / `warms.isEmpty` / `warms.residentBytes` reads in these two files to the
`(scope: scope)` method form. Grep first: `grep -n 'warms\.\(heldIds\|isEmpty\|residentBytes\)' Sources`.
If there are none (likely — these are mostly test/metrics reads), there's no caller edit and the gate is
pure hardening.

---

## 4. >>> TEST AGENT — add/update the suites <<<

Scope: `Tests/MLXPromptCacheTests/`. The suites are headless (`StubModel`, no `ModelContext`). They mint the
witness via the **`package` initialiser** reachable under `@testable import MLXPromptCache` — this is *why*
the init is `package`: no context needed in tests.

**Migration rule — every test that calls a gated coordinator door:**
1. Add one line near the top of the test: `let scope = PerformScope()`
2. Pass `scope: scope` at each `coord.advance` / `coord.release` / `coord.warm` / `coord.finishWarm` /
   `coord.heldCache` / `coord.evictSessions` call. `scope:` sits **before** the defaulted `persist:` /
   `shouldPause:`, so default-omitting calls still read naturally.
3. **MACRO BOUNDARY — the landmine (ticket §6.2).** `PerformScope` is `~Escapable`, and a `~Escapable` value
   **cannot cross a `#expect` / `#require` macro** — the error is the misleading *"requires that 'PerformScope'
   conform to 'Escapable'"*. So a gated call whose result is asserted must be **called into a local first, then
   asserted** — never inlined inside the macro:
   ```swift
   // WRONG — will NOT compile:
   let first = try #require(coord.heldCache(warms, id: id, model: model, scope: scope))
   // RIGHT — call, bind, then assert on the plain (Escapable) result:
   let held = coord.heldCache(warms, id: id, model: model, scope: scope)
   let first = try #require(held)
   ```
   This applies to **every** `#expect(...)` / `#require(...)` that wraps a gated call — `heldCache` in
   `WarmStoreTests`, and any assertion that inlines a gated door. `warm`/`finishWarm` results already land in a
   `let out = …` before a `guard case`/`switch` (not a macro), so those are fine as-is.
4. Leave **direct** `sessions.advance(…)` / `warms.hold/entry/resume(…)` calls unchanged — those are the
   `package` store methods, never gated.
5. No `#expect` changes in **meaning** — the asserted value is identical. Some change in **shape** (rule 3):
   where a gated call fed the macro, it splits into bind-then-assert.

**Files & specifics:**
- `SessionStoreTests.swift` — the three gated sites (the `coord.advance`+`coord.release` pair, and the two
  `coord.evictSessions` tests). Thread `scope`.
- `WarmStoreTests.swift` — all ~24 `coord.warm` / `coord.heldCache` / `coord.finishWarm` calls thread
  `scope`; the direct `warms.releaseAll()` call becomes `warms.releaseAll(scope: PerformScope())`.
- **New positive test** — `evictSessions` behaviour is already covered by the eviction tests; if the
  `~Escapable` fallback is ever taken, add `evictionUsesAFreshScopePerCall` (ticket §6.4) to pin that a
  freshly-minted token per call still evicts correctly. Under the confirmed `~Escapable` build it's
  redundant — add it only if the witness is ever downgraded.
- **Negative/compile guard** — Swift Testing has no "expect-no-compile". Add the fixture from ticket §6.3
  (`offPerformEvictionMustNotCompile` — an eviction call with **no** `scope`) to a **separate target that CI
  builds and expects to FAIL**, or document it as a review checkpoint. State which; a silent absence reads as
  "covered" when it isn't.
- **If Appendix A was taken:** update every `warms.heldIds` / `warms.isEmpty` / `warms.residentBytes` read in
  the suites to the `(scope: PerformScope())` method form. This is the bulk of that option's churn — expect
  ~15–20 read sites in `WarmStoreTests` alone.

**Done-check:** `swift build --build-tests` is clean, and the full suite is green headless (no behaviour
assertions changed). Report the gate/compile-negative decision explicitly in your summary.

---

## 5. Release sequencing (the breaking-change discipline)

1. **Grep every consumer** for the gated door names before tagging — nothing must call them without `scope:`
   after the bump: `grep -rn '\.\(advance\|release\|releaseAll\|warm\|finishWarm\|heldCache\|evictSessions\)(' <consumer packages>`.
2. Tag a new module version (this is a **breaking** public-API bump — treat it as such in the version number).
3. Bump CyberBench's pin to it, resolve, then land §3's caller edits **in the same PR/release** as the pin
   bump — the API break and the caller fix cannot be split across releases or the consumer won't build.
4. Sanity: run the app once and confirm the cache still reuses (`reuse: … HIT`, low `prefilled`) — the codec
   is untouched so this should be unchanged, but it's the cheap end-to-end confirmation.

---

## Appendix A — gate the three public reads (optional hardening, recommended)

`WarmStore.swift`, convert the three properties to scoped methods (property can't take a parameter):

```swift
// BEFORE
public var heldIds: [UUID] { Array(live.keys) }
public var isEmpty: Bool { live.isEmpty }
public var residentBytes: Int { live.values.reduce(0) { $0 + $1.bytes } }

// AFTER
public func heldIds(scope: borrowing PerformScope) -> [UUID] { Array(live.keys) }
public func isEmpty(scope: borrowing PerformScope) -> Bool { live.isEmpty }
public func residentBytes(scope: borrowing PerformScope) -> Int { live.values.reduce(0) { $0 + $1.bytes } }
```

Then update read call sites in the two consumer files (§3.5) and the test suites. **Heads-up — the SAME macro
boundary as §4-3, and here it dominates the churn:** nearly every suite read is *inside* an assertion
(`#expect(warms.isEmpty)`, `#expect(warms.residentBytes == 0)`, `#expect(warms.heldIds.count == 3)`). Once the
property becomes a method that borrows a `~Escapable` scope, each of those hits the macro wall and must be
pulled into a local first:

```swift
// WRONG: #expect(warms.isEmpty(scope: scope))   — ~Escapable can't cross the macro
let empty = warms.isEmpty(scope: scope);        #expect(empty)
let bytes = warms.residentBytes(scope: scope);  #expect(bytes == 0)
let ids   = warms.heldIds(scope: scope);        #expect(ids.count == 3)
```

So the ~15–20 read sites are ~15–20 **two-line bind-then-assert rewrites**, not one-line swaps. After this,
**no** public path — mutator or read — can touch `live` off-`perform`; the `@unchecked Sendable` invariant is
fully compile-enforced for external callers, and ticket §3.5 can be marked *closed* rather than accepted.

Cost: one extra breaking-signature change (already breaking, so free on the version), plus the ~15–20 read-site
edits — but each a two-line rewrite through the macro boundary, so budget **an afternoon, not an hour**, if you
take this. Value: the invariant stops being "airtight for writes, convention for reads."

---

## Appendix B — non-blocking cleanup (ticket §7.3)

`SessionStore.victimsOverBudget` recomputes `WarmStore.footprint(cache)` inside the sort comparator and again
per victim — O(n log n) full cache walks, where `WarmStore` caches the value in `Entry.bytes`. Correct but
wasteful. Map `id → footprint` once, then sort. Separate from this change; do it whenever `SessionStore`
grows an `Entry` of its own.
