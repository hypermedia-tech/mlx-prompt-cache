# Gate `WarmStore`'s public reads — before/after

**Status: APPLIED to the `MLXPromptCache` package — builds, 29 tests pass.** `heldIds`/`isEmpty`/`residentBytes`
are now `func …(scope:)`; the module's own `victimsOverBudget` uses a private ungated `residentBytesUnchecked`;
`WarmStoreTests` reads are the one-line `(scope: scope)` form; and `MLXPromptCacheBench`'s two off-`perform`
reads were wrapped into `perform` blocks to obtain a scope. Ticket §3.5 is **closed**. The consumer edits in §2
(`TanukiPlatform`) are the only part not applied/built here.

**What:** convert the three `WarmStore` **public** reads — `heldIds` / `isEmpty` / `residentBytes` — into
`PerformScope`-gated methods, so **no** public path (mutator *or* read) can touch `live` off-`perform`.

**Why:** the applied mutator gate makes an off-`perform` structural write a compile error, but a read racing an
in-`perform` write is still UB (ticket §3.5). These three getters are the last public seam left on the
convention. Gating them makes the `@unchecked Sendable` invariant compile-enforced end-to-end for external
callers.

**Ships with:** the same breaking release as the mutator gate — it's another breaking-signature change, free on
a version that is already breaking. `PerformScope` already exists (declared in `PromptCacheCoordinator.swift`),
so no new type.

---

## 1. Module — `Sources/MLXPromptCache/WarmStore.swift`

A property can't take a parameter, so each becomes a method. Bodies are identical.

**BEFORE** (≈ lines 103–105):

```swift
public var heldIds: [UUID] { Array(live.keys) }
public var isEmpty: Bool { live.isEmpty }
public var residentBytes: Int { live.values.reduce(0) { $0 + $1.bytes } }
```

**AFTER:**

```swift
public func heldIds(scope: borrowing PerformScope) -> [UUID] { Array(live.keys) }
public func isEmpty(scope: borrowing PerformScope) -> Bool { live.isEmpty }
public func residentBytes(scope: borrowing PerformScope) -> Int { live.values.reduce(0) { $0 + $1.bytes } }
```

Nothing else in `WarmStore` changes. `SessionStore`'s `residentBytes` / `victimsOverBudget` are `package`, not
public — not part of this gap; leave them.

---

## 2. Caller — CyberBench (`MLXConversationEngine.swift`, `MLXContextWarmer.swift`)

Grep both adapters first:

```
grep -n 'warms\.\(heldIds\|isEmpty\|residentBytes\)' Packages/Infrastructure/Sources/.../Adapters
```

- **If none** (the likely case — neither adapter reads these today, they call only through the coordinator):
  **no caller change.** This gap-closer is module + tests only.
- **For any found:** the read is already inside a `perform` block that mints
  `let scope = coordinator.scope(context)` (from the mutator migration), so just append the argument —
  `warms.isEmpty` → `warms.isEmpty(scope: scope)`. (Plain token, so even a read inside a `#expect`/`#require`
  crosses fine — see §3; no special handling anywhere.)

---

## 3. Test read-site rewrites — ✅ APPLIED (library suite done)

**This is done** in `Tests/MLXPromptCacheTests/WarmStoreTests.swift` — all reads are the one-line
`(scope: scope)` form and the suite is green. Nothing to apply for the library. The pattern below is retained
as reference (and for any CyberBench-side suite that reads these three). `SessionStoreTests` reads none of them.
The tests already mint `let scope = PerformScope()` from the mutator-gate migration, so only the reads changed —
do **not** re-add the scope line.

**No macro caveat — these are one-line swaps.** `PerformScope` is a plain (`Escapable`) value, so it crosses
`#expect` / `#require` with no trouble. Append the argument in place:

```swift
// BEFORE                              →  AFTER
#expect(warms.isEmpty)                    #expect(warms.isEmpty(scope: scope))
#expect(warms.isEmpty == false)           #expect(warms.isEmpty(scope: scope) == false)
#expect(warms.residentBytes == 0)         #expect(warms.residentBytes(scope: scope) == 0)
#expect(warms.residentBytes > 0)          #expect(warms.residentBytes(scope: scope) > 0)
#expect(warms.heldIds.count == 3)         #expect(warms.heldIds(scope: scope).count == 3)
#expect(warms.heldIds == [b])             #expect(warms.heldIds(scope: scope) == [b])
```

**Count:** ~15–20 sites across `WarmStoreTests` — every current `warms.isEmpty` / `warms.residentBytes` /
`warms.heldIds`. One-line swaps; budget ~an hour including the module signature change.

(Historical note: an earlier `~Escapable` draft of `PerformScope` could NOT cross the `#expect`/`#require` macro
and would have forced two-line bind-then-assert rewrites here. That type was dropped for toolchain portability,
so these stay one-liners.)

**Done-check:** `swift build --build-tests` is clean and the suite is green headless. No assertion *meaning*
changes — only the read splits into bind-then-assert.

---

## 4. Release note

Breaking public API — fold into the mutator-gate release (ticket §7.2), never a separate cut. After this lands,
ticket §3.5 is **closed** rather than accepted: every public path to `live`, read or write, is compile-gated to
inside `perform`.
