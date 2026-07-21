# SessionStore Eviction — the unenforced concurrency invariant

**Status:** the `MLXPromptCache` package change (§3, §6) is **applied and verified** — it builds and all 29
SessionStore/WarmStore tests pass. Consumer edits (§5) are specified but **not yet applied**. Option A vs B/C is
still the open decision.
**Scope:** `MLXPromptCache` — `PromptCacheCoordinator` cache-mutating doors (+ `WarmStore` public mutators).
**Consumers pending:** `MLXConversationEngine.swift`, `MLXContextWarmer.swift` (separate `TanukiPlatform` package).
**Provenance:** raised while adding test coverage for §3.1 (app-budgeted session eviction, `5be44ea`).

> **On the AFTER code:** every BEFORE block is copied verbatim from the source as it was before the change. The
> AFTER blocks were **applied to the `MLXPromptCache` package and verified**: it builds and all 29
> SessionStore/WarmStore tests pass on the installed toolchain — `~Escapable`, `@lifetime(immortal)`,
> `@lifetime(borrow context)` and the `LifetimeDependence` flag all compile. The two consumers in the separate
> `TanukiPlatform` package (§5, §5.1) were **not** built here; their edits are specified but unverified. Line
> references are exact as of writing.

---

## 1. The problem, precisely

`SessionStore.live` is a plain `[UUID: [KVCache]]` mutated with **no lock**. The type is `@unchecked Sendable`,
so the compiler checks nothing. Safety rests on one convention: *every* caller is inside
`ModelContainer.perform`, which serialises model access. Concurrent structural mutation of a Swift `Dictionary`
from two threads is **undefined behaviour** (bucket reallocation → heap corruption).

The §3.1 eviction door did not create this — `advance`/`release` already ride the same convention — but it
**widened the blast radius**. Eviction's natural trigger is memory pressure / a timer / a RAM callback, i.e.
*off* the generation path, which is exactly where the convention is most likely to be broken. `advance` and
`release` happen to be called on the generation path by construction; `evictSessions` invites an off-`perform`
call site, and its signature (`overBudget:keep:`) gives the compiler nothing to object to.

**This is not a bug today** — the sole consumer calls every door inside `perform` (see §5). It is an
unenforceable public contract on the most misuse-prone method. The fix makes the contract a compile error.

---

## 2. Why an actor / a Mutex are the wrong tools

**Actor — reject.** The serialisation primitive is not an actor; mlx-swift-lm built `SerialAccessContainer`
*because* an actor cannot hold exclusion across an `async` body. Verbatim
(`mlx-swift-lm/Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift:3-6`):

```
/// This is used as a building block for ``SerialAccessContainer``.  Normal locks
/// do not work with `async` blocks and an `actor` does not guarantee exclusive access
/// for the duration of an `async` function.
```

An actor also cannot hand its isolated non-`Sendable` `[KVCache]` out to the model call. Wrong tool.

**Mutex — reject as a half-measure.** A `Mutex` round `live` makes the *dictionary* safe but not the returned
caches: `advance` returns a `[KVCache]` that lives on outside the lock while the model mutates its buffers. It
protects the map, not the contents — false confidence. This is the type comment's own objection, and it is
correct for the cache-returning methods.

The invariant to preserve is: *touching `live` happens only while the `perform` critical section is held.* The
idiomatic way to enforce that in Swift 6.2 is a **proof-of-context token** the caller cannot fabricate.

---

## 3. The fix — a `PerformScope` witness (Option A)

A value obtainable **only** inside `perform`, required by every cache-mutating door. External callers cannot
construct it (its initialiser is `package`); the only public way to get one needs the `ModelContext` that
`perform` hands in — and that context is non-`Sendable`, so it cannot escape the closure. A cache-mutating call
from off the model queue therefore **does not compile**. `~Escapable` so the token cannot be stashed and reused
after `perform` returns.

`SessionStore` and `WarmStore` are unchanged — their mutating methods are `package`, so external code already
cannot reach them; the only external path to `live` is the coordinator's public doors, which is what we gate.

### 3.1 `PromptCacheCoordinator` — add the witness type and its minter

**BEFORE** (`Sources/MLXPromptCache/PromptCacheCoordinator.swift:39-42`, verbatim):

```swift
public final class PromptCacheCoordinator: Sendable {
    private let store: PromptCacheStore

    public init(store: PromptCacheStore) { self.store = store }
```

**AFTER** (new `PerformScope` type at file scope + a `scope(_:)` minter on the class):

```swift
/// Proof that the holder is executing inside `ModelContainer.perform`. Every `PromptCacheCoordinator`
/// door that mutates the live-cache stores (`SessionStore`/`WarmStore`) requires one. External callers
/// cannot construct it — the initialiser is `package` — so the ONLY way to obtain one is
/// `PromptCacheCoordinator.scope(_:)`, which needs a `ModelContext` and can therefore only be called from
/// inside a `perform` block. A cache-mutating call from off the model queue is a compile error, not a
/// runtime race. `~Escapable` so it cannot be stashed and reused after the `perform` returns.
public struct PerformScope: ~Escapable {
    @lifetime(immortal)
    package init() {}
}

public final class PromptCacheCoordinator: Sendable {
    private let store: PromptCacheStore

    public init(store: PromptCacheStore) { self.store = store }

    /// Mint a `PerformScope`. Requires the `ModelContext` handed to a `ModelContainer.perform` closure —
    /// non-`Sendable`, so it cannot escape that closure — meaning a scope can only be created inside
    /// `perform`. The single door to the cache-mutating API's proof-of-context.
    @lifetime(borrow context)
    public func scope(_ context: borrowing ModelContext) -> PerformScope { PerformScope() }
```

### 3.2 `PromptCacheCoordinator` — the session-door extension

**BEFORE** (`PromptCacheCoordinator.swift:325-359`, verbatim):

```swift
extension PromptCacheCoordinator {
    /// Consumer-facing turn driver — the only public door to the live caches. Requires `model` (only
    /// reachable via `context.model` inside `perform`), nudging "touch caches inside perform only" at the
    /// type level. Seeds conversation `id` from the durable disk root (`store.reuse(forTokens: rootTokens)`)
    /// on the first turn; thereafter the held cache is extended in place, never reloaded.
    public func advance(
        _ sessions: SessionStore,
        id: UUID,
        fullPromptTokens: [Int],
        rootTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters
    ) -> (input: LMInput, cache: [KVCache]) {
        sessions.advance(
            id: id,
            fullPromptTokens: fullPromptTokens,
            warmRoot: { store.reuse(forTokens: rootTokens) },
            makeCache: { makePromptCache(model: model, parameters: parameters) }
        )
    }

    /// Free conversation `id`'s live cache. Idempotent. Call inside `perform`.
    public func release(_ sessions: SessionStore, id: UUID) {
        sessions.release(id)
    }
    
    /// Evict the largest held sessions over an APP-SUPPLIED byte budget, keeping `keep`. Unlike the warm-side
    /// budget (which `WarmStore` stores at init), the session budget is passed in — the app owns it, resolving
    /// live system RAM. NO persist-before-release: a session's durable source is the day-chunked log
    /// (reassemble on next resume), so eviction just drops RAM. Idempotent. Call inside `perform` — the same
    /// contract as `release`.
    public func evictSessions(_ sessions: SessionStore, overBudget budgetBytes: Int, keep: UUID) {
        for id in sessions.victimsOverBudget(budgetBytes, excluding: keep) { sessions.release(id) }
    }
}
```

**AFTER** (each door gains `scope: borrowing PerformScope`; bodies unchanged):

```swift
extension PromptCacheCoordinator {
    /// Consumer-facing turn driver — the only public door to the live caches. Requires a `PerformScope`
    /// (obtainable ONLY inside `perform`, see `scope(_:)`), so a call from off the model queue does not
    /// compile — the enforced form of the old "pass `model` as a nudge". Seeds conversation `id` from the
    /// durable disk root on the first turn; thereafter the held cache is extended in place, never reloaded.
    public func advance(
        _ sessions: SessionStore,
        id: UUID,
        fullPromptTokens: [Int],
        rootTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters,
        scope: borrowing PerformScope
    ) -> (input: LMInput, cache: [KVCache]) {
        sessions.advance(
            id: id,
            fullPromptTokens: fullPromptTokens,
            warmRoot: { store.reuse(forTokens: rootTokens) },
            makeCache: { makePromptCache(model: model, parameters: parameters) }
        )
    }

    /// Free conversation `id`'s live cache. Idempotent. The `PerformScope` gates this to inside `perform`.
    public func release(_ sessions: SessionStore, id: UUID, scope: borrowing PerformScope) {
        sessions.release(id)
    }

    /// Evict the largest held sessions over an APP-SUPPLIED byte budget, keeping `keep`. NO
    /// persist-before-release: a session's durable source is the day-chunked log (reassemble on next
    /// resume), so eviction just drops RAM. Idempotent. The `PerformScope` gates this to inside `perform` —
    /// the same domain as `advance`/`release`, now enforced by the type system rather than a doc comment.
    public func evictSessions(
        _ sessions: SessionStore,
        overBudget budgetBytes: Int,
        keep: UUID,
        scope: borrowing PerformScope
    ) {
        for id in sessions.victimsOverBudget(budgetBytes, excluding: keep) { sessions.release(id) }
    }
}
```

### 3.3 `PromptCacheCoordinator` — the warm-side doors

Three doors mutate `WarmStore` and need the same gate.

**BEFORE** — `warm(_ warms:…)` signature (`PromptCacheCoordinator.swift:201-209`) and its internal
`finishWarm` call (`:236`):

```swift
    public func warm(
        _ warms: WarmStore,
        id: UUID,
        promptTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters,
        persist: WarmPersistence = .onCompletion,
        shouldPause: () -> Bool = { false }
    ) -> PromptWarmOutcome {
```

```swift
            // Already at the boundary: persist if we are holding unpersisted work, then finish.
            return finishWarm(warms, id: id, model: model)
```

**AFTER**:

```swift
    public func warm(
        _ warms: WarmStore,
        id: UUID,
        promptTokens: [Int],
        model: any LanguageModel,
        parameters: GenerateParameters,
        scope: borrowing PerformScope,
        persist: WarmPersistence = .onCompletion,
        shouldPause: () -> Bool = { false }
    ) -> PromptWarmOutcome {
```

```swift
            // Already at the boundary: persist if we are holding unpersisted work, then finish.
            return finishWarm(warms, id: id, model: model, scope: scope)
```

**BEFORE** — `finishWarm` (`:292-297`) and `heldCache` (`:307-313`):

```swift
    @discardableResult
    public func finishWarm(
        _ warms: WarmStore,
        id: UUID,
        model: any LanguageModel
    ) -> PromptWarmOutcome {
```

```swift
    public func heldCache(
        _ warms: WarmStore,
        id: UUID,
        model: any LanguageModel
    ) -> [KVCache]? {
        warms.entry(id)?.cache
    }
```

**AFTER**:

```swift
    @discardableResult
    public func finishWarm(
        _ warms: WarmStore,
        id: UUID,
        model: any LanguageModel,
        scope: borrowing PerformScope
    ) -> PromptWarmOutcome {
```

```swift
    public func heldCache(
        _ warms: WarmStore,
        id: UUID,
        model: any LanguageModel,
        scope: borrowing PerformScope
    ) -> [KVCache]? {
        warms.entry(id)?.cache
    }
```

**Also — three INTERNAL `warms.release(...)` calls must forward the scope.** `warm(_ warms:)` and `finishWarm`
call `warms.release` themselves; once §3.4 gates that method, these internal callers stop compiling until the
in-scope `scope` is forwarded. The build fails on exactly these three lines if they are missed.

**BEFORE** (`warm`'s budget loop `:270-273`, `warm`'s completion release `:280`, `finishWarm` `:299-300`):

```swift
        for victim in warms.victimsOverBudget(excluding: id) {
            persistHeld(warms, id: victim)
            warms.release(victim)
        }
        …
            if shouldWrite { warms.release(id) }
        …
        persistHeld(warms, id: id)
        warms.release(id)
```

**AFTER**:

```swift
        for victim in warms.victimsOverBudget(excluding: id) {
            persistHeld(warms, id: victim)
            warms.release(victim, scope: scope)
        }
        …
            if shouldWrite { warms.release(id, scope: scope) }
        …
        persistHeld(warms, id: id)
        warms.release(id, scope: scope)
```

### 3.4 `WarmStore` — gate the PUBLIC mutators (the hole the coordinator doesn't cover)

Unlike `SessionStore` (whose mutators are all `package`, unreachable externally), `WarmStore` exposes two
**public** methods that mutate `live` directly, bypassing the coordinator entirely. `releaseAll` is even
documented for "memory pressure, shutdown" — the exact off-`perform` call site this whole change targets. They
must take the same witness, or the enforcement leaks.

**BEFORE** (`Sources/MLXPromptCache/WarmStore.swift:93-100`, verbatim):

```swift
    // MARK: - Public lifecycle

    /// Free one warm's live cache. Idempotent. **Does not persist** — use
    /// `PromptCacheCoordinator.finishWarm` to keep the work. Call inside `perform`.
    public func release(_ id: UUID) { live[id] = nil }

    /// Free every held cache (model swap, memory pressure, shutdown). Call inside `perform`.
    public func releaseAll() { live.removeAll() }
```

**AFTER** (both gain `scope: borrowing PerformScope`; bodies unchanged):

```swift
    // MARK: - Public lifecycle

    /// Free one warm's live cache. Idempotent. **Does not persist** — use
    /// `PromptCacheCoordinator.finishWarm` to keep the work. The `PerformScope` gates this to inside `perform`.
    public func release(_ id: UUID, scope: borrowing PerformScope) { live[id] = nil }

    /// Free every held cache (model swap, memory pressure, shutdown). The `PerformScope` gates this to inside
    /// `perform` — the "memory pressure" caller must route through the model queue, not call from a bare handler.
    public func releaseAll(scope: borrowing PerformScope) { live.removeAll() }
```

`PerformScope` lives in the same module (declared in `PromptCacheCoordinator.swift`, §3.1), so `WarmStore` can
reference it with no import change.

### 3.5 What is intentionally NOT gated, and the read residual

- **`prepare(...)` and the plain `warm(promptTokens:model:parameters:shouldPause:)`** touch only the
  `Sendable` `PromptCacheStore` (disk + its `Mutex`-guarded catalog), never `SessionStore`/`WarmStore`'s `live`.
  They are safe off-`perform` and stay ungated — do **not** add a scope to them.
- **`SessionStore`'s `advance`/`release`/`residentBytes`/`victimsOverBudget`** are `package`: external code
  cannot reach them, and the in-package callers (the coordinator's already-gated doors, and the tests) are
  trusted. No change.
- **Read residual — a decision.** `WarmStore`'s public `heldIds` / `isEmpty` / `residentBytes` *read* `live`. A
  read concurrent with an in-`perform` write is also UB, strictly. This note gates the **mutators** (where the
  heap-corrupting structural writes happen) and leaves these three reads on the `perform` convention, because
  gating them forces every metrics/logging read to carry a scope for little practical gain. If you want the
  invariant fully airtight, convert them to `func heldIds(scope:)` etc. the same way — flagged, not done, so the
  boundary is explicit rather than silently assumed safe.

### 3.6 `Package.swift` — enable the nonescapable/lifetime features

`~Escapable` and the `@lifetime` attribute are lifetime-dependency features. On some 6.x toolchains they compile
as-is; on others they need the experimental feature enabled on the target. Add it defensively; drop it if the
build accepts the attributes without it (see §7.1).

**BEFORE** (`Package.swift:28-34`, verbatim):

```swift
        .target(
            name: "MLXPromptCache",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
```

**AFTER**:

```swift
        .target(
            name: "MLXPromptCache",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("LifetimeDependence"),
            ]
        ),
```

(`swift-tools-version` is already `6.3` and `swiftLanguageModes: [.v6]`; no change there.)

---

## 4. What `@unchecked` this does and does not delete

- **External callers** (everyone outside the package — including `MLXConversationEngine`): every public path
  that *mutates* `live` is now gated — the coordinator session/warm doors (§3.2–3.3) **and** `WarmStore`'s own
  public `release`/`releaseAll` (§3.4). Off-`perform` mutation becomes a compile error. Public *reads* remain on
  the convention by choice (§3.5).
- **`@unchecked Sendable` on `SessionStore`/`WarmStore` stays.** They still hold non-`Sendable` `[KVCache]` and
  are captured into the `@Sendable` perform closure, so they must remain `Sendable`. Deleting `@unchecked`
  entirely needs the larger refactor (caches living *inside* `ModelContext`, no free-standing store object) —
  out of scope for this note.
- Net: the risk that this note is about — an off-`perform` `evictSessions` from a pressure handler — is
  eliminated at compile time. The residual trust (`package` methods, in-package tests) is confined to the
  library's own code, not its public surface.

---

## 5. Consumer changes — `MLXConversationEngine.swift`

The consumer already calls every door **inside** `container.perform { context in … }` (and `endConversation`
inside `await container.perform`). So the change is mechanical: mint one scope per `perform` and thread it.

**BEFORE** — inside `conversationTurnStream`'s `container.perform { context in … }`:

- the bank warm (`:146-153`):

```swift
                        let banked = coordinator.warm(
                            warms,
                            id: warmID,
                            promptTokens: Array(rootTokens.prefix(bankBoundary + 1)),
                            model: context.model,
                            parameters: params,
                            persist: .onCompletion,
                            shouldPause: { Task.isCancelled })
```

- the finish on cancel (`:185`): `coordinator.finishWarm(warms, id: warmID, model: context.model)`
- the seed (`:198-201`) and reseed (`:211-214`): `coordinator.advance(sessions, id: …, model: context.model, parameters: params)`
- the diverged-prefix release (`:210`, `:219`): `coordinator.release(sessions, id: conversationId)`

**AFTER** — add one line at the top of the `perform` closure, then pass `scope:` at every call:

```swift
        let raw = try await container.perform { context in
            let scope = coordinator.scope(context)   // proof-of-perform, minted once per block
            …
                        let banked = coordinator.warm(
                            warms,
                            id: warmID,
                            promptTokens: Array(rootTokens.prefix(bankBoundary + 1)),
                            model: context.model,
                            parameters: params,
                            scope: scope,
                            persist: .onCompletion,
                            shouldPause: { Task.isCancelled })
            …
                        coordinator.finishWarm(warms, id: warmID, model: context.model, scope: scope)
            …
                var advanced = coordinator.advance(
                    sessions, id: conversationId,
                    fullPromptTokens: fullTokens, rootTokens: rootTokens,
                    model: context.model, parameters: params, scope: scope)
            …
                    coordinator.release(sessions, id: conversationId, scope: scope)
```

`coordinator` is currently constructed *inside* the `perform` closure (`:85`), so `scope` sits naturally beside
it. `endConversation` (`:267-269`) gets the same one-line treatment:

```swift
        await container.perform { context in
            let scope = coordinator.scope(context)
            coordinator.release(sessions, id: conversationId, scope: scope)
        }
```

(Note: `endConversation` currently binds `perform { _ in … }` — the `_` becomes `context` to mint the scope.)

**Where `evictSessions` lands:** the §3.1 plan calls it inside this same `perform`, right after `advance`:

```swift
                coordinator.evictSessions(sessions, overBudget: sessionBudgetBytes(),
                                          keep: conversationId, scope: scope)
```

It cannot be written anywhere else — no `scope`, no call. That is the whole point.

### 5.1 `MLXContextWarmer.swift` — the second consumer (read and verified)

`MLXContextWarmer` shares the same `WarmStore` and is the other driver of these doors. Read in full: it makes
**one** gated call — `coordinator.warm(...)` — and it is already inside `container.perform { context in … }`. It
does **not** call `warms.release` or `warms.releaseAll` anywhere (the `.uncacheable` branch at `:150-161`
deliberately does *not* release). So the only edit here is threading a scope into the one `warm` call.

**BEFORE** (`MLXContextWarmer.swift:126-134`, verbatim):

```swift
            let outcome = coordinator.warm(
                warms,
                id: warmID,
                promptTokens: capped,
                model: context.model,
                parameters: GenerateParameters(),
                persist: .everyTokens(Self.bankInterval),
                shouldPause: { shouldYield() || Task.isCancelled }
            )
```

**AFTER**:

```swift
            let scope = coordinator.scope(context)
            let outcome = coordinator.warm(
                warms,
                id: warmID,
                promptTokens: capped,
                model: context.model,
                parameters: GenerateParameters(),
                scope: scope,
                persist: .everyTokens(Self.bankInterval),
                shouldPause: { shouldYield() || Task.isCancelled }
            )
```

**Both consumers now confirmed by reading**: `MLXConversationEngine` (§5) and `MLXContextWarmer` (here) call
only through the coordinator, only inside `perform`, and neither calls `WarmStore.release`/`releaseAll`
directly. The `WarmStore` public-mutator gate (§3.4) therefore closes a latent public hole rather than fixing a
present call site — worth keeping, but no in-repo consumer edit follows from it.

### 5.2 In-repo executables — `MLXPromptCacheScratch` and `MLXPromptCacheBench` (APPLIED)

Not app consumers, but they call the gated doors and are built by this package, so they had to change in the
same commit or the package would not link. Both were updated and are part of the verified build. Each call sits
inside an `mc.perform { ctx in … }` closure; a scope is minted per closure with `coordinator.scope(ctx)` (or
`coord.scope(ctx)`) and threaded through.

- **`Sources/MLXPromptCacheScratch/main.swift`** — one `perform` closure: `advance` ×2 + `release` ×1. Added
  `let scope = coordinator.scope(context)` at the top of the closure; passed `scope: scope` to all three.

- **`Sources/MLXPromptCacheBench/main.swift`** — six gated calls across several closures: `warm(_ warms:)` ×2,
  `heldCache` ×2, `finishWarm` ×2. Each closure mints `let scope = coord.scope(ctx)` (the two one-line
  `finishWarm` calls take `scope: coord.scope(ctx)` inline) and threads it.

The plain `coord.warm(promptTokens:…)` and `coord.prepare(…)` calls in both files are ungated (§3.5) and were
left as-is.

---

## 6. Test changes — exact before/after

The headless suites call the coordinator doors with a `StubModel` and no `ModelContext`. They mint the token via
the `package` initialiser (reachable under `@testable import`), so **no context is needed in tests** — this is
what keeps the suites headless, and is the reason the initialiser is `package` rather than tied to `scope(_:)`.

### 6.0 Migration rule

In every test that calls a gated door, add one line near the top and pass `scope: scope` at each call:

```swift
let scope = PerformScope()
```

Direct `sessions.advance(…)` / `warms.*` calls (the `package` store methods the tests also use) are **unchanged**
— they never went through the coordinator, so they were never gated.

### 6.1 `SessionStoreTests` — the three gated call sites

**Test 8, BEFORE** (`SessionStoreTests.swift:164-171`):

```swift
        let (delta, cache) = coord.advance(sessions, id: id,
                                           fullPromptTokens: Fixture.tokens(512 + 20),
                                           rootTokens: root, model: twoLayerModel(),
                                           parameters: GenerateParameters())
        #expect(PromptCacheIO.tokenLength(cache) == 512)                                   // seeded from the durable root
        #expect((delta.text.tokens.shape.last ?? 0) == 20)                                // only the new turn

        coord.release(sessions, id: id)
```

**Test 8, AFTER** (mint once after `let id = UUID()`, thread it):

```swift
        let scope = PerformScope()
        let (delta, cache) = coord.advance(sessions, id: id,
                                           fullPromptTokens: Fixture.tokens(512 + 20),
                                           rootTokens: root, model: twoLayerModel(),
                                           parameters: GenerateParameters(), scope: scope)
        #expect(PromptCacheIO.tokenLength(cache) == 512)                                   // seeded from the durable root
        #expect((delta.text.tokens.shape.last ?? 0) == 20)                                // only the new turn

        coord.release(sessions, id: id, scope: scope)
```

**Test 13, BEFORE** (`:246`) → **AFTER**:

```swift
        coord.evictSessions(sessions, overBudget: fKeep, keep: keep)
```
```swift
        let scope = PerformScope()
        coord.evictSessions(sessions, overBudget: fKeep, keep: keep, scope: scope)
```

**Test 14, BEFORE** (`:271`, `:274`, `:276`) → **AFTER** (one `let scope = PerformScope()` after `let before…`):

```swift
        coord.evictSessions(sessions, overBudget: before * 2, keep: keep)       // budget above resident
        …
        coord.evictSessions(sessions, overBudget: 1, keep: keep)               // now shed `other`
        …
        coord.evictSessions(sessions, overBudget: 1, keep: keep)               // idempotent
```
```swift
        coord.evictSessions(sessions, overBudget: before * 2, keep: keep, scope: scope)   // budget above resident
        …
        coord.evictSessions(sessions, overBudget: 1, keep: keep, scope: scope)            // now shed `other`
        …
        coord.evictSessions(sessions, overBudget: 1, keep: keep, scope: scope)            // idempotent
```

### 6.2 `WarmStoreTests` — every `coord.warm` / `heldCache` / `finishWarm`

All ~24 coordinator calls across the suite gain `scope:`. Representative before/after for each door:

```swift
// warm — BEFORE (:50)
let out = coord.warm(warms, id: id, promptTokens: tokens, model: twoLayerModel(),
                     parameters: params, shouldPause: { true })
// warm — AFTER
let out = coord.warm(warms, id: id, promptTokens: tokens, model: twoLayerModel(),
                     parameters: params, scope: scope, shouldPause: { true })

// finishWarm — BEFORE (:167)        →  AFTER
_ = coord.finishWarm(warms, id: id, model: model)
_ = coord.finishWarm(warms, id: id, model: model, scope: scope)
```

**`heldCache` needs care — a `~Escapable` scope cannot cross a `#require`/`#expect` macro boundary.** The
compiler rejects `try #require(coord.heldCache(…, scope: scope))` ("requires that 'PerformScope' conform to
'Escapable'"). Pull the call into a local first, then assert on the local:

```swift
// BEFORE (:76)
let first = try #require(coord.heldCache(warms, id: id, model: model))
// AFTER — bind, then require:
let firstHeld = coord.heldCache(warms, id: id, model: model, scope: scope)
let first = try #require(firstHeld)
```

The same rewrite applies to the two `#expect(… heldCache(…) …)` sites — `completionUnderNeverKeepsTheCacheResident`
and `hybridWarmHoldsAcrossPausesAndCompletes`: bind `let held = coord.heldCache(…, scope: scope)` on its own
line, then `#expect(… held ?? [] …)`. (This was the one compiler-forced deviation from the doc's original
before/after; everything else applied verbatim.)

Note the argument order: `scope:` sits **before** the defaulted `persist:`/`shouldPause:`, so calls that omit
those defaults still read naturally. Behaviour and every `#expect` are unchanged.

`releaseAllDropsEverything` also calls the newly-gated `WarmStore.releaseAll()` directly (`:356`):

```swift
// BEFORE (:356)              →  AFTER
warms.releaseAll()
warms.releaseAll(scope: PerformScope())
```

(There are no direct `warms.release(_:)` call sites in the current suite; if any are added, they take the same
`scope:` argument.)

### 6.3 New test — the gate holds at compile time (the point of the whole change)

The guarantee is **compile-time**, so it cannot be a runtime `#expect`. The regression guard is a fixture that
must **fail to compile** — kept out of the test target and checked by a build-negative step (or documented and
reviewed). Exact fixture:

```swift
// NEGATIVE FIXTURE — must NOT compile. An eviction call with no PerformScope is the off-`perform`
// misuse this change forbids; if this ever builds, the gate has regressed.
func offPerformEvictionMustNotCompile(_ coord: PromptCacheCoordinator, _ s: SessionStore, _ keep: UUID) {
    coord.evictSessions(s, overBudget: 1, keep: keep)   // expected error: missing argument for parameter 'scope'
}
```

Swift Testing has no built-in "expect-no-compile", so this is enforced by CI (`swift build` on a
`CompileNegative` target expected to fail) or by review, not by the suite. State it explicitly — a silent
absence would read as "covered".

### 6.4 New test — a scope minted for one turn does not leak to the next

`~Escapable` is what stops a stashed token from re-authorising a later off-`perform` call. If §7.1 forces the
fallback to a plain (escapable) `PerformScope`, add this positive test to pin the weaker guarantee that at least
the *eviction behaviour* is unchanged under a freshly-minted token each call:

```swift
@Test func evictionUsesAFreshScopePerCall() throws {
    let (_, _, coord) = try makeStore()
    let sessions = SessionStore()
    let keep = UUID(), victim = UUID()
    let fKeep = WarmStore.footprint(seed(sessions, keep, cache: Fixture.syntheticCache(tokens: 256)))
    seed(sessions, victim, cache: Fixture.syntheticCache(tokens: 1024))
    coord.evictSessions(sessions, overBudget: fKeep, keep: keep, scope: PerformScope())
    #expect(sessions.residentBytes == fKeep)   // victim shed; keep survives — identical to test 13
}
```

Under the `~Escapable` design this is redundant with test 13; under the escapable fallback it documents that the
per-call mint is the intended usage. Behavioural coverage of eviction itself is already complete (tests 9–14).

---

## 7. Open items before merge

1. **Toolchain — CONFIRMED.** `~Escapable` + `@lifetime(immortal)` / `@lifetime(borrow context)` + the
   `LifetimeDependence` flag compile as written, and the 29 SessionStore/WarmStore tests pass. One deviation the
   compiler forced: a `~Escapable` scope cannot cross a `#require`/`#expect` macro, so `heldCache(…, scope:)` is
   bound to a local before assertions (§6.2). Fallback to a plain escapable `struct PerformScope { package
   init() {} }` remains available if the lifetime features become a burden — external enforcement is identical,
   the only loss is a stashable token.
2. **Public-API break.** Breaking for every external caller of the gated doors — `MLXConversationEngine`
   (coordinator doors, §5) **and** any `WarmStore.release`/`releaseAll` caller such as `MLXContextWarmer`
   (§5.1). All consumer edits must land in the same release; bump the library major/minor accordingly. Grep
   consumer packages for `.release(`/`.releaseAll(`/`.evictSessions(`/`.advance(`/`.finishWarm(`/`.heldCache(`
   before cutting the release.
3. **`WarmStore.footprint` in the sort comparator** (`SessionStore.victimsOverBudget:63-66`) recomputes the
   footprint on every comparison — O(n log n) full cache walks. Correct, but wasteful; map ids→footprint once,
   then sort. Separate, non-blocking cleanup.

---

## 8. Decision

- **A (this note): `PerformScope` witness** — recommended. Compile-enforces the invariant for all external
  callers, keeps headless tests, small mechanical consumer edit.
- **B: per-store `SerialAccessContainer`, `async` doors** — only if eviction must run outside `perform`; adds a
  second lock domain and a lock-ordering rule (sessions-lock always inside `perform`, never the reverse).
- **C: document as a known limitation** — no code change; honest only if the sole consumer's call is provably
  inside `perform` forever.

The `MLXPromptCache` package changes (§3, §6) are applied and verified (builds, 29 tests pass). The consumer
edits (§5) are specified but not yet applied, and the A-vs-B/C decision is still open — sign-off needed before
the consumer edits land and the API break is cut into a release.
