# SessionCache → SessionStore reshape — implementation plan

**Goal:** make the conversation-cache primitive clean under Swift 6.2 strict concurrency by moving
ownership of the live `[KVCache]` *behind* `ModelContainer`'s serialised access, keyed by `UUID`, so
nothing non-`Sendable` ever crosses an isolation boundary. No back-compat concern — nothing downstream
consumes `SessionCache` yet.

This is a **write-up of the steps**, not the edits themselves. Each step gives the exact target code so
it can be applied verbatim, plus how to verify it. §7 folds in the Swift-concurrency review (verified
against the `mlx-swift-lm` source); §8 covers a separate pre-existing cleanup.

---

## 0. Grounding (read this first)

Written against branch **`feature/fix-concurrency`**, working tree clean. The relevant files as they
stand today:

| File | Role today |
|------|-----------|
| `Sources/MLXPromptCache/SessionCache.swift` | `public final class SessionCache` — **non-`Sendable`**, holds `public private(set) var cache: [KVCache]`. |
| `Sources/MLXPromptCache/PromptCacheCoordinator.swift` | `openSession(rootTokens:model:parameters:) -> SessionCache` at the bottom (line ~187). |
| `Tests/MLXPromptCacheTests/SessionCacheTests.swift` | 7 tests over the class. |
| `Sources/MLXPromptCacheScratch/main.swift` | conversation gate (line ~251) holds a `SessionCache` inside one `perform`. |

Baseline concurrency audit of the rest of the module (all already clean):

- `PromptCacheStore` — `final class … Sendable`, all state behind `Mutex` (`import Synchronization`), `@Sendable` log closure.
- `PromptCacheCoordinator` — `final class … Sendable`, holds only the `Sendable` store.
- `Catalog.Hit/Plan/BoundaryPlan`, `HotCache`, `HotCodec`, `CacheBytes/TensorBytes` — `Sendable` value types.
- `PreparedCache`, `Reused` — **intentionally non-`Sendable`** value types that hold `[KVCache]`; doc-commented "create and use on the model's thread, never send across an isolation boundary."
- `Package.swift` — `swiftLanguageModes: [.v6]`, tools 6.3 ⇒ **complete** concurrency checking is already on. The module does **not** opt into main-actor-default isolation — correct for a reusable library (see §7).

**How `perform` serialises (verified in the dependency).** `ModelContainer` is `final class: Sendable`
and wraps its `ModelContext` in `SerialAccessContainer<ModelContext>` — itself a
`final class @unchecked Sendable` backed by an `AsyncMutex`
(`mlx-swift-lm/Libraries/MLXLMCommon/Utilities/SerialAccessContainer.swift:44`). So `perform` guarantees
**serialised exclusive access** for the whole closure — not OS-thread pinning. That distinction matters
and is used throughout this doc.

So the module is already strict-concurrency-friendly *except* for one thing: `SessionCache` is a
non-`Sendable` **class** that the intended consumer (an `actor`) must hold as isolated state and shuttle
into a `@Sendable perform` closure every turn. That is the single rough edge this reshape removes.

---

## 1. The problem, precisely

`[KVCache]` is non-`Sendable`, and MLX is unsafe under **concurrent** access — so it must be touched only
within `ModelContainer.perform`, which serialises all model access. Today's shape:

```
actor Consumer {
    var sessions: [UUID: SessionCache]   // ← non-Sendable state on the actor
    func turn(...) async {
        try await container.perform { ctx in       // @Sendable closure
            let s = self.sessions[id]              // ← captures non-Sendable across the boundary
            let delta = s.advance(...)
            generate(cache: s.cache, ...)
        }
    }
}
```

The `SessionCache` (and its `[KVCache]`) crosses the actor→executor boundary on **every** turn. Because
`perform`'s action is `@Sendable`, a non-`Sendable` capture only compiles via the `perform(nonSendable:)`
escape hatch (the `SendableBox` path, `ModelContainer.swift:112`) — a compile barrier papered over, not
resolved.

**Target:** the consumer holds a plain `let store` that is `Sendable`, captures it into the `@Sendable`
closure with zero ceremony, passes a `UUID`, and gets back the delta `LMInput` + the live cache to hand
straight to `MLXLMCommon.generate(cache:)`. The `[KVCache]` is created, grown, and freed **entirely
inside the store**, only ever within `perform`.

---

## 2. Design decisions (the load-bearing reasoning)

### 2.1 Why `@unchecked Sendable` is honest here, not a hack

The new `SessionStore` stores `var live: [UUID: [KVCache]]`. `[KVCache]` is non-`Sendable`, so a plain
`Sendable` conformance is rejected by the compiler. We annotate `@unchecked Sendable` with this invariant:

> The `live` map — and every `[KVCache]` in it — is **only ever read or mutated inside
> `ModelContainer.perform`**, which serialises all access to the model. The consumer funnels *all* GPU
> work through `perform`, so there is genuinely no concurrent access to `live`.

Why this is sound, not a shortcut:

1. **The race `Sendable` guards against cannot occur.** `Sendable` checking exists to prevent two
   isolation domains touching shared mutable state concurrently. Here, by construction, all access is
   serialised through the one `perform` domain. The guarantee is supplied by the consumer's serialised
   access — exactly what `@unchecked Sendable` is *for*: "I, the author, guarantee thread-safety by a
   mechanism the compiler can't see."
2. **It mirrors the dependency's own pattern.** `mlx-swift-lm` wraps its non-`Sendable` `ModelContext`
   in `SerialAccessContainer<T>: @unchecked Sendable` (`SerialAccessContainer.swift:44`) for this exact
   reason. `SessionStore` is the same shape one layer up: a `final class @unchecked Sendable` wrapping
   non-`Sendable` state whose access is serialised. We are not inventing a sketchy pattern — we're
   re-using the library's.
3. **A `Mutex` would be the *wrong* tool.** One might reach for `Mutex<[UUID: [KVCache]]>` to get a
   *checked* `Sendable`. But that creates a **second access path** to the caches, reachable from any
   thread *outside* `perform`. A caller could then take the lock and drive MLX off the serialised
   `perform` path — concurrently with `perform`-driven GPU work on the same container — which is the
   concurrent-MLX hazard we must avoid. The mutex would compile green and *invite* the bug. The library
   made the same call: `SerialAccessContainer` is `@unchecked Sendable`, **not** a lock-wrapped Sendable,
   and its own `concurrency.md` explains why (see §2.6, §7). We want to defer to `perform`'s single
   serialised domain, not add a competing one.

### 2.2 Why a `class`, not a `struct`

The consumer holds one `let store` and captures it into the `@Sendable perform` closure. It must be a
**reference type** so the captured handle and the consumer's `let` refer to the *same* map — and so
`generate(cache:)` growing a `[KVCache]` in place is observed by the store's stored reference on the next
turn. A `struct` captured by a `let` closure can't be mutated and would give copy semantics that defeat
"held and extended across turns." ⇒ `final class SessionStore`.

### 2.3 `[KVCache]` in the return type — why it's safe (access, not the signature)

This is the point that (rightly) drew scrutiny; here is the resolution, grounded in the reference (§2.6)
and the dependency (§7), not asserted.

**The real invariant is *where the cache is accessed* — inside the serialised `perform` domain — not
whether the type appears in a signature.** MLX is unsafe under concurrent access; `Sendable` is Swift's
*compile-time proxy* for that runtime law. Therefore:

- `advance` is called **inside `perform`**, and the `[KVCache]` it returns is handed straight to
  `MLXLMCommon.generate(cache:)` **within that same `perform`**. It never escapes the serialised domain,
  so it never breaks the real invariant. This is exactly how the reference passes its live cache around
  (§2.6, lesson 3), and it is enforced by the dependency: `perform`'s return is `sending R` with
  `R: Sendable`, so a `[KVCache]` *cannot* be returned out of `perform` at all — only used in place.
- What the **old** design did wrong was different: the consumer stored the cache on its **actor** and
  re-entered `perform` on later turns, so each turn the cache had to cross the actor→executor boundary.
  *That* crossing is what `Sendable` flags. The new design deletes it — the consumer stores only the
  `Sendable` `SessionStore` + a `UUID`; the `[KVCache]` lives inside the store and is only ever touched
  inside `perform`. **Nothing non-`Sendable` crosses a boundary.**

Two reassurances worth stating explicitly:

- "No `[KVCache]` in a public signature" is a useful *heuristic* for "don't let the consumer hold it as
  isolated state." The `advance` return, produced and consumed inside `perform`, satisfies the actual
  invariant the heuristic protects — it is not a loophole.
- **`@unchecked Sendable` on `SessionStore` does *not* make `[KVCache]` `Sendable`.** Only the store's
  internal map is covered. A consumer that tries to stash the returned live cache into actor/`Sendable`
  state still gets a fresh `Sendable` error from the compiler. The unsafe surface is contained to the
  map; "don't hold the cache" stays compiler-enforced at the consumer.

> **Stricter alternative (the letter of the brief):** have `SessionStore` *drive* generation
> (`generate(id:…,context:) -> [Int]`) so `[KVCache]` appears in no signature at all. Not the default —
> the north-star keeps the consumer calling `MLXLMCommon.generate` directly, and it balloons the store's
> surface (streaming, params, stop conditions). Safety is identical either way (both keep the cache
> inside `perform`), so this is an API-taste choice, not a correctness one.

### 2.4 Type-nudge + narrowed surface: the coordinator is the only public door

Per the brief ("prefer making entry points REQUIRE the model/context so the invariant is nudged by the
type signature") **and** the concurrency review's WARN-1 (minimise the off-`perform`-reachable surface):

- The raw primitives (`advance`/`release`) are **`package`** on `SessionStore` — reachable by the
  coordinator seam and the package's own test/scratch targets, but **not** by external dependents.
- The only **`public`** entry points are on `PromptCacheCoordinator`: `advance(…model:…)` (which requires
  `any LanguageModel`, a value only obtainable from `context.model` inside `perform`) and `release(…)`.

So an external consumer *cannot* reach the live caches except through the coordinator, and advancing a
turn forces them to hold a `model` — which nudges "inside `perform`" at the type level. `release` takes
no model (it's a map removal), but routing it through the coordinator keeps the whole surface uniform and
one-door; it must still be called inside `perform` (same serialised discipline).

### 2.5 Naming

`SessionStore` (brief's suggested name; "name your call"). It is the id-keyed owner of live conversation
caches. Test suite renamed to `SessionStoreTests`.

### 2.6 What oMLX confirms and teaches (reference cross-check)

`~/workspace/learning/omlx` is the Python reference this module ports. Python has no `Sendable` or actor
isolation, so it can't teach *compile-time* checking — but the *runtime* law beneath `Sendable` (no
concurrent MLX access) is the same, and oMLX enforces it explicitly. Three lessons:

1. **One serialised domain owns all cache/GPU work — mandatory, not stylistic.** oMLX runs every MLX op
   on a single-worker pool: `ThreadPoolExecutor(max_workers=1, …)` (`engine_core.py:get_mlx_executor`,
   ~L111), because MLX GPU ops across all models *"MUST be serialized onto one thread"* or Metal
   command-buffer races segfault (issue #85). The Swift equivalent is `ModelContainer.perform`'s
   serialised access. So `@unchecked Sendable` + "only inside `perform`" is the port of oMLX's serialised
   worker, not a workaround for the compiler.
2. **Don't add your own lock — defer to the single domain.** oMLX pins a *thread-local* stream
   (`engine_core.py:_init_mlx_thread`, ~L81); touching an array from another thread fails with *"There is
   no Stream(gpu, 0) in current thread."* Thread-pinning is oMLX's *mechanism* for non-concurrency;
   mlx-swift's mechanism is `SerialAccessContainer`'s `AsyncMutex`. Same end (no concurrent MLX), different
   means. Either way the transferable rule is: route all cache access through the one serialised domain,
   don't create a second one with a `Mutex` (§2.1, point 3).
3. **The live cache is passed around *freely* — inside that domain.** oMLX threads its live `prompt_cache`
   (a `list` of layers) through params, dataclass fields (`_VLMMTPDecodeState.prompt_cache`), and return
   values (`_VLMMTPResponse.prompt_cache`) with no taboo about it "appearing in a signature." The only
   rule is *where* it's touched — the direct answer to the §2.3 worry.

The one genuinely dangerous pattern oMLX documents — reading cache bytes off the serialised domain — is
covered in §6, item 6.

---

## 3. Target API

```swift
// Sources/MLXPromptCache/SessionCache.swift  → rename file to SessionStore.swift (optional but cleaner)

/// Owns the live KV caches for in-flight conversations, keyed by id. The live `[KVCache]` for a
/// conversation is created, grown, and freed entirely inside this type — nothing non-`Sendable` is
/// stored by, or handed for retention to, the consumer.
///
/// `@unchecked Sendable` invariant: `live` (and every `[KVCache]` in it) is only ever read or mutated
/// inside `ModelContainer.perform`, which serialises all model access (via `SerialAccessContainer` /
/// `AsyncMutex`). There is never concurrent access to the map, so the data race `Sendable` guards against
/// cannot occur. This mirrors mlx-swift-lm's own `SerialAccessContainer<T>: @unchecked Sendable`, which
/// wraps the non-`Sendable` `ModelContext` the same way. A `Mutex` is deliberately NOT used: it would add
/// a second access path reachable off `perform` and defeat the single-serialised-domain guarantee.
/// See docs/session-store-reshape.md §2.1 + §7.
///
/// The raw entry points are `package`: reachable by the coordinator seam and the package's own tests,
/// never by external dependents (who use the `public` `PromptCacheCoordinator` doors — §2.4).
public final class SessionStore: @unchecked Sendable {
    private var live: [UUID: [KVCache]] = [:]

    public init() {}

    /// Advance conversation `id` by one turn. Seeds on the FIRST call for `id` — from the durable disk
    /// root (`warmRoot`) if present, else a fresh (hybrid-correct) empty cache from `makeCache`. Returns
    /// ONLY the tokens beyond the cache's resident offset, plus the live cache to generate over.
    /// `warmRoot`/`makeCache` are evaluated at most once (seed only) and never on a resumed turn.
    /// Call only inside `ModelContainer.perform` (see the type invariant).
    package func advance(
        id: UUID,
        fullPromptTokens: [Int],
        warmRoot: () -> Reused?,
        makeCache: () -> [KVCache]
    ) -> (input: LMInput, cache: [KVCache]) {
        let cache: [KVCache]
        if let existing = live[id] {
            cache = existing
        } else {
            cache = warmRoot()?.cache ?? makeCache()
            live[id] = cache
        }
        let resident = PromptCacheIO.tokenLength(cache) ?? 0
        let start = min(resident, fullPromptTokens.count)   // clamp; a diverged prefix yields an empty delta
        return (LMInput(tokens: MLXArray(Array(fullPromptTokens[start...]))), cache)
    }

    /// Free the GPU/RAM for one conversation. Idempotent. Dropping the store's only long-lived reference
    /// to the `[KVCache]` releases the Metal buffers via ARC. Call inside `perform` (same discipline).
    package func release(_ id: UUID) { live[id] = nil }
}
```

```swift
// Sources/MLXPromptCache/PromptCacheCoordinator.swift  — replaces the `openSession` extension

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
}
```

Consumer usage (the whole point — zero ceremony, nothing non-`Sendable` crosses):

```swift
let sessions = SessionStore()          // Sendable
let coordinator = PromptCacheCoordinator(store: store)   // Sendable
let id = UUID()

try await container.perform { ctx in   // @Sendable closure captures sessions + coordinator + id
    let (delta, cache) = coordinator.advance(
        sessions, id: id,
        fullPromptTokens: full, rootTokens: root,
        model: ctx.model, parameters: params)
    for await g in try MLXLMCommon.generate(input: delta, cache: cache, parameters: params, context: ctx) { … }
}
// when the conversation ends:
try await container.perform { _ in coordinator.release(sessions, id: id) }
```

---

## 4. Step-by-step

### Step 1 — Reshape `SessionCache.swift`

Delete the entire `public final class SessionCache { … }` and replace with the `SessionStore` from §3.
(Optionally `git mv Sources/MLXPromptCache/SessionCache.swift Sources/MLXPromptCache/SessionStore.swift`
— filename is cosmetic; keep or rename, your call.) Imports stay `Foundation` / `MLX` / `MLXLMCommon`.
The `advance`/`release` primitives are **`package`** (§2.4).

Behavior preserved from the old class:
- seed from the durable disk root via `store.reuse` (hybrid-native — the reconstructed cache carries
  Mamba layers) → now the `warmRoot` closure, evaluated once on seed.
- per-turn delta computed from the cache's own offset (`PromptCacheIO.tokenLength`, hybrid-safe) → identical.
- free on release → `release(_ id:)`, now idempotent per-id.

Known-limitation carried over unchanged (out of scope to fix): a **diverged** prefix (prompt shorter than
resident, or history rewritten) just **clamps** to an empty delta; it does not re-seed from the store. The
old comment claimed "re-seeds via the store" but the code only clamped — same behavior kept. Flag as a
follow-up if divergence handling is wanted.

### Step 2 — Reshape the coordinator seam

In `PromptCacheCoordinator.swift`, **delete** the `openSession(rootTokens:model:parameters:) -> SessionCache`
extension (line ~186–197) and **add** the `advance(…)` + `release(…)` extension from §3. It must stay in
the same file as `PromptCacheCoordinator` because it references the `private let store` (Swift `private`
is file-scoped; the existing `warm`/`prefillChunked` extensions rely on the same access).

### Step 3 — Rewrite the tests (`SessionCacheTests.swift` → `SessionStoreTests.swift`)

Rename the file and suite to `SessionStoreTests`. The tests call the `package` primitives directly — the
test target is in the same package, and `@testable import MLXPromptCache` (kept for `PromptCacheIO`) also
exposes them. Keep all four load-bearing proofs; adapt inspection to the returned live cache (since
`.cache` is no longer a public property). Full target:

```swift
import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite
struct SessionStoreTests {
    let sig = Fixture.signature

    private func twoLayerModel() -> StubModel { StubModel { [KVCacheSimple(), KVCacheSimple()] as [KVCache] } }
    private func hybridModel()   -> StubModel { StubModel { [MambaCache(), KVCacheSimple()] as [KVCache] } }

    private func makeStore() throws -> (URL, PromptCacheStore, PromptCacheCoordinator) {
        let dir = Fixture.tempDir()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 30, signature: sig)
        return (dir, store, PromptCacheCoordinator(store: store))   // default blockSize 256
    }

    /// One turn: advance (seeds on the first call for `id`), prefill the delta, append `answerLen`
    /// "generated" tokens. StubModel advances each KVCacheSimple by the input length, so the held cache
    /// grows by exactly (delta + answer). Returns (deltaLen, the live cache).
    @discardableResult
    private func runTurn(_ sessions: SessionStore, id: UUID, model: StubModel,
                         fullPrompt: [Int], answerLen: Int,
                         warmRoot: () -> Reused?, makeCache: () -> [KVCache]) -> (Int, [KVCache]) {
        let (delta, cache) = sessions.advance(id: id, fullPromptTokens: fullPrompt,
                                              warmRoot: warmRoot, makeCache: makeCache)
        _ = model.callAsFunction(delta.text, cache: cache, state: nil)                    // prefill the delta
        if answerLen > 0 {
            _ = model.callAsFunction(LMInput(tokens: MLXArray(Fixture.tokens(answerLen))).text,
                                     cache: cache, state: nil)                             // append the answer
        }
        return (delta.text.tokens.shape.last ?? 0, cache)
    }

    // 1 — warm root seeds the session; turn 1's history is resident, not re-prefilled.
    @Test func seedFromWarmRootResidentNotReprefilled() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)                                                     // boundary 512
        guard case .complete = coord.warm(promptTokens: root, model: twoLayerModel(),
                                          parameters: GenerateParameters())
        else { Issue.record("warm did not complete"); return }
        let sessions = SessionStore()
        let (_, cache) = sessions.advance(id: UUID(), fullPromptTokens: root,
            warmRoot: { store.reuse(forTokens: root) },
            makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect(PromptCacheIO.tokenLength(cache) == 512)
    }

    // 2 — advance returns only the new turn's tokens.
    @Test func advanceReturnsOnlyDelta() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: twoLayerModel(), parameters: GenerateParameters())
        let sessions = SessionStore()
        let (delta, _) = sessions.advance(id: UUID(), fullPromptTokens: Fixture.tokens(512 + 20),
            warmRoot: { store.reuse(forTokens: root) },
            makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect((delta.text.tokens.shape.last ?? 0) == 20)
    }

    // 3 — THE CRUX: the cache is HELD and extended, never reloaded, and store.reuse fires exactly once.
    @Test func growsAcrossTurnsHeldNotReloaded() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: twoLayerModel(), parameters: GenerateParameters())
        let model = twoLayerModel()
        let sessions = SessionStore()
        let id = UUID()
        var reuseCalls = 0
        let warm: () -> Reused? = { reuseCalls += 1; return store.reuse(forTokens: root) }
        let make: () -> [KVCache] = { [KVCacheSimple(), KVCacheSimple()] as [KVCache] }

        var resident = 512
        var attnId: ObjectIdentifier?
        for (qLen, aLen) in [(20, 10), (15, 8), (30, 12)] {
            let (d, cache) = runTurn(sessions, id: id, model: model,
                                     fullPrompt: Fixture.tokens(resident + qLen), answerLen: aLen,
                                     warmRoot: warm, makeCache: make)
            if attnId == nil { attnId = ObjectIdentifier(cache[1] as AnyObject) }
            #expect(d == qLen)                                                            // only the new turn prefilled
            resident += qLen + aLen
            #expect(PromptCacheIO.tokenLength(cache) == resident)                         // grew by turn + answer
            #expect(ObjectIdentifier(cache[1] as AnyObject) == attnId)                    // SAME held object every turn
        }
        #expect(reuseCalls == 1)                                                          // seeded once, never re-reused
    }

    // 4 — hybrid seed: the recurrent (Mamba) layer is held live across turns, never serialised away.
    @Test func hybridSeedRidesRecurrentLive() throws {
        let (_, store, coord) = try makeStore()
        let root = Fixture.tokens(600)
        _ = coord.warm(promptTokens: root, model: hybridModel(), parameters: GenerateParameters())
        let model = hybridModel()
        let sessions = SessionStore()
        let id = UUID()
        let warm: () -> Reused? = { store.reuse(forTokens: root) }
        let make: () -> [KVCache] = { [MambaCache(), KVCacheSimple()] as [KVCache] }

        var resident = 512
        var last: [KVCache] = []
        for (qLen, aLen) in [(20, 10), (15, 8)] {
            let (d, cache) = runTurn(sessions, id: id, model: model,
                                     fullPrompt: Fixture.tokens(resident + qLen), answerLen: aLen,
                                     warmRoot: warm, makeCache: make)
            #expect(d == qLen)
            #expect(cache.first is MambaCache)                                            // recurrent layer held live
            resident += qLen + aLen
            #expect(PromptCacheIO.tokenLength(cache) == resident)
            last = cache
        }
        #expect(last.first is MambaCache)
    }

    // 5 — no warm root: the first advance is the whole prompt.
    @Test func emptySeedFirstAdvanceIsFullPrompt() {
        let sessions = SessionStore()
        let (delta, _) = sessions.advance(id: UUID(), fullPromptTokens: Fixture.tokens(30),
            warmRoot: { nil }, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect((delta.text.tokens.shape.last ?? 0) == 30)
    }

    // 6 — a prompt shorter than the resident cache clamps rather than underflowing.
    @Test func divergedPrefixClamps() {
        let model = twoLayerModel()
        let sessions = SessionStore()
        let id = UUID()
        let (_, cache) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(40),
            warmRoot: { nil }, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        _ = model.callAsFunction(LMInput(tokens: MLXArray(Fixture.tokens(40))).text, cache: cache, state: nil)
        let (delta, _) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(25),
            warmRoot: { nil }, makeCache: { [] as [KVCache] })                            // not re-seeded (already live)
        #expect((delta.text.tokens.shape.last ?? 0) == 0)
    }

    // 7 — release frees the cache (and is idempotent): a later advance for the same id RE-SEEDS.
    @Test func releaseFreesCache() {
        let sessions = SessionStore()
        let id = UUID()
        let (_, cache) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(30),
            warmRoot: { nil }, makeCache: { [KVCacheSimple(), KVCacheSimple()] as [KVCache] })
        #expect(!cache.isEmpty)
        sessions.release(id)
        var reseeded = false
        let (_, cache2) = sessions.advance(id: id, fullPromptTokens: Fixture.tokens(10),
            warmRoot: { nil }, makeCache: { reseeded = true; return [KVCacheSimple()] as [KVCache] })
        #expect(reseeded)                                                                 // entry was dropped → makeCache ran
        #expect(PromptCacheIO.tokenLength(cache2) == 0)                                    // brand-new empty cache
        sessions.release(id); sessions.release(id)                                        // idempotent — no crash
    }
}
```

Mapping to the brief's "load-bearing" requirements:
- *held & extended across turns (same object, `store.reuse` once after seed)* → test 3: `ObjectIdentifier`
  stable every turn **and** `reuseCalls == 1`.
- *per-turn delta is only the new tokens* → tests 2, 3, 4 (`d == qLen`).
- *a hybrid seed keeps its recurrent layer live* → test 4 (`cache.first is MambaCache` after turns).
- *release frees it* → test 7 (re-seed proves the entry was dropped; idempotent double-release).

### Step 4 — Update the conversation gate in `main.swift`

Replace the `SessionCache`-based `held` block (lines ~266–286) with the `SessionStore` form. `ConvOut`,
`generateTokens`, `makePromptCache`, and the surrounding warm-the-root code are unchanged.

```swift
        let sessions = SessionStore()
        let convId = UUID()

        // Both turns inside ONE perform. Nothing non-Sendable crosses: `sessions` + `coordinator` + the
        // UUID are all Sendable; the live [KVCache] is created, grown, and freed inside the store.
        let held: ConvOut = try await mc.perform { context in
            // Turn 1 — whole document resident (seeded from the warm root), only the question prefills.
            let (d1in, cache1) = coordinator.advance(sessions, id: convId,
                fullPromptTokens: document + qTokens, rootTokens: document,
                model: context.model, parameters: convParams)
            let d1 = d1in.text.tokens.shape.last ?? 0
            let t1 = Date(); var a1: [Int] = []; var ttft1 = 0.0
            for await g in try generateTokens(input: d1in, cache: cache1, parameters: convParams, context: context) {
                if case .token(let tok) = g { if a1.isEmpty { ttft1 = Date().timeIntervalSince(t1) * 1000 }; a1.append(tok) }
            }
            // Turn 2 — same id ⇒ the SAME live cache (document+Q1+A1 resident); only Q2 prefills.
            let (d2in, cache2) = coordinator.advance(sessions, id: convId,
                fullPromptTokens: document + qTokens + a1 + qbTokens, rootTokens: document,
                model: context.model, parameters: convParams)
            let d2 = d2in.text.tokens.shape.last ?? 0
            let t2 = Date(); var a2: [Int] = []; var ttft2 = 0.0
            for await g in try generateTokens(input: d2in, cache: cache2, parameters: convParams, context: context) {
                if case .token(let tok) = g { if a2.isEmpty { ttft2 = Date().timeIntervalSince(t2) * 1000 }; a2.append(tok) }
            }
            coordinator.release(sessions, id: convId)
            return ConvOut(d1: d1, d2: d2, a1: a1, a2Held: a2, ttft1: ttft1, ttft2: ttft2)
        }
```

The cold-comparison block below it and the `deltaOnly` / `heldEqCold` asserts are unchanged. Behavior is
identical: turn 1 delta `== qTokens.count`, turn 2 delta `== qbTokens.count`, held answer token-identical
to the cold full-prefill.

### Step 5 — Swift 6 strict-concurrency pass (whole module)

The audit in §0 and the review in §7 found the rest of the module clean under `.v6`. After Steps 1–4,
confirm:

- [ ] `SessionStore` is the only `@unchecked Sendable` in the module, and its invariant comment is present.
- [ ] The raw `advance`/`release` are `package`; the only `public` doors are on the coordinator.
- [ ] No `[KVCache]` (nor `SessionStore`'s internals) appears as **stored** consumer state or as a
      parameter/return that crosses into a stored position — only the in-place `advance` return.
- [ ] `grep -rn "perform(nonSendable"` returns nothing in the consumer path (the escape hatch is gone).
- [ ] `PreparedCache` / `Reused` doc comments still accurately say "model thread only."
- [ ] Build emits **zero** concurrency warnings (not just zero errors) under `.v6`.

No other files need edits for the reshape; the §8 cleanup is separate.

---

## 5. Verification

```bash
swift build 2>&1 | tee /tmp/mlxpc-build.txt          # expect: clean, no warnings
swift test  --filter SessionStoreTests               # expect: 7/7 green
swift test                                           # expect: whole suite green (incl. §8 store tests)
```

Device/manual gate (needs the real hybrid model + Metal; not part of `swift test`): run the
`MLXPromptCacheScratch` executable and confirm the conversation-gate line prints:

```
conversation gate [...]: d1=<qTok>(Q <qTok>) d2=<qbTok>(Q2 <qbTok>) · TTFT turn1 … → turn2 … · delta-only ✅ · held==cold ✅
```

i.e. turn 2 prefilled only the new question, and the held-cache answer is byte-identical to a cold
full-prefill on the real hybrid model.

---

## 6. Risks & open decisions for the human

1. **`@unchecked Sendable` trust boundary.** Soundness rests on "all access inside `perform`." Baked into
   §3: the raw `advance`/`release` are `package`, and the only public door (`coordinator.advance`) requires
   `model` (obtainable only inside `perform`). A debug-only executor assertion is possible but there's no
   clean "am I inside this container's `perform`?" check in mlx-swift, so the model-requiring door + the
   documented invariant is the pragmatic guard — and it matches how the module already trusts callers of
   `Reused`/`PreparedCache`, and how mlx-swift trusts callers of `ModelContext`.
2. **`[KVCache]` in the `advance` return** — accepted per §2.3, and the compiler still forbids the
   dangerous misuse (storing it). If you want the stricter "no `[KVCache]` anywhere public," switch to the
   store-drives-generation design (noted, not recommended).
3. **Diverged-prefix handling** — still clamp-only (§Step 1). Not in scope; flag as a follow-up if
   conversation editing/branching is a real use case.
4. **File rename** `SessionCache.swift → SessionStore.swift` — cosmetic; do it or don't.
5. **Naming** — `SessionStore` chosen; alternatives `ConversationCaches`, `LiveCacheStore` if a different
   noun reads better in the consumer.
6. **Snapshot byte-extraction must stay inside `perform` (oMLX #1106).** oMLX hit real
   SIGABRT/kernel-panic bugs (#1106, #300, #888) when an *async* store-cache worker read KV bytes via
   `memoryview` while the inference thread issued a buffer-pool reclaim (`mx.clear_cache`). Its rules:
   byte-extraction *"Must be called on the inference thread"* (`boundary_snapshot_store.py:640`), and any
   off-thread buffer access is wrapped in `_mx_buffer_access_lock` (`scheduler.py:165`). **We are safe
   today** — `PromptCacheIO.save` copies (`liveCache.map { $0.copy() }`) and serialises synchronously
   inside `perform`; no async worker, no lock needed. **Keep it that way.** If snapshotting is ever moved
   to a background `Task`/actor to unblock generation, extract tensor→bytes inside `perform` first (hand
   plain `Data` to the writer), or you port oMLX's crash, not just its cache.

---

## 7. Concurrency review (verified against mlx-swift-lm, `.v6`)

Folded in from a Swift-concurrency pass over the whole module. **0 BLOCKERs, 3 WARNs, 4 NOTEs.** The
module is clean under `.v6`; the reshape is sound and idiomatic.

### Verdict on the reshape — adopt

`@unchecked Sendable SessionStore` is **sound and idiomatic**, confirmed against the dependency's own
source:

- `ModelContainer` is `final class: Sendable` (`ModelContainer.swift:32`), doc-commented *"guarantees
  single threaded access,"* implemented via `private let context: SerialAccessContainer<ModelContext>`.
- **`SerialAccessContainer<T>: @unchecked Sendable`** (`SerialAccessContainer.swift:44`), backed by a
  `private actor AsyncMutex` (:8). This is the exact pattern the reshape uses — a `final class @unchecked
  Sendable` wrapping non-`Sendable` state whose access is serialised.
- The library's own `concurrency.md` documents **"Why Not Actor?"** — *actors release isolation at `await`
  points*; a serial container holds exclusivity for the whole async op. That is a second, independent
  reason (on top of §2.6) that a plain `actor` is the wrong tool here.
- `perform`'s action is `@Sendable … -> sending R` with `R: Sendable` (`ModelContainer.swift:90`). Two
  consequences, both confirming §1–§2.3: captured state must be `Sendable` (⇒ `SessionStore: Sendable`
  deletes the `perform(nonSendable:)` escape hatch), and the return must be `Sendable` (⇒ `[KVCache]` is
  used in place, never returned out of `perform`).
- `concurrency.md`'s "Keep Arrays Within Isolation" ("all array operations in same perform block") is
  exactly the `SessionStore` usage pattern; `ChatSession` is documented "NOT thread-safe (single task
  only)" — the same single-owner posture these cache-holding types take.

### WARNs

1. **Narrow the `@unchecked` blast radius — DONE in §3.** The raw `advance`/`release` are `package`; the
   only public door is `coordinator.advance` (requires `model`) / `coordinator.release`. This is the
   smallest surface that still lets the coordinator wrap them and tests exercise them.
2. **Invariant wording: "serialised via `perform`", not "the model thread" — DONE in §2.1/§2.3/§2.6 +
   the §3 doc comment.** mlx-swift's guarantee is *serialised exclusive access* (`SerialAccessContainer`/
   `AsyncMutex`), not OS-thread pinning (an `AsyncMutex` continuation can resume on any pool thread). For
   data-race/`Sendable` purposes serialisation is exactly what's needed. oMLX's thread-pinning is its
   *mechanism* for the same non-concurrency; the doc now says so.
3. **Disk IO under the catalog `Mutex` (pre-existing, not the reshape).** Addressed in §8.

### NOTEs

1. `@unchecked Sendable` on `SessionStore` does **not** make `[KVCache]` `Sendable` — only the internal
   map. The consumer still gets a fresh `Sendable` error if they try to stash the returned live cache into
   actor state (§2.3). The unsafe surface is contained.
2. `warmRoot`/`makeCache` are correctly **non-escaping, non-`@Sendable`** (called synchronously at seed).
   Keep them so — making them `@Sendable`/escaping would invite storing them off-`perform`.
3. The module correctly does **not** opt into main-actor-default isolation (approachable concurrency) —
   right call for a reusable library; it stays nonisolated so consumers on any actor use it.
4. `HotCodec.reconstruct` transfers a raw buffer to `MLXArray(finalizer:)`. Not a concurrency issue
   (single owner, inside `perform`), but it's the module's one raw-memory seam — a one-line "don't touch
   `buf` after ownership transfers" comment would help.

### Positives (patterns to keep)

- The store already embodies the distinction the reshape formalises: value-type bookkeeping
  (`Catalog`/`HotCache`) → `Mutex` (any-thread mutual exclusion of *bytes/metadata*, never live caches);
  live GPU caches (`[KVCache]`) → `perform`-serialised, never in a `Mutex`. Coherent.
- "Sendable-out-of-the-lock" discipline in `reuse` — takes `Hit`/`Entry`/bytes out, reconstructs
  `[KVCache]` *outside* the lock (`PromptCacheStore.swift:54`). Textbook; never holds a lock across GPU
  work.
- Deliberately non-`Sendable` `Reused`/`PreparedCache` with "model thread only" docs — correct; resist
  slapping `Sendable` on them.

---

## 8. Separate pre-existing cleanup — disk IO under the catalog `Mutex`

**This is not part of the reshape.** It's a standing item WARN-3 surfaced; folded in because the module
should be clean.

### Observation

`PromptCacheStore.record` and `reuse`'s self-heal path call `writeIndex` (JSON-encode + `data.write(to:)`)
**inside** `catalog.withLock`:

```swift
// record(), ~PromptCacheStore.swift:157
let toDelete: [String] = catalog.withLock { cat in
    let deleted = cat.commit(plan, byteSize: bytes, budgetBytes: budgetBytes)
    Self.writeIndex(cat, to: idxURL)          // ← disk IO while holding the lock
    return deleted
}
// reuse() self-heal, ~PromptCacheStore.swift:67 — same shape (evict + writeIndex under the lock)
```

Holding the `Mutex` across disk IO blocks any concurrent `reuse`/`peek`/`record` for the write's duration.

### Actual severity: LOW — but worth cleaning

In the module's intended architecture every store call happens inside `ModelContainer.perform`, which
serialises everything — so there is **no concurrent waiter** on the catalog lock and the lock-across-IO
costs nothing today. It bites only if the store is driven from genuinely concurrent callers (e.g. a future
disk-only warmer off the `perform` path). And even then the failure mode is benign: `index.json` is a
*durable backup*; the in-memory catalog (behind the `Mutex`) is authoritative during a session, and
`loadOrReset` / `reuse` self-heal a stale or vanished index on next use — *"never a wrong answer, at worst
a cold prefill,"* consistent with the module's existing `peek` honesty note.

### The fix (correct, not the naive hoist)

Move the encode+write **out** of the catalog lock, and serialise the writes through a dedicated lock with
a **monotonic write-generation guard** so a stale snapshot can never overwrite a newer one on disk:

```swift
// New stored property on PromptCacheStore (Synchronization already imported):
private let indexIO = Mutex<Int>(0)          // highest catalog generation persisted to disk

// Catalog gains a monotonic `var writeGeneration = 0`, bumped in EVERY on-disk-relevant mutation
// (commit, evict/drop). (Codable field; greenfield store wipes on signature change, so no migration.)

// record(): capture snapshot + generation UNDER the lock, persist OUTSIDE it.
let idxURL = directory.appendingPathComponent("index.json")
let (toDelete, snapshot, gen): ([String], Catalog, Int) = catalog.withLock { cat in
    let deleted = cat.commit(plan, byteSize: bytes, budgetBytes: budgetBytes)
    return (deleted, cat, cat.writeGeneration)   // Catalog is a value type → snapshot copies out (COW)
}
indexIO.withLock { lastGen in
    guard gen > lastGen else { return }          // a newer snapshot already hit disk — skip
    Self.writeIndex(snapshot, to: idxURL)
    lastGen = gen
}
```

The `reuse` self-heal path gets the same treatment (capture `(snapshot, gen)` after `evict`, persist via
`indexIO`). Because commits/evicts serialise on the catalog lock, the captured `writeGeneration`s strictly
increase, and a higher-generation snapshot always subsumes all lower ones — so "skip if not newer" is
correct last-writer-wins. `reuse`/`peek`/`record`-catalog-mutation no longer block on the disk write; only
index writes serialise among themselves (necessary for ordering, and infrequent).

> ⚠️ **Do NOT simply move `writeIndex` outside the lock without the generation guard.** Under concurrent
> `record`s the writes can reorder and a stale `index.json` can clobber a newer one. The guard (or keeping
> the write under the lock) is what makes it correct. Given the self-healing backup semantics above, the
> unguarded hoist is *defensible* but not deterministic — prefer the guard if you touch this at all.

### Verification for §8

- Existing `PromptCacheStoreTests` / `PromptCacheStoreHybridTests` stay green.
- Add a small test: fire N concurrent `record`s (via `DispatchQueue.concurrentPerform` or a `TaskGroup`)
  against one store, then re-open it and assert the reloaded catalog matches the final in-memory state
  (newest generation on disk). Proves the guard, and that no write is lost or reordered.
