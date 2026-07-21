# Story — Chat resume (Arm A) + day-chunked conversation log (IMPLEMENTATION)

**Status:** Implementation-ready for Arm A; the two analysis-side custody edges (§4) reconcile with R1/R2.
**Supersedes:** the DRAFT of the same name (which framed the store as a straight "lift"). It is **not** a
lift-and-shift — our concurrency model and strict package split force real adaptation, so every adapted
piece below is shown **full Tanuki-before → CyberBench-after** with the changes justified.
**Origin:** `CyberBench-Story-Chat-Reload-STUB.md`, `CyberBench-Chat-Cache-and-Log-Orientation.md`.
**Grounded in:** disk reads 2026-07-21 — CyberBench: `SessionStore`/`WarmStore` (MLXPromptCache `e0086a7`),
`FileSystemDerivedArtifactStore`, `FileSystemInferenceRecordStore`, `InferenceRecorder`, `Conversation`,
`ConversationViewModel`/`Reducer`/`State`, `Evidence`, `WindowServices`, `InferenceRecordStoreProtocol`.
Tanuki: `CalendarDate`, `DailyLog`, `Message`, `ChatShardRef`, `ChatMeta`, `ChatHistoryRepositoryProtocol`,
`YAMLChatHistoryRepository`. Provenance specs: `…Provenance-and-Derived-Data-Treatment.md`,
`ARCH/CyberBench-CaseProvenance-BOM-Design.md`.

---

## §0 — Scope

- **IN:** Arm A (stop-ending-on-switch; hold N conversations live; hot-swap on file select; resume by
  reassembly; app-owned `SessionStore` memory budget); the day-chunked conversation log, in-vault, mirroring
  the evidence path; the conversation `closed` status and Fresh Chat.
- **OUT (seam only):** Arm B fast-slot recall — the sole requirement is `index()` exists. Not designed here.
- **OUT (owned by R1/R2, §4):** the finding↔conversation custody edge and the conversation hash's entry into
  the BOM graph. This story freezes the hash at the right points; the provenance epic consumes it.

---

## §0.5 — Reviews applied (2026-07-21)

Reviewed with `/hex-mvvm-arch-review`, `/swift-concurrency-expert`, `/swiftui-performance-audit`. Findings
folded into §3 below:
- **BLOCKER (hex, ADR-015):** the VM held a Domain port to `close()`. Fixed — `CloseConversationUseCase`
  (§3.5); the VM injects use cases only.
- **BLOCKER (concurrency):** eviction was wired off-`perform` via `SessionStore`'s `package` methods. Fixed —
  a public `PromptCacheCoordinator.evictSessions` door called inside `MLXConversationEngine.perform`, budget
  injected as a live-resolve provider (adr-cb-008) (§3.1/§3.7).
- **WARN (hex + concurrency):** the resume `Task` wasn't cancellable (stale-dispatch race). Fixed — a stored
  `resumeTask`, cancel-previous + drop-if-cancelled (§3.7).
- **swiftui-performance:** clean — transcript already `LazyVStack` + value-driven rows; keep it that way.
- Estimate: ~2–2.5d (library door + store + use cases + VM + wiring + tests).

## §1 — Design invariants (the spine every section obeys)

1. **Reassembly is already in the engine.** `SessionStore.advance` seeds a fresh conversation from the warm
   root (banked evidence prefix — a catalog-probe) and returns only `fullPromptTokens[resident...]` as the
   delta (`SessionStore.swift:35-43`). Resume = load the conversation's turns from the day-chunked log →
   `RunConversationTurnUseCase` assembles the full messages → `advance` seeds evidence-from-warm and
   re-prefills only the turns. **No new cache-reassembly code.** Arm A is: don't release; look the
   conversation up; bound memory.
2. **The log mirrors the evidence path** — exactly as `FileSystemDerivedArtifactStore.artifactRoot` maps
   `Evidence/Audio/x.m4a` → `Derived/Audio/x.m4a/`. Conversations →
   `Conversations/Audio/x.m4a/<conversationId>/<YYYY-MM-DD>.yaml` (+ `meta.yaml`). Consequence: the store is
   **evidence-scoped** (`for evidence: Evidence, vaultRoot: URL` on every method — Derived's exact shape),
   app-wide and stateless. v1 is single-file (`AskPane`: "v1: exactly one Evidence"); multi-file corpus is a
   flagged future (§6).
3. **Concurrency: the module default is `@MainActor` + `NonisolatedNonsendingByDefault`.** File IO must run
   off the main actor, so the store is a `nonisolated public struct` with `@concurrent` methods (the
   `FileSystemDerivedArtifactStore` pattern, not Tanuki's plain-`async` `final class`). Domain value types are
   `nonisolated public struct`; the Domain stays clock-free (no `Date()` inside a Domain type).
4. **Strict package split.** Port + entities in **Domain**; adapter in **Infrastructure** (imports Domain +
   Yams + CryptoKit); use case in **Application**; VM in **Presentation**. Tanuki's single-module
   `TanukiKit` collapses all of these — every adapted type below gains explicit package placement + imports.
5. **Hashing at set points, never rolling** (specs, §4): a day-chunk is hashed when it finalises;
   `ConversationMeta.contentHash` is nil while open and frozen at close; a finding pins the hash it saw.
6. **A conversation never spans models** (named 2026-07-21): a deep-slot model change closes+seals every
   held conversation (§3.7), so one conversation is single-model. This is currently only a UI behaviour;
   named here as a domain invariant. NOT load-bearing for the L2 attestation — the epic's R4 puts
   `methodRef` per-basis-entry, so a multi-model basis would validate anyway (each turn's call carries its
   own model/method). It stands for record *coherence* (a conversation reads as one interrogation session),
   not for BOM correctness. Relaxing it for UX is safe against the standards; keep it for legibility.

---

## §2 — Adapted from Tanuki (full before → after; it is NOT a straight copy)

### 2.1 `CalendarDate` — Domain value type

**Tanuki BEFORE** (`Tanuki/.../Domain/Chat/Entities/CalendarDate.swift`):

```swift
public struct CalendarDate: Codable, Sendable, Equatable, Hashable, Comparable {
    public let year: Int; public let month: Int; public let day: Int

    public init(from date: Date = Date()) {
        let calendar = Calendar.current
        let shifted = date.addingTimeInterval(
            -Double(Constants.ChatHistory.dayBoundaryHour) * 3_600)   // ← Tanuki-only constant
        self.year = calendar.component(.year, from: shifted)
        self.month = calendar.component(.month, from: shifted)
        self.day = calendar.component(.day, from: shifted)
    }
    public init(year: Int, month: Int, day: Int) { … }
    public var filename: String { String(format: "%04d-%02d-%02d.yaml", year, month, day) }
    public var asDate: Date { Calendar.current.date(from: DateComponents(year:month:day:)) ?? Date() }
    public static func < (lhs, rhs) -> Bool { … }
}
```

**CyberBench AFTER** (`Packages/Domain/Sources/Domain/Inference/Entities/CalendarDate.swift`):

```swift
import Foundation

/// A calendar-day identity — the key for the day-chunked conversation log. Foundation-only, clock-free:
/// the CURRENT day is derived in Infrastructure and passed in (Domain never reads the clock).
nonisolated public struct CalendarDate: Codable, Sendable, Equatable, Hashable, Comparable {
    public let year: Int
    public let month: Int
    public let day: Int

    public init(year: Int, month: Int, day: Int) {
        self.year = year; self.month = month; self.day = day
    }

    /// `YYYY-MM-DD.yaml` — the day-chunk filename.
    public var filename: String { String(format: "%04d-%02d-%02d.yaml", year, month, day) }

    public static func < (lhs: CalendarDate, rhs: CalendarDate) -> Bool {
        (lhs.year, lhs.month, lhs.day) < (rhs.year, rhs.month, rhs.day)
    }
}
```

**Changed & why:** (1) `nonisolated` — required for a Domain value type under the MainActor module default.
(2) **Dropped `init(from: Date = Date())`** — a Domain type must not read the clock; the Infrastructure store
computes today via `Calendar.current` and calls `init(year:month:day:)`. (3) **Dropped the
`Constants.ChatHistory.dayBoundaryHour` shift** — boundary is system midnight; the late-night-rollover hour is
a personal-assistant nicety with no meaning for evidence chat. (4) Dropped `asDate` (unused without recall;
tuple-`<` replaces the hand-rolled compare).

### 2.2 `ConversationDayChunk` ← `DailyLog`

**Tanuki BEFORE** — the whole file, verbatim (`Tanuki/.../Domain/Chat/Entities/DailyLog.swift`):

```swift
import Foundation

/// One day's worth of one chat's messages — a *day-shard* of a chat thread,
/// persisted by `YAMLChatHistoryRepository` at
/// `ChatHistory/chats/<chatId>/<YYYY-MM-DD>.yaml`. Past shards are immutable;
/// only today's shard is rewritten on append.
///
/// `totalTokens` is no longer written by the adapter (it stays `nil` on
/// disk); `index()` computes each finalised shard's total on read. The field
/// is retained so `DailyLog`'s `Codable` shape is byte-stable — it's a
/// candidate for removal in a later cleanup, left for now to keep the shard
/// reuse truly drop-in.
public struct DailyLog: Codable, Sendable, Equatable {
    public let date: CalendarDate
    public var messages: [Message]

    /// Sum of `TokenCounter.estimateTokens(for:)` over `messages`,
    /// computed once at the today→yesterday rollover and persisted.
    /// `nil` means either (a) this is today's in-flight log, or (b) the
    /// log predates the precompute feature and awaits startup-backfill.
    /// Historical-search batch planners filter on `totalTokens != nil`.
    public var totalTokens: Int?

    public init(
        date: CalendarDate,
        messages: [Message] = [],
        totalTokens: Int? = nil
    ) {
        self.date = date
        self.messages = messages
        self.totalTokens = totalTokens
    }

    /// Append a message to this day's log. The adapter calls this after
    /// loading the existing day file; the mutated `DailyLog` is then
    /// re-serialised back to disk.
    public mutating func append(_ message: Message) {
        messages.append(message)
    }
}
```

(Tanuki's `Message` = `{ id: UUID, role: MessageRole, content: String, timestamp: Date }`; CyberBench does
NOT adapt it — the chunk holds the existing `ConversationTurn` (question+answer) instead, so `[Message]`
becomes `[ConversationTurn]`.)

**CyberBench AFTER** (`Packages/Domain/Sources/Domain/Inference/Entities/ConversationDayChunk.swift`):

```swift
import Foundation

/// One day-chunk of a conversation — the append-only shard the log persists. Immutable once it is no
/// longer today's chunk. Holds TURNS (question+answer), not single messages: CyberBench's unit is the turn.
nonisolated public struct ConversationDayChunk: Codable, Sendable, Equatable {
    public let date: CalendarDate
    public var turns: [ConversationTurn]
    public var totalTokens: Int?     // cheap estimate, computed on read for past chunks; nil for today

    public init(date: CalendarDate, turns: [ConversationTurn] = [], totalTokens: Int? = nil) {
        self.date = date; self.turns = turns; self.totalTokens = totalTokens
    }
}
```

**Changed & why:** `nonisolated`; `[Message]` → `[ConversationTurn]` (the existing CyberBench entity —
`Conversation.swift:3`; the append unit is a turn, not a message).

### 2.3 `ConversationChunkRef` ← `ChatShardRef`

**Tanuki BEFORE** — the whole file, verbatim (`Tanuki/.../Domain/Chat/Entities/ChatShardRef.swift`):

```swift
import Foundation

/// A pointer to one `(chat, day)` shard plus its token cost — the element
/// the pool-wide historical search plans batches over (Story 3). Produced by
/// `ChatHistoryRepositoryProtocol.index()`. `totalTokens` is computed on read
/// from the shard's messages for finalised (past) shards, and `nil` for the
/// in-flight (today's) shard — which preserves the deployed "search excludes
/// today" behaviour.
public struct ChatShardRef: Sendable, Equatable {
    public let chatId: UUID
    public let date: CalendarDate
    public let totalTokens: Int?

    public init(chatId: UUID, date: CalendarDate, totalTokens: Int?) {
        self.chatId = chatId
        self.date = date
        self.totalTokens = totalTokens
    }
}
```

**CyberBench AFTER** (`…/Entities/ConversationChunkRef.swift`):

```swift
import Foundation

/// A pointer to one (conversation, day) chunk + its cheap token estimate — the Arm-B recall seam element.
nonisolated public struct ConversationChunkRef: Sendable, Equatable {
    public let conversationId: UUID
    public let date: CalendarDate
    public let totalTokens: Int?
    public init(conversationId: UUID, date: CalendarDate, totalTokens: Int?) {
        self.conversationId = conversationId; self.date = date; self.totalTokens = totalTokens
    }
}
```

**Changed & why:** `nonisolated`; `chatId` → `conversationId` (our vocabulary).

### 2.4 `ConversationMeta` ← `ChatMeta`

**Tanuki BEFORE** — the whole file, verbatim (`Tanuki/.../Domain/Chat/Entities/ChatMeta.swift`):

```swift
import Foundation

/// Metadata for one chat in the registry. Persisted as
/// `ChatHistory/chats/<id>/meta.yaml`; the chat's actual messages live in
/// sibling `<date>.yaml` day-shards (`DailyLog`). This is also the picker's
/// data source — `ChatHistoryRepositoryProtocol.list()` returns `[ChatMeta]`
/// ordered newest `updatedAt` first.
///
/// `title` is a placeholder at create time, set from the first user message
/// in v1 (Story 4) and from a fast-slot LLM pass in a follow-up (Epic
/// §"Overview"). `updatedAt` bumps on every `append` so the picker can order
/// by recency without reading any shard.
public struct ChatMeta: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public var title: String
    public let createdAt: Date
    public var updatedAt: Date

    public init(id: UUID, title: String, createdAt: Date, updatedAt: Date) {
        self.id = id
        self.title = title
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}
```

**CyberBench AFTER** (`…/Entities/ConversationMeta.swift`):

```swift
import Foundation

/// The per-conversation registry record (`…/<conversationId>/meta.yaml`). Case material.
/// `contentHash` is nil while the chat is OPEN (mutable) and frozen at CLOSE — the seal-aligned
/// "integrity at rest" point (§4). `closedAt` is the whole of the status: "this chat is over."
nonisolated public struct ConversationMeta: Codable, Sendable, Equatable, Identifiable {
    public let id: UUID
    public let caseId: UUID
    public let inputRefs: [UUID]        // the evidence selection (v1: one) — the resume lookup key
    public let startedAt: Date
    public var updatedAt: Date          // bumped on append — orders "which conversation is active"
    public var closedAt: Date?          // nil = open/live; set once at model-change / Fresh Chat / close
    public var closedReason: String?    // "model changed" | "new chat" | "closed"
    public var contentHash: String?     // nil while open; SHA-256 rollup frozen at close (§4)

    public init(id: UUID, caseId: UUID, inputRefs: [UUID], startedAt: Date,
                updatedAt: Date, closedAt: Date? = nil, closedReason: String? = nil,
                contentHash: String? = nil) {
        self.id = id; self.caseId = caseId; self.inputRefs = inputRefs; self.startedAt = startedAt
        self.updatedAt = updatedAt; self.closedAt = closedAt; self.closedReason = closedReason
        self.contentHash = contentHash
    }
}
```

**Changed & why:** `nonisolated`; **dropped `title`** (no picker in v1 — ruling 5); **added** `caseId` +
`inputRefs` (case material + the resume key), `closedAt`/`closedReason` (the status), `contentHash` (frozen
at close). `createdAt`→`startedAt` to match `Conversation.startedAt`.

### 2.5 `ConversationLogStoreProtocol` ← `ChatHistoryRepositoryProtocol`

**Tanuki BEFORE** — the whole file, verbatim (`Tanuki/.../Domain/Chat/Ports/ChatHistoryRepositoryProtocol.swift`):

```swift
import Foundation

/// Domain port for the chat registry. One conformer (`YAMLChatHistoryRepository`)
/// over one pool of chat folders shared by every surface; surfaces differ only
/// in which `chatId` they have active, never in storage. Implementations live
/// in Infrastructure.
public protocol ChatHistoryRepositoryProtocol: Sendable {
    // MARK: Registry / picker

    /// Every chat's `ChatMeta`, newest `updatedAt` first. Reads one small
    /// `meta.yaml` per chat — never touches a shard.
    func list() async throws -> [ChatMeta]

    /// Create an empty chat (writes `meta.yaml`, no shards yet) and return
    /// its metadata.
    func create(title: String) async throws -> ChatMeta

    /// Set a chat's title; bumps `updatedAt`. No-op if the chat is absent.
    func rename(_ id: UUID, to title: String) async throws

    /// Remove a chat's whole folder (meta + every shard). No-op if absent.
    func delete(_ id: UUID) async throws

    /// Remove the entire pool. Content removal wipes the chat directory by
    /// path (DataResetService); this method is the in-app clear-all seam,
    /// currently exercised only by tests until the clear-all UI returns.
    func deleteAll() async throws

    // MARK: Active chat

    /// Append a message to the chat's *today* shard (creating the shard and,
    /// defensively, the chat folder + meta if missing); bumps `meta.updatedAt`.
    /// The per-turn hot path — rewrites only today's bounded shard.
    func append(_ message: Message, to id: UUID) async throws

    /// The chat's day-shard dates, newest first. Drives the respond path's
    /// continuity walk (Story 2).
    func shardDates(_ id: UUID) async throws -> [CalendarDate]

    /// Load one `(chat, day)` shard, or `nil` if absent.
    func loadShard(_ id: UUID, date: CalendarDate) async throws -> DailyLog?

    // MARK: Recall (whole-pool historical search)

    /// Every `(chat, day)` shard across the pool, with each finalised shard's
    /// token total computed on read (today's shard reports `nil`). Drives the
    /// historical-search batch planner (Story 3).
    ///
    /// **Deliberate replacement for the date-keyed `availableDates()` +
    /// `dateIndex()`.** No `loadAll()` — search walks the pool in batches via
    /// `index()` then `loadShard(_:date:)` per shard. If the index scan ever
    /// profiles hot, swap the on-read computation for a header-only parse or a
    /// sidecar index file — the port contract is unchanged.
    func index() async throws -> [ChatShardRef]
}
```

**CyberBench AFTER** (`Packages/Domain/Sources/Domain/Inference/Ports/ConversationLogStoreProtocol.swift`):

```swift
import Foundation

/// The day-chunked conversation log — vault case material, laid out mirroring the evidence path
/// (`Conversations/<evidence-inner-path>/<conversationId>/…`, like DerivedArtifactStore mirrors to
/// `Derived/`). Evidence-scoped + `vaultRoot` per call (the FileSystemDerivedArtifactStore shape), so the
/// conformer is app-wide + stateless. Append-only: past day-chunks are immutable; only today's is rewritten.
public protocol ConversationLogStoreProtocol: Sendable {
    /// Create the conversation folder + meta (no chunks yet). Idempotent.
    func create(_ meta: ConversationMeta, for evidence: Evidence, vaultRoot: URL) async throws
    /// Append one turn to the conversation's TODAY chunk; bumps `meta.updatedAt`. The per-turn hot path —
    /// rewrites only today's bounded chunk. `today` is passed in (Domain is clock-free).
    func append(_ turn: ConversationTurn, to conversationId: UUID, for evidence: Evidence,
                today: CalendarDate, vaultRoot: URL) async throws
    /// Close the conversation: stamp `closedAt`/`closedReason` and freeze `contentHash` (§4). Idempotent.
    func close(_ conversationId: UUID, for evidence: Evidence, reason: String,
               at when: Date, vaultRoot: URL) async throws
    /// Reassemble the full Conversation (meta + all chunk turns, oldest-first) — the hot resume path.
    func loadConversation(_ conversationId: UUID, for evidence: Evidence, vaultRoot: URL)
        async throws -> Conversation?
    /// Reassemble by id ALONE (no evidence) — the RARE validation path (the promote/finding gate has a
    /// `conversationRef` but not the file). Walks the mirror tree; not for the hot path.
    func loadConversation(_ conversationId: UUID, caseVaultRoot: URL) async throws -> Conversation?
    /// Every conversation meta for THIS file, newest `updatedAt` first — the resume/hot-swap lookup.
    func listMeta(for evidence: Evidence, vaultRoot: URL) async throws -> [ConversationMeta]
    /// Chunk refs for this file, with cheap token estimates — the Arm-B recall seam. (Not consumed in v1.)
    func index(for evidence: Evidence, vaultRoot: URL) async throws -> [ConversationChunkRef]
}
```

**Changed & why:** keyed by **(evidence, conversationId)** not a global `chatId`; **`vaultRoot` per call**
(per-case vault, decision 12, threaded like Derived); **added `close`** (freeze point — no Tanuki analog);
dropped `rename`/`deleteAll`/global `list` (no picker, nothing deleted — append-only custody). `: Sendable`
required.

### 2.6 `FileSystemConversationLogStore` ← `YAMLChatHistoryRepository` (the load-bearing adaptation)

**Tanuki BEFORE** — the whole file, verbatim (`Tanuki/.../Infrastructure/Persistence/Adapters/YAMLChatHistoryRepository.swift`):

```swift
import Foundation
import Yams

public final class YAMLChatHistoryRepository: ChatHistoryRepositoryProtocol, Sendable {
    private let chatsDirectory: URL // pool root: …/ChatHistory/chats
    private let tokenCounter: TokenCounter
    private let logger: Logger

    public init(chatsDirectory: URL, tokenCounter: TokenCounter, logger: Logger) {
        self.chatsDirectory = chatsDirectory
        self.tokenCounter = tokenCounter
        self.logger = logger
    }

    // MARK: - Registry

    public func list() async throws -> [ChatMeta] {
        let dirs = chatFolders()
        let metas = dirs.compactMap { loadMeta(at: $0) }
        return metas.sorted { $0.updatedAt > $1.updatedAt }
    }

    public func create(title: String) async throws -> ChatMeta {
        let now = Date()
        let meta = ChatMeta(id: UUID(), title: title, createdAt: now, updatedAt: now)
        try FileManager.default.createDirectory(at: chatFolder(meta.id), withIntermediateDirectories: true)
        try writeMeta(meta)
        logger.info("Created chat \(meta.id)")
        return meta
    }

    public func rename(_ id: UUID, to title: String) async throws {
        guard var meta = loadMeta(at: chatFolder(id)) else { return }
        meta.title = title
        meta.updatedAt = Date()
        try writeMeta(meta)
    }

    public func delete(_ id: UUID) async throws {
        let folder = chatFolder(id)
        guard FileManager.default.fileExists(atPath: folder.path(percentEncoded: false)) else { return }
        try FileManager.default.removeItem(at: folder)
        logger.info("Deleted chat \(id)")
    }

    public func deleteAll() async throws {
        guard FileManager.default.fileExists(atPath: chatsDirectory.path(percentEncoded: false)) else { return }
        try FileManager.default.removeItem(at: chatsDirectory)
        logger.info("Deleted entire chat pool")
    }

    // MARK: - Active chat

    public func append(_ message: Message, to id: UUID) async throws {
        let perfApp = Perf.now()
        defer { Perf.end("io.append", since: perfApp) }
        let folder = chatFolder(id)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        // Defensive meta upsert — the VM creates before it appends, but a
        // missing meta must not lose the message.
        var meta = loadMeta(at: folder) ?? ChatMeta(id: id, title: "New Chat", createdAt: Date(), updatedAt: Date())

        let today = CalendarDate()
        let shardURL = folder.appendingPathComponent(today.filename)
        var shard: DailyLog = if FileManager.default.fileExists(atPath: shardURL.path(percentEncoded: false)) {
            try YAMLDecoder().decode(DailyLog.self, from: Data(contentsOf: shardURL))
        } else {
            DailyLog(date: today)
        }
        shard.append(message)
        shard.totalTokens = nil // in-flight; index() computes on read

        try write(shard, to: shardURL)

        meta.updatedAt = Date()
        try writeMeta(meta)

        logger.info("Appended to chat \(id) shard \(today.filename) (now \(shard.messages.count) messages)")
    }

    public func shardDates(_ id: UUID) async throws -> [CalendarDate] {
        shardURLs(in: chatFolder(id))
            .compactMap { parseDate(from: $0.lastPathComponent) }
            .sorted { $0 > $1 } // newest first
    }

    public func loadShard(_ id: UUID, date: CalendarDate) async throws -> DailyLog? {
        let url = chatFolder(id).appendingPathComponent(date.filename)
        guard FileManager.default.fileExists(atPath: url.path(percentEncoded: false)) else { return nil }
        do {
            return try YAMLDecoder().decode(DailyLog.self, from: Data(contentsOf: url))
        } catch {
            logger.warning("Failed to decode shard \(id)/\(date.filename): \(error.localizedDescription)")
            throw error
        }
    }

    // MARK: - Recall

    public func index() async throws -> [ChatShardRef] {
        let perfIdx = Perf.now()
        let today = CalendarDate()
        var refs: [ChatShardRef] = []
        defer { Perf.end("io.index", since: perfIdx, "shards=\(refs.count)") }
        for folder in chatFolders() {
            guard let id = UUID(uuidString: folder.lastPathComponent) else { continue }
            for shardURL in shardURLs(in: folder) {
                guard let date = parseDate(from: shardURL.lastPathComponent) else { continue }
                // Today's shard reports nil (search excludes today). Finalised
                // shards compute their total on read — same cost the deployed
                // dateIndex() paid by loading every file.
                let total: Int? = if date == today {
                    nil
                } else if let shard = try? await loadShard(id, date: date) {
                    shard.messages.reduce(0) { $0 + tokenCounter.estimateTokens(for: $1) }
                } else {
                    nil
                }
                refs.append(ChatShardRef(chatId: id, date: date, totalTokens: total))
            }
        }
        return refs
    }

    // MARK: - Path + IO helpers

    private func chatFolder(_ id: UUID) -> URL {
        chatsDirectory.appendingPathComponent(id.uuidString, isDirectory: true)
    }

    private func metaURL(_ folder: URL) -> URL {
        folder.appendingPathComponent("meta.yaml")
    }

    /// Immediate subdirectories of the pool root whose name is a UUID.
    private func chatFolders() -> [URL] {
        guard FileManager.default.fileExists(atPath: chatsDirectory.path(percentEncoded: false)),
              let entries = try? FileManager.default.contentsOfDirectory(
                  at: chatsDirectory, includingPropertiesForKeys: [.isDirectoryKey]
              )
        else { return [] }
        return entries.filter { UUID(uuidString: $0.lastPathComponent) != nil }
    }

    /// Day-shard files in a chat folder (every `*.yaml` except `meta.yaml`).
    private func shardURLs(in folder: URL) -> [URL] {
        guard let entries = try? FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
        else { return [] }
        return entries.filter { $0.pathExtension == "yaml" && $0.lastPathComponent != "meta.yaml" }
    }

    private func loadMeta(at folder: URL) -> ChatMeta? {
        let url = metaURL(folder)
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? YAMLDecoder().decode(ChatMeta.self, from: data)
    }

    private func writeMeta(_ meta: ChatMeta) throws {
        let yaml = try YAMLEncoder().encode(meta)
        try yaml.write(to: metaURL(chatFolder(meta.id)), atomically: true, encoding: .utf8)
    }

    private func write(_ shard: DailyLog, to url: URL) throws {
        // Block-literal `|` for multi-line message content, allowUnicode for
        // non-ASCII — carried over verbatim from the deployed adapter.
        let encoder = YAMLEncoder()
        encoder.options.newLineScalarStyle = .literal
        encoder.options.allowUnicode = true
        try encoder.encode(shard).write(to: url, atomically: true, encoding: .utf8)
    }

    private func parseDate(from filename: String) -> CalendarDate? {
        let stem = filename.replacingOccurrences(of: ".yaml", with: "")
        let parts = stem.split(separator: "-")
        guard parts.count == 3,
              let year = Int(parts[0]), let month = Int(parts[1]), let day = Int(parts[2])
        else { return nil }
        return CalendarDate(year: year, month: month, day: day)
    }
}
```

**CyberBench AFTER** (`Packages/Infrastructure/Sources/Infrastructure/FileSystem/Adapters/FileSystemConversationLogStore.swift`):

```swift
import CryptoKit
import Domain
import Foundation
import Yams

/// Day-chunked conversation log in the vault, mirroring the evidence path exactly as
/// `FileSystemDerivedArtifactStore` does (`Evidence/Audio/x.m4a` → `Conversations/Audio/x.m4a/`). Append-only:
/// past day-chunks immutable, only today's rewritten. Stateless — `vaultRoot` per call → app-wide instance.
/// `@concurrent` throughout: YAML + SHA-256 stay off MainActor (matches the derived/inference stores).
nonisolated public struct FileSystemConversationLogStore: ConversationLogStoreProtocol {
    private let logger: (any PlatformLogger)?
    public init(logger: (any PlatformLogger)? = nil) { self.logger = logger }

    // MARK: - Disk documents (FileSpec self-identifying headers; ADR-014 DTO shape)

    private struct MetaDocument: Codable {
        static let currentDocType = "cyberbench/conversation-meta"
        static let currentSchemaVersion = 1
        let docType: String; let schemaVersion: Int
        let id: UUID; let caseId: UUID; let inputRefs: [UUID]
        let startedAt: Date; var updatedAt: Date
        var closedAt: Date?; var closedReason: String?; var contentHash: String?
        init(from m: ConversationMeta) {
            docType = Self.currentDocType; schemaVersion = Self.currentSchemaVersion
            id = m.id; caseId = m.caseId; inputRefs = m.inputRefs; startedAt = m.startedAt
            updatedAt = m.updatedAt; closedAt = m.closedAt; closedReason = m.closedReason
            contentHash = m.contentHash
        }
        func toDomain() -> ConversationMeta {
            ConversationMeta(id: id, caseId: caseId, inputRefs: inputRefs, startedAt: startedAt,
                             updatedAt: updatedAt, closedAt: closedAt, closedReason: closedReason,
                             contentHash: contentHash)
        }
    }

    private struct ChunkDocument: Codable {
        static let currentDocType = "cyberbench/conversation-chunk"
        static let currentSchemaVersion = 1
        let docType: String; let schemaVersion: Int
        let date: CalendarDate; let turns: [ConversationTurn]
        init(from c: ConversationDayChunk) {
            docType = Self.currentDocType; schemaVersion = Self.currentSchemaVersion
            date = c.date; turns = c.turns
        }
        func toDomain() -> ConversationDayChunk { ConversationDayChunk(date: date, turns: turns) }
    }

    private struct DocumentHeader: Codable { let docType: String; let schemaVersion: Int }

    // MARK: - Writes

    @concurrent
    public func create(_ meta: ConversationMeta, for evidence: Evidence, vaultRoot: URL) async throws {
        let dir = conversationDir(meta.id, for: evidence, vaultRoot: vaultRoot)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try writeMeta(meta, dir: dir)
    }

    @concurrent
    public func append(_ turn: ConversationTurn, to conversationId: UUID, for evidence: Evidence,
                       today: CalendarDate, vaultRoot: URL) async throws {
        let dir = conversationDir(conversationId, for: evidence, vaultRoot: vaultRoot)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let chunkURL = dir.appending(path: today.filename)
        var chunk: ConversationDayChunk =
            (try? decodeChunk(at: chunkURL)) ?? ConversationDayChunk(date: today)
        chunk.turns.append(turn)
        try write(ChunkDocument(from: chunk), to: chunkURL)   // today's chunk: full rewrite (bounded)
        if var meta = try? loadMeta(dir: dir) { meta.updatedAt = turn.producedAt; try writeMeta(meta, dir: dir) }
    }

    @concurrent
    public func close(_ conversationId: UUID, for evidence: Evidence, reason: String,
                      at when: Date, vaultRoot: URL) async throws {
        let dir = conversationDir(conversationId, for: evidence, vaultRoot: vaultRoot)
        guard var meta = try? loadMeta(dir: dir), meta.closedAt == nil else { return }  // idempotent
        // Freeze: hash the ordered chunk bytes (set point — "integrity at rest", specs §4).
        var hasher = SHA256()
        for date in try chunkDates(dir: dir) {
            hasher.update(data: try Data(contentsOf: dir.appending(path: date.filename)))
        }
        meta.closedAt = when
        meta.closedReason = reason
        meta.contentHash = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        try writeMeta(meta, dir: dir)
    }

    // MARK: - Reads

    @concurrent
    public func loadConversation(_ conversationId: UUID, for evidence: Evidence, vaultRoot: URL)
        async throws -> Conversation? {
        try reassemble(dir: conversationDir(conversationId, for: evidence, vaultRoot: vaultRoot))
    }

    @concurrent
    public func loadConversation(_ conversationId: UUID, caseVaultRoot vaultRoot: URL)
        async throws -> Conversation? {
        // Rare path (promote/finding validation): no evidence in hand → find the id-named folder.
        guard let dir = findConversationDir(id: conversationId,
                                            under: vaultRoot.appending(path: "Conversations")) else { return nil }
        return try reassemble(dir: dir)
    }

    private func reassemble(dir: URL) throws -> Conversation? {
        guard let meta = try? loadMeta(dir: dir) else { return nil }
        let turns = try chunkDates(dir: dir).flatMap {
            try decodeChunk(at: dir.appending(path: $0.filename)).turns }
        return Conversation(id: meta.id, caseId: meta.caseId, inputRefs: meta.inputRefs,
                            turns: turns, startedAt: meta.startedAt)
    }

    /// Depth-first walk of the mirror tree for a folder named `id` that holds a `meta.yaml`.
    private func findConversationDir(id: UUID, under root: URL) -> URL? {
        let target = id.uuidString
        guard let en = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isDirectoryKey]) else { return nil }
        for case let url as URL in en where url.lastPathComponent == target
            && FileManager.default.fileExists(
                atPath: url.appending(path: "meta.yaml").path(percentEncoded: false)) { return url }
        return nil
    }

    @concurrent
    public func listMeta(for evidence: Evidence, vaultRoot: URL) async throws -> [ConversationMeta] {
        let root = conversationsRoot(for: evidence, vaultRoot: vaultRoot)
        return conversationDirs(in: root)
            .compactMap { try? loadMeta(dir: $0) }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    @concurrent
    public func index(for evidence: Evidence, vaultRoot: URL) async throws -> [ConversationChunkRef] {
        let root = conversationsRoot(for: evidence, vaultRoot: vaultRoot)
        var refs: [ConversationChunkRef] = []
        for dir in conversationDirs(in: root) {
            guard let id = UUID(uuidString: dir.lastPathComponent) else { continue }
            let today = Self.today()
            for date in (try? chunkDates(dir: dir)) ?? [] {
                // Cheap estimate (no tokenizer — Arm B is seam-only); today's chunk reports nil.
                let total: Int? = date == today ? nil
                    : (try? decodeChunk(at: dir.appending(path: date.filename)))?.turns
                        .reduce(0) { $0 + ($1.question.count + $1.answer.count) / 4 }
                refs.append(ConversationChunkRef(conversationId: id, date: date, totalTokens: total))
            }
        }
        return refs
    }

    // MARK: - Path (mirrors FileSystemDerivedArtifactStore.artifactRoot) + IO helpers

    /// "Evidence/Audio/x.m4a" → "<vaultRoot>/Conversations/Audio/x.m4a" — the file's conversation root.
    private func conversationsRoot(for evidence: Evidence, vaultRoot: URL) -> URL {
        let prefix = "Evidence/"
        let inner = evidence.relativePath.hasPrefix(prefix)
            ? String(evidence.relativePath.dropFirst(prefix.count)) : evidence.relativePath
        return vaultRoot.appending(path: "Conversations/\(inner)")
    }
    private func conversationDir(_ id: UUID, for evidence: Evidence, vaultRoot: URL) -> URL {
        conversationsRoot(for: evidence, vaultRoot: vaultRoot).appending(path: id.uuidString)
    }
    private func conversationDirs(in root: URL) -> [URL] {
        guard let entries = try? FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: [.isDirectoryKey]) else { return [] }
        return entries.filter { UUID(uuidString: $0.lastPathComponent) != nil }
    }
    private func chunkDates(dir: URL) throws -> [CalendarDate] {
        let entries = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        return entries.compactMap { parseDate(from: $0.lastPathComponent) }.sorted()   // oldest-first
    }
    private func loadMeta(dir: URL) throws -> ConversationMeta {
        try decode(MetaDocument.self, at: dir.appending(path: "meta.yaml"),
                   docType: MetaDocument.currentDocType, version: MetaDocument.currentSchemaVersion).toDomain()
    }
    private func writeMeta(_ meta: ConversationMeta, dir: URL) throws {
        try write(MetaDocument(from: meta), to: dir.appending(path: "meta.yaml"))
    }
    private func decodeChunk(at url: URL) throws -> ConversationDayChunk {
        try decode(ChunkDocument.self, at: url,
                   docType: ChunkDocument.currentDocType, version: ChunkDocument.currentSchemaVersion).toDomain()
    }
    private func decode<T: Codable>(_ t: T.Type, at url: URL, docType: String, version: Int) throws -> T {
        let data = try Data(contentsOf: url)
        let header = try YAMLDecoder().decode(DocumentHeader.self, from: data)
        guard header.docType == docType else { throw ConversationLogError.wrongDocType(docType, header.docType) }
        guard header.schemaVersion <= version else { throw ConversationLogError.newerSchema(header.schemaVersion) }
        return try YAMLDecoder().decode(T.self, from: data)
    }
    private func write<T: Encodable>(_ doc: T, to url: URL) throws {
        try Data(try encoder().encode(doc).utf8).write(to: url, options: .atomic)
    }
    private func parseDate(from filename: String) -> CalendarDate? {
        guard filename.hasSuffix(".yaml"), filename != "meta.yaml" else { return nil }
        let parts = filename.dropLast(5).split(separator: "-")
        guard parts.count == 3, let y = Int(parts[0]), let m = Int(parts[1]), let d = Int(parts[2])
        else { return nil }
        return CalendarDate(year: y, month: m, day: d)
    }
    /// The ONE clock read — in Infrastructure, keeping CalendarDate (Domain) clock-free (§2.1).
    static func today(now: Date = Date()) -> CalendarDate {
        let c = Calendar.current.dateComponents([.year, .month, .day], from: now)
        return CalendarDate(year: c.year!, month: c.month!, day: c.day!)
    }
    private func encoder() -> YAMLEncoder {   // house style — identical to the derived/inference stores
        let e = YAMLEncoder(); e.options.newLineScalarStyle = .literal; e.options.allowUnicode = true; return e
    }
}
```

**Changed & why (the substance of "not a lift"):**
- **`final class` → `nonisolated public struct` + `@concurrent` methods.** Tanuki's plain-`async` class would
  inherit MainActor under our module default and do YAML/SHA-256 on the main actor; `@concurrent` forces the
  cooperative pool (matches `FileSystemDerivedArtifactStore`/`FileSystemInferenceRecordStore`).
- **Global `chatsDirectory` → per-call `vaultRoot` + evidence-path mirror.** No stored pool root; the path is
  derived per call from `evidence.relativePath` (invariant #2). Enables per-file grouping + the scoped resume
  scan; app-wide stateless instance.
- **`Message` → `ConversationTurn`; `TokenCounter` dropped** (char/4 estimate — Arm B seam-only, no tokenizer
  in the store).
- **Added self-identifying doc headers + `close()`/freeze** (FileSpec discipline + the §4 hash set point) —
  neither exists in Tanuki.
- **Package edges:** `import Domain` (the port + entities), `Yams`, `CryptoKit`; the port lives in Domain, the
  adapter in Infrastructure — the split Tanuki's single module doesn't have.

*(New: `ConversationLogError` in `Domain/Inference/Errors/` — `wrongDocType`, `newerSchema`, mirroring
`InferenceError`'s doc-validation cases.)*

---

## §3 — Modified CyberBench surfaces (before → after)

### 3.1 `MLXPromptCache/SessionStore.swift` — add the eviction surface (library PR)

Simpler than `WarmStore`'s: a session's durable source is the LOG, so **no persist-before-evict** — eviction
just drops RAM; next resume reassembles.

- **BEFORE** — the whole file, verbatim (`MLXPromptCache/Sources/MLXPromptCache/SessionStore.swift`):

```swift
import Foundation
import MLX
import MLXLMCommon

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
///
/// The raw entry points are `package`: reachable by the coordinator seam and the package's own tests,
/// never by external dependents (who use the `public` `PromptCacheCoordinator` doors).
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
- **AFTER — the whole file** (the two `package` methods added before the closing brace; bodies mirror
  `WarmStore.victimsOverBudget`, no persistence — the log is the durable source, so eviction just drops RAM):

```swift
import Foundation
import MLX
import MLXLMCommon

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
///
/// The raw entry points are `package`: reachable by the coordinator seam and the package's own tests,
/// never by external dependents (who use the `public` `PromptCacheCoordinator` doors).
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

    /// Live resident bytes across all held conversations. (`WarmStore` caches this per-`Entry`; `SessionStore`
    /// holds raw `[KVCache]`, so it recomputes.) Call inside `perform`.
    package var residentBytes: Int { live.values.reduce(0) { $0 + WarmStore.footprint($1) } }

    /// Ids to drop when resident bytes exceed an APP-SUPPLIED budget, largest-first (size policy, not LRU).
    /// The budget is a parameter — unlike `WarmStore`, which stores its own. Call inside `perform`.
    package func victimsOverBudget(_ budgetBytes: Int, excluding keep: UUID) -> [UUID] {
        guard budgetBytes > 0, residentBytes > budgetBytes else { return [] }
        var over = residentBytes - budgetBytes
        var out: [UUID] = []
        for (id, cache) in live.sorted(by: { WarmStore.footprint($0.value) > WarmStore.footprint($1.value) })
        where id != keep {
            out.append(id)
            over -= WarmStore.footprint(cache)
            if over <= 0 { break }
        }
        return out
    }
}
```

**Both are `package`** (concurrency review): external dependents never touch `SessionStore` directly, and all
`live` access must be inside `ModelContainer.perform` (the type's `@unchecked Sendable` invariant). External
callers go through `PromptCacheCoordinator`'s session doors, which already encode that contract.

**BEFORE** — the coordinator's session-door extension, verbatim (`PromptCacheCoordinator.swift:325-350`):

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
}
```

`release(_ sessions:, id:)` is the precedent — `sessions` first, and *"Call inside `perform`"* is its **real**
documented contract, not a note I invented. The warm side already runs budget eviction the same way, inside
`warm` (`PromptCacheCoordinator.swift:269-273`, verbatim):

```swift
        // 5. Budget: persist-then-release the largest other warms if we are over.
        for victim in warms.victimsOverBudget(excluding: id) {
            persistHeld(warms, id: victim)
            warms.release(victim)
        }
```

**AFTER** — the full extension, with `evictSessions` added beside `advance`/`release`:

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

`MLXConversationEngine`, **inside the existing `container.perform` block, right after `advance`**, calls
`coordinator.evictSessions(sessions, overBudget: sessionBudgetBytes(), keep: conversationId)` — where
`sessionBudgetBytes: @Sendable () -> Int` is injected and resolves **live system RAM on each call**
(adr-cb-008 resolve-on-each-use). The app owns the number; the library owns the mechanism; the VM never
touches `SessionStore`.

**`WarmStore.footprint` — visibility widened so `SessionStore` can reuse it.**

BEFORE — verbatim (`WarmStore.swift:106-112`):

```swift
    /// Bytes of KV state a cache holds. Reads `state`, which for an attention layer is already
    /// sliced to `offset`, so this is the live footprint rather than the allocated capacity.
    static func footprint(_ cache: [KVCache]) -> Int {
        cache.reduce(0) { total, layer in
            total + layer.state.reduce(0) { $0 + $1.nbytes }
        }
    }
```

AFTER — `package` prepended (currently file-private `static`; `SessionStore` is in the same module but a
different file, so it needs `package` to reach it); body unchanged:

```swift
    /// Bytes of KV state a cache holds. Reads `state`, which for an attention layer is already
    /// sliced to `offset`, so this is the live footprint rather than the allocated capacity.
    package static func footprint(_ cache: [KVCache]) -> Int {
        cache.reduce(0) { total, layer in
            total + layer.state.reduce(0) { $0 + $1.nbytes }
        }
    }
```

### 3.2 `InferenceRecordStoreProtocol` — remove the conversation methods

⚠️ **Same file R1 reworks — sequence with the BOM step.** The split *reduces* what R1 touches.

- **BEFORE** — the three conversation members of `InferenceRecordStoreProtocol.swift`, verbatim (`:36-45`):

```swift
    /// Persist the conversation record (vault case material, C6 flag 3). Rewritten
    /// atomically after each turn; best-effort relative to the turn's call record (the
    /// commit point).
    func writeConversation(_ conversation: Conversation) async throws
    
    func loadConversation(id: UUID, caseId: UUID) async throws -> Conversation?
    
    /// The review sweep, mirroring allAnalyses: every conversation the vault holds,
    /// header-validated, undecodable files skipped (row-skip posture).
    func allConversations(caseId: UUID) async throws -> [Conversation]
```

  (For context, the file's header comment also names conversations: *"Analyses (body + record) AND
  conversations live in the VAULT…"* — that sentence's conversation clause goes too.)
- **AFTER:** those three members **deleted** — conversations move to `ConversationLogStoreProtocol`. The port
  keeps only `writeAnalysis` / `writeCall` / `loadCall` / `loadAnalysis` / `analysisBody` / `promoteAnalysis`
  / `allAnalyses` (the analysis + call-ledger surface, unchanged).

### 3.3 `FileSystemInferenceRecordStore` — drop the conversation DTO + methods

- **BEFORE** — verbatim, the `ConversationDocument` DTO (`FileSystemInferenceRecordStore.swift:117-148`):

```swift
    private struct ConversationDocument: Codable {
        static let currentDocType = "cyberbench/conversation"
        static let currentSchemaVersion = 1
        
        let docType: String
        let schemaVersion: Int
        let id: UUID
        let caseId: UUID
        let inputRefs: [UUID]
        let turns: [ConversationTurn]
        let startedAt: Date
        
        init(from conversation: Conversation) {
            docType = Self.currentDocType
            schemaVersion = Self.currentSchemaVersion
            id = conversation.id
            caseId = conversation.caseId
            inputRefs = conversation.inputRefs
            turns = conversation.turns
            startedAt = conversation.startedAt
        }
        
        func toDomain() -> Conversation {
            Conversation(
                id: id,
                caseId: caseId,
                inputRefs: inputRefs,
                turns: turns,
                startedAt: startedAt
            )
        }
    }
```

  …and the three methods (`:389-432`):

```swift
    @concurrent public func writeConversation(_ conversation: Conversation) async throws {
        let conversationsDir = vaultRoot.appending(path: "Conversations")
        try FileManager.default.createDirectory(
            at: conversationsDir, withIntermediateDirectories: true
        )
        let yaml = try encoder().encode(ConversationDocument(from: conversation))
        try Data(yaml.utf8).write(
            to: conversationsDir.appending(path: "\(conversation.id.uuidString).yaml"),
            options: .atomic
        )
    }
    
    @concurrent public func loadConversation(id: UUID, caseId: UUID) async throws -> Conversation? {
        let url = vaultRoot.appending(path: "Conversations/\(id.uuidString).yaml")
        guard let data = try? Data(contentsOf: url) else { return nil}
        let conversation = try decodeDocument(
            ConversationDocument.self,
            from: data,
            expecting: ConversationDocument.currentDocType,
            supportedVersion: ConversationDocument.currentSchemaVersion).toDomain()
        return conversation.caseId == caseId ? conversation : nil
    }
    
    /// The review sweep (mirrors allAnalyses' posture): header-validated per file; a file
    /// that fails validation is SKIPPED, not fatal — a pulled vault may hold newer-schema
    /// records and one bad file must not blank the listing.
    @concurrent public func allConversations(caseId: UUID) async throws -> [Conversation] {
        var conversations: [Conversation] = []
        for record in yamlFiles(in: vaultRoot.appending(path: "Conversations")) {
            do {
                let data = try Data(contentsOf: record)
                let conversation = try decodeDocument(
                    ConversationDocument.self,
                    from: data,
                    expecting: ConversationDocument.currentDocType,
                    supportedVersion: ConversationDocument.currentSchemaVersion
                ).toDomain()
                if conversation.caseId == caseId { conversations.append(conversation) }
            } catch {
                logger?.warning("allConversations: skipped Conversations/\(record.lastPathComponent) — \(error)")
            }
        }
        return conversations.sorted { $0.startedAt < $1.startedAt }
    }
```

- **AFTER:** the DTO + all three methods **deleted** (they live in `FileSystemConversationLogStore`, §2.6, in
  the day-chunked shape). Note the old single-file docType was `cyberbench/conversation` — **retired** in
  favour of `cyberbench/conversation-meta` + `-chunk` (the R4 doc-type note). Everything else in the file is
  untouched.

### 3.4 `InferenceRecorder.recordTurn` — append to the log instead of full-rewrite

- **BEFORE** — the whole `recordTurn` method, verbatim (`InferenceRecorder.swift:84-129`):

```swift
    public func recordTurn(
        conversation: Conversation,
        question: String,
        execution: SlotExecution,
        prepared: SelectionPromptAssembler.AssembledPrompt,
        startedAt: Date
    ) async throws -> (turn: ConversationTurn, conversation: Conversation) {
        let provenance = await successProvenance(
            execution: execution,
            requestedRole: .deep,
            promptHash: prepared.promptHash,
            inputRefs: conversation.inputRefs
        )
        let callId = UUID()
        let turnId = UUID()
        // Last exit: cancel before the first byte hits disk writes NOTHING.
        try Task.checkCancellation()
        try await records.writeCall(
            InferenceCall(
                id: callId,
                caseId: conversation.caseId,
                provenance: provenance,
                startedAt: startedAt,
                completedAt: Date(),
                outcome: .answered(conversationRef: conversation.id, turnId: turnId)
            )
        )
        let turn = ConversationTurn(
            id: turnId,
            question: question,
            answer: execution.output.text,
            inferenceCallRef: callId,
            producedAt: Date()
        )
        var updated = conversation
        updated.turns.append(turn)
        do {
            try await records.writeConversation(updated)
        } catch {
            logger.warning(
                "Conversation rewrite failed (ledger committed; next turn catches up): \(error.localizedDescription)"
            )
        }
        logger.info("Turn recorded: call \(callId) → conversation \(conversation.id)")
        return (turn, updated)
    }
```
- **AFTER — the whole method.** The `InferenceRecorder` gains a `conversationLog: ConversationLogStoreProtocol`
  dependency (added to its `init`); `recordTurn` gains `for evidence: Evidence` + `vaultRoot: URL`; the
  call-write (the commit point) is byte-identical; only the conversation write changes from
  `records.writeConversation(updated)` (full-rewrite) to `conversationLog.append(...)`:

```swift
    public func recordTurn(
        conversation: Conversation,
        question: String,
        execution: SlotExecution,
        prepared: SelectionPromptAssembler.AssembledPrompt,
        startedAt: Date,
        for evidence: Evidence,
        vaultRoot: URL
    ) async throws -> (turn: ConversationTurn, conversation: Conversation) {
        let provenance = await successProvenance(
            execution: execution,
            requestedRole: .deep,
            promptHash: prepared.promptHash,
            inputRefs: conversation.inputRefs
        )
        let callId = UUID()
        let turnId = UUID()
        // Last exit: cancel before the first byte hits disk writes NOTHING.
        try Task.checkCancellation()
        try await records.writeCall(
            InferenceCall(
                id: callId,
                caseId: conversation.caseId,
                provenance: provenance,
                startedAt: startedAt,
                completedAt: Date(),
                outcome: .answered(conversationRef: conversation.id, turnId: turnId)
            )
        )
        let turn = ConversationTurn(
            id: turnId,
            question: question,
            answer: execution.output.text,
            inferenceCallRef: callId,
            producedAt: Date()
        )
        var updated = conversation
        updated.turns.append(turn)
        // The one changed block: today's chunk gets the turn, instead of rewriting the whole conversation.
        // Application reads the clock here (Domain's CalendarDate stays clock-free, §2.1) rather than calling
        // the Infrastructure store's static — that would be an Application→Infrastructure layer violation.
        let c = Calendar.current.dateComponents([.year, .month, .day], from: Date())
        let today = CalendarDate(year: c.year!, month: c.month!, day: c.day!)
        do {
            try await conversationLog.append(
                turn, to: conversation.id, for: evidence, today: today, vaultRoot: vaultRoot)
        } catch {
            logger.warning(
                "Conversation append failed (ledger committed; next turn catches up): \(error.localizedDescription)"
            )
        }
        logger.info("Turn recorded: call \(callId) → conversation \(conversation.id)")
        return (turn, updated)
    }
```

*(`recordTurn`'s signature gains `evidence: Evidence` + `vaultRoot: URL`; `RunConversationTurnUseCase` already
has both — it passes `selection`/`vaultRoot` through.)*

### 3.5 NEW use cases (Application) — `ResumeConversationUseCase` + `CloseConversationUseCase`

```swift
// Packages/Application/Sources/Application/Inference/UseCases/ResumeConversationUseCase.swift
import Domain
import Foundation

public protocol ResumeConversationUseCaseProtocol: Sendable {
    /// The ACTIVE conversation for a file, reassembled from the log — or nil if none/all closed.
    /// "Active" = newest `updatedAt`, not closed. The hot-swap lookup.
    func execute(for evidence: Evidence, vaultRoot: URL) async throws -> Conversation?
}

public final class ResumeConversationUseCase: ResumeConversationUseCaseProtocol {
    private let store: ConversationLogStoreProtocol
    public init(store: ConversationLogStoreProtocol) { self.store = store }
    public func execute(for evidence: Evidence, vaultRoot: URL) async throws -> Conversation? {
        guard let active = try await store.listMeta(for: evidence, vaultRoot: vaultRoot)
            .first(where: { $0.closedAt == nil }) else { return nil }
        return try await store.loadConversation(active.id, for: evidence, vaultRoot: vaultRoot)
    }
}
```

**`CloseConversationUseCase`** — the seam the VM uses so it never holds the Domain port (hex BLOCKER fix).
"Close" freezes the log (the §4 hash set point) AND releases the held session; one action, two effects.

```swift
// Packages/Application/Sources/Application/Inference/UseCases/CloseConversationUseCase.swift
import Domain
import Foundation

public protocol CloseConversationUseCaseProtocol: Sendable {
    /// "This chat is over": freeze the conversation log (stamp closedAt + contentHash) and release the
    /// held session. Idempotent. Used by Fresh Chat and by the model-change seal (ruling 4).
    func execute(_ conversation: Conversation, for evidence: Evidence, reason: String, vaultRoot: URL) async
}

public final class CloseConversationUseCase: CloseConversationUseCaseProtocol {
    private let store: ConversationLogStoreProtocol
    private let registry: any SlotRegistryProtocol
    public init(store: ConversationLogStoreProtocol, registry: any SlotRegistryProtocol) {
        self.store = store; self.registry = registry
    }
    public func execute(_ conversation: Conversation, for evidence: Evidence,
                        reason: String, vaultRoot: URL) async {
        try? await store.close(conversation.id, for: evidence, reason: reason, at: Date(), vaultRoot: vaultRoot)
        await registry.endConversation(conversation.id, caseId: conversation.caseId, role: .deep)
    }
}
```

*(`EndConversationUseCase` stays for window-close — release-only, no freeze, so a closed window's chat is
still resumable. Only Fresh Chat + model-change truly close.)*

### 3.6 `ConversationReducer` / `ConversationState` — new actions

- **`ConversationState`** gains nothing (the same fields carry a resumed conversation).
- **`ConversationReducer.Action`** (`ConversationReducer.swift:5-20`): **replace `.selectionChanged`** (which
  reset to empty) **with** `.conversationResumed(Conversation?)`, and **add** `.freshChatStarted`.

**BEFORE** — the `Action` enum + the `.selectionChanged` reduce case, verbatim (`ConversationReducer.swift`):

```swift
    public enum Action: Sendable {
        case questionEdited(String)
        case thinkingToggled(Bool)
        case askStarted(Conversation)
        case chunkArrived(String)
        case turnCompleted(turn: ConversationTurn, conversation: Conversation)
        case runCancelled
        case runFailed(String)
        case selectionChanged
        case composerEdited(String)
        case commentEdited(String)
        case findingComposed
        case composeFailed(String)
        case conversationEnded(reason: String)
        case errorDismissed
    }

    // …and its reduce(state:action:) case:
        case .selectionChanged:
            // The chat FOLLOWS the navigator (owner ruling, 2026-07-19): a different Selection is
            // a different conversation, so the dock resets to an empty chat for the new evidence.
            // Nothing durable is lost — every turn is already on disk twice before the VM sees it:
            // its InferenceCall in the ledger (the commit point) and the rolled-up
            // Conversations/<id>.yaml. This clears the VIEW, never the record.
            state.conversation = nil
            state.turns = []
            state.liveAnswer = ""
            state.question = ""
            state.phase = .idle
            state.endedNotice = nil
            state.errorMessage = nil
```

**AFTER** — `.selectionChanged` → `.conversationResumed`; add `.freshChatStarted`:

```swift
        case .conversationResumed(let conversation):
            // Hot-swap: adopt the file's active conversation (nil = none yet → empty chat, first ask mints one).
            state.conversation = conversation
            state.turns = conversation?.turns ?? []
            state.liveAnswer = ""; state.question = ""; state.phase = .idle
            state.endedNotice = nil; state.errorMessage = nil
        case .freshChatStarted:
            // Fresh Chat: the old conversation is closed by the VM; the dock resets so the next ask mints new.
            state.conversation = nil; state.turns = []
            state.liveAnswer = ""; state.question = ""; state.phase = .idle
```

### 3.7 `ConversationViewModel` — focus→resume, Fresh Chat, seal-all, hold-N

- **`focus(on:)`** (`ConversationViewModel.swift:187-197`):
  - **BEFORE** — the whole `focus(on:)` method + its doc comment, verbatim (`ConversationViewModel.swift:179-197`):

```swift
    /// The navigator selection changed and the chat follows it (owner ruling, 2026-07-19). Ends the
    /// outgoing conversation — which releases the engine's held session — and resets the dock to an
    /// empty chat for the new evidence. No-op when the selection is unchanged, so a re-render can
    /// never destroy a live conversation.
    ///
    /// Known consequence: `compose()` needs a live aggregate, so a finding must be composed BEFORE
    /// switching. The conversation itself is durable (ledger + Conversations/<id>.yaml); what does
    /// not exist yet is a view that loads one back — `allConversations` is the seam for it.
    public func focus(on refs: [UUID]) {
        guard let existing = conversation, existing.inputRefs != refs else { return }
        // Cancel any in-flight turn for the OUTGOING selection first. The use case's cancel arm
        // ends the conversation registry-side; the explicit end below is idempotent.
        stop()
        let endConversation = self.endConversation
        let id = existing.id
        let caseId = existing.caseId
        Task { await endConversation.execute(conversationId: id, caseId: caseId) }
        dispatch(.selectionChanged)
    }
```
  - **AFTER:** `stop()` the in-flight turn only; **do NOT end** the outgoing conversation (its KV stays resident
    in `SessionStore`, budget-governed); resume the incoming file:

```swift
    @ObservationIgnored private var resumeTask: Task<Void, Never>?
    public func focus(on evidence: Evidence?) {   // now takes the Evidence, for the store lookup
        guard let evidence, evidence.id != conversation?.inputRefs.first else { return }
        stop()
        resumeTask?.cancel()                          // WARN fix: a newer selection cancels the prior resume
        resumeTask = Task { [weak self] in
            guard let self else { return }
            let resumed = try? await self.resumeConversation.execute(for: evidence, vaultRoot: self.vaultRoot)
            if Task.isCancelled { return }            // drop a stale reassembly — no racing dispatch
            self.dispatch(.conversationResumed(resumed))
        }
    }
```

- **`freshChat()`** — NEW: closes the active conversation (a set point — freezes the hash, §4) and resets:

```swift
    public func freshChat() {   // injects closeConversation: CloseConversationUseCaseProtocol — NOT the port
        stop()
        if let c = conversation, let evidence = target.first {
            Task { await closeConversation.execute(c, for: evidence, reason: "new chat", vaultRoot: vaultRoot) }
        }
        dispatch(.freshChatStarted)
    }
```

- **VM dependencies:** injects `resumeConversation: ResumeConversationUseCaseProtocol` +
  `closeConversation: CloseConversationUseCaseProtocol` (use cases, per ADR-015) — never `ConversationLogStoreProtocol`.
- **Model change → seal ALL.** **BEFORE** — the residency watcher `start()` and the `end(reason:)` it calls,
  verbatim (`ConversationViewModel.swift:110-121, 232-239`):

```swift
    public func start() async {
        let stream = await observeResidency.execute()
        var lastDeepModelId: String?
        for await residency in stream {
            let current = residency[.deep]?.modelId
            if let previous = lastDeepModelId, previous != current, conversation != nil {
                logger.info("Deep slot re-tenanted (\(previous) → \(current ?? "none")); ending conversation")
                end(reason: "Model changed — conversation ended.")
            }
            lastDeepModelId = current
        }
    }

    public func end(reason: String) {
        guard let conversation else { return }
        let endConversation = endConversation
        let id = conversation.id
        let caseId = conversation.caseId
        Task { await endConversation.execute(conversationId: id, caseId: caseId) }
        dispatch(.conversationEnded(reason: reason))
    }
```

  **AFTER:** on a deep-slot model change the watcher closes+seals **every** held conversation via
  `closeConversation.execute(…, reason: "model changed")` — freeze each log + release each session (ruling 4).
  `end(reason:)`'s single-conversation, release-only path is superseded for the model-change case.
- **Hold-N + budget:** `SessionStore` already holds N by id. Eviction is **not** the VM's job — it runs in
  `MLXConversationEngine` inside `perform` after `advance`, via `coordinator.evictSessions(...)` reading the
  injected `sessionBudgetBytes` provider (§3.1). Evicting = a reassembly next time, never a wrong answer.

*(`AskPane.swift:41-42` changes `focus(on: target.map(\.id))` → `focus(on: target.first)` — it now hands the
Evidence, not the id list; still single-file v1.)*

### 3.8 `WindowServices` — wiring (per-case, `vaultRoot = project.folder`)

- **BEFORE** — the whole `makeCaseInference` factory, verbatim (`WindowServices.swift:126-183`):

```swift
    private static func makeCaseInference(
        project: Project, index: CaseIndex, _ d: AppDependencies
    ) -> CaseInference {
        // ONE store, ONE assembler, ONE recorder — shared by the one-shot and conversation doors
        // (decision 16: one crash-contract home per case).
        let inferenceRecords = FileSystemInferenceRecordStore(vaultRoot: project.folder, logger: d.logger)
        let assembler = SelectionPromptAssembler(derivedStore: d.derivedStore)
        let recorder = InferenceRecorder(
            records: inferenceRecords,
            assets: d.modelAssetStore,
            index: index.analyses,
            logger: d.logger
        )

        let interrogation = InterrogationViewModel(
            caseId: project.manifest.id,
            vaultRoot: project.folder,
            runInference: RunInferenceUseCase(
                assembler: assembler,
                recorder: recorder,
                registry: d.slotRegistry
            ),
            promoteAnalysis: PromoteAnalysisToFindingUseCase(
                records: inferenceRecords,
                assets: d.modelAssetStore,
                index: index.analyses,
                logger: d.logger
            ),
            observeAnalyses: ObserveAnalysesUseCase(observation: index.analysisObservation),
            loadBody: LoadAnalysisBodyUseCase(records: inferenceRecords),
            backfillIndex: BackfillAnalysisIndexUseCase(
                records: inferenceRecords,
                index: index.analyses,
                logger: d.logger
            ),
            logger: d.logger
        )

        let conversation = ConversationViewModel(
            caseId: project.manifest.id,
            vaultRoot: project.folder,
            runTurn: RunConversationTurnUseCase(
                assembler: assembler,
                recorder: recorder,
                registry: d.slotRegistry
            ),
            composeFinding: ComposeChatFindingUseCase(
                records: inferenceRecords,
                index: index.analyses,
                logger: d.logger
            ),
            endConversation: EndConversationUseCase(registry: d.slotRegistry),
            observeResidency: ObserveSlotResidencyUseCase(registry: d.slotRegistry),
            logger: d.logger
        )

        return CaseInference(interrogation: interrogation, conversation: conversation)
    }
```

  (Note `InterrogationViewModel` + `RunInferenceUseCase` here — R2 removes those; this story only adds to the
  `conversation` half. The two overlap in this one factory, which is why R1+R2+this are cleanest as one PR.)
- **AFTER — the whole method** (added lines marked; `interrogation` is untouched — R2 removes it, not this
  story):

```swift
    private static func makeCaseInference(
        project: Project, index: CaseIndex, _ d: AppDependencies
    ) -> CaseInference {
        // ONE store, ONE assembler, ONE recorder — shared by the one-shot and conversation doors
        // (decision 16: one crash-contract home per case).
        let inferenceRecords = FileSystemInferenceRecordStore(vaultRoot: project.folder, logger: d.logger)
        let conversationLog = FileSystemConversationLogStore(logger: d.logger)   // ADDED — app-wide, stateless
        let assembler = SelectionPromptAssembler(derivedStore: d.derivedStore)
        let recorder = InferenceRecorder(
            records: inferenceRecords,
            assets: d.modelAssetStore,
            index: index.analyses,
            conversationLog: conversationLog,   // ADDED — recorder appends turns to the day-chunked log
            logger: d.logger
        )

        let interrogation = InterrogationViewModel(
            caseId: project.manifest.id,
            vaultRoot: project.folder,
            runInference: RunInferenceUseCase(
                assembler: assembler,
                recorder: recorder,
                registry: d.slotRegistry
            ),
            promoteAnalysis: PromoteAnalysisToFindingUseCase(
                records: inferenceRecords,
                assets: d.modelAssetStore,
                index: index.analyses,
                logger: d.logger
            ),
            observeAnalyses: ObserveAnalysesUseCase(observation: index.analysisObservation),
            loadBody: LoadAnalysisBodyUseCase(records: inferenceRecords),
            backfillIndex: BackfillAnalysisIndexUseCase(
                records: inferenceRecords,
                index: index.analyses,
                logger: d.logger
            ),
            logger: d.logger
        )

        let conversation = ConversationViewModel(
            caseId: project.manifest.id,
            vaultRoot: project.folder,
            runTurn: RunConversationTurnUseCase(
                assembler: assembler,
                recorder: recorder,
                registry: d.slotRegistry
            ),
            composeFinding: ComposeChatFindingUseCase(
                records: inferenceRecords,
                index: index.analyses,
                logger: d.logger
            ),
            resumeConversation: ResumeConversationUseCase(store: conversationLog),   // ADDED
            closeConversation: CloseConversationUseCase(                             // ADDED
                store: conversationLog, registry: d.slotRegistry),
            endConversation: EndConversationUseCase(registry: d.slotRegistry),       // stays: window-close
            observeResidency: ObserveSlotResidencyUseCase(registry: d.slotRegistry),
            logger: d.logger
        )

        return CaseInference(interrogation: interrogation, conversation: conversation)
    }
```

  (The `sessionBudgetBytes` provider is injected into `MLXConversationEngine`, not built here —
  `MLXConversationEngine` is constructed in the per-resident-model factory, and that injection is §3.9's
  before→after. Land these additions in the decomposed window-services helper, not a monolith — CR1–CR3.)

### 3.9 `MLXConversationEngine` — inject the budget provider + evict after `advance`

**BEFORE** — the stored properties + `init`, verbatim (`MLXConversationEngine.swift:14-32`):

```swift
    private let host: MLXModelHost
    private let cacheStores: PromptCacheStoreProvider
    private let logger: PlatformLogger
    private let sessions = SessionStore()
    /// Live warm prefixes, SHARED with `MLXContextWarmer` (MLXPromptCache 0.5.0). Both doors warm
    /// the same per-file prefix, so both must reach the same holder: a warm that yields mid-prefill
    /// now keeps its cache in RAM instead of writing it out, and a holder this door cannot see is a
    /// holder whose work this door re-prefills from cold. Injected rather than constructed — the
    /// factory owns the one instance per resident model, exactly as it owns the one `MLXModelHost`.
    private let warms: WarmStore

    public init(
        host: MLXModelHost,
        cacheStores: PromptCacheStoreProvider,
        warms: WarmStore,
        logger: PlatformLogger
    ) {
        self.host = host
        self.cacheStores = cacheStores
        self.warms = warms
        self.logger = logger
    }
```

**AFTER** — one stored property + one `init` parameter (the injected, live-resolving budget):

```swift
    private let warms: WarmStore
    private let sessionBudgetBytes: @Sendable () -> Int   // app-owned; resolves live RAM (adr-cb-008)

    public init(
        host: MLXModelHost,
        cacheStores: PromptCacheStoreProvider,
        warms: WarmStore,
        sessionBudgetBytes: @escaping @Sendable () -> Int,
        logger: PlatformLogger
    ) {
        self.host = host
        self.cacheStores = cacheStores
        self.warms = warms
        self.sessionBudgetBytes = sessionBudgetBytes
        self.logger = logger
    }
```

**BEFORE** — the successful held-session branch after `advance`, verbatim (`MLXConversationEngine.swift:224-233`):

```swift
                } else {
                    generateInput = advanced.input
                    cache = advanced.cache
                    // `reused` is derived from values already in hand: advance() returns
                    // fullPromptTokens[heldLength...], so total − prefilled IS what the held
                    // session already covered. No extra work.
                    let prefilled = generateInput.text.tokens.shape.last ?? 0
                    logger.info("[MLX] chat turn — \(fullTokens.count) total, "
                        + "\(prefilled) prefilled, \(fullTokens.count - prefilled) reused (held session)")
                }
```

**AFTER** — evict other over-budget sessions once THIS conversation's cache is confirmed resident — still
inside the same `container.perform`, so the perform-contract on `SessionStore`/coordinator holds
(`sessionBudgetBytes` is captured before `perform` alongside `sessions`/`warms`, exactly as they already are,
`:59-63`):

```swift
                } else {
                    generateInput = advanced.input
                    cache = advanced.cache
                    let prefilled = generateInput.text.tokens.shape.last ?? 0
                    logger.info("[MLX] chat turn — \(fullTokens.count) total, "
                        + "\(prefilled) prefilled, \(fullTokens.count - prefilled) reused (held session)")
                    // Hold-N budget: this conversation is now resident — drop the largest OTHERS if over.
                    coordinator.evictSessions(sessions, overBudget: sessionBudgetBytes(), keep: conversationId)
                }
```

  (`coordinator`, `sessions`, and `conversationId` are already in scope here — `coordinator` built at
  `:85`, `sessions`/`warms` captured at `:59-63`. The evict goes only in the *successful held-session* branch;
  the degraded full-prefill and store-unavailable arms hold nothing to evict.)

---

## §4 — Custody (spec-grounded; the two R1/R2 edges)

**Hashing is at SET POINTS, never rolling** — from the specs: `contentHash` is *"computed once when the
artifact is generated," "integrity at rest / on snapshot,"* frozen at snapshot (derived-data treatment);
custody is *"git commit DAG + batch-at-seal,"* versioned *"only at a seal"* (BOM design §1.3 / SPEC-6).
Applied here:
- Each **day-chunk** is immutable once past → hashable at rest (today's is mutable, unhashed).
- **`ConversationMeta.contentHash` is nil while open, frozen at `close()`** (§2.6) — the conversation's "at
  rest" moment, the exact placement of "hash as evidence comes in."
- A **finding pins the hash it saw at compose** (git-commit semantics; the chat may continue).

**Owned by R1/R2 (this story freezes the hash; the BOM step consumes it):**
1. **Where the conversation `contentHash` enters the BOM reference graph** (and the rollup grain — whole-
   conversation here; per-chunk if the graph wants leaf granularity).
2. **The finding↔conversation edge** — `AnalysisComposition.conversationRef` (R1) + the pinned hash close it;
   no turn-selection.

---

## §5 — DoD

- Selecting a file **resumes** its active conversation (turns shown from the log) instead of an empty chat;
  the outgoing conversation's KV stays resident.
- N conversations held live; swapping between files re-prefills no evidence corpus (catalog-probe); only a
  resumed conversation's turns re-prefill, once. Over an app-supplied budget → largest-first eviction, logged;
  an evicted session reassembles identically next time.
- After relaunch, reselecting a file reassembles from the day-chunked log (warm evidence prefix + turns delta).
- Turns persist as `Conversations/<evidence-path>/<id>/<date>.yaml`; per-turn write touches only today's chunk.
- Fresh Chat closes the active conversation (freezes `contentHash`) and starts a new one; a deep-slot model
  change closes + seals **all** held conversations.
- A pulled vault's conversations list + load with no reconcile step (direct vault reads).
- Tests: resume round-trip across two dates (oldest-first reassembly); `close` freezes hash + is idempotent;
  `victimsOverBudget` largest-first; reassembly-after-eviction == never-evicted run; `focus` resumes not ends;
  `append` touches only today's chunk; header-validated decode skips a bad file (logged); the by-id
  `loadConversation(_:caseVaultRoot:)` resolves a conversation without the evidence (walks the mirror tree).

---

## §6 — Open / future

- **This epic, at cut:** reload UX (resume affordance; LRU vs focus-biased eviction; show transcript
  immediately or on request); the `SessionMemoryBudget` derivation from live RAM (precedent: K3 disk budget).
- **Future (owner leans against, not this epic):** multi-file **corpus** conversations — the evidence-path
  mirror can't hold a multi-file selection; needs its own layout call. Decide against cited sources if revisited.

---

## §7 — Blast radius

- **NEW** — Domain: `CalendarDate`, `ConversationDayChunk`, `ConversationChunkRef`, `ConversationMeta`,
  `ConversationLogStoreProtocol`, `ConversationLogError`. Application: `ResumeConversationUseCase`,
  `CloseConversationUseCase` (+ protocols). Infrastructure: `FileSystemConversationLogStore`.
- **MODIFY** — Library: `SessionStore` (+ `package` eviction), `PromptCacheCoordinator` (+ `evictSessions`
  door), `WarmStore.footprint` (→ `package static`). Domain: `InferenceRecordStoreProtocol` (−3 methods).
  Application: `InferenceRecorder.recordTurn`, `RunConversationTurnUseCase` (thread `evidence`). Infrastructure:
  `FileSystemInferenceRecordStore` (−conversation DTO+methods), `MLXConversationEngine` (in-`perform` eviction
  + injected `sessionBudgetBytes` provider). Presentation: `ConversationViewModel`, `ConversationReducer`. App:
  `WindowServices`, `AskPane`.
- **Tests** — new `FileSystemConversationLogStoreTests`, `ResumeConversationUseCaseTests`, `SessionStore`
  eviction; updated `InferenceRecorderTests`, `FileSystemInferenceRecordStoreTests` (drop conversation cases),
  `ConversationReducerTests`/`ConversationViewModelTests` (resume/freshChat).

---

## §8 — R1/R2 reconciliation

Sequence the `InferenceRecordStoreProtocol` / `FileSystemInferenceRecordStore` conversation-removal (§3.2–3.3)
with R1's edits to those same files (they collapse the analysis model). The removal *shrinks* R1's surface.
Once R1 lands, wire the §4 edges as built (`conversationRef` + pinned hash); nothing in §2 changes.

**⚠ Owned cross-story break — the promote-gate conversation loader (flagged 2026-07-21).** R1 §2.5's new
`PromoteAnalysisToFindingUseCase` body calls `records.loadConversation(id:caseId:)` on
`InferenceRecordStoreProtocol` — the exact method this story's §3.2 **deletes** (conversations move to
`ConversationLogStoreProtocol`). Whichever of {R1, this story} lands second **owns** repointing that gate to the **by-id loader added for
exactly this** — `ConversationLogStoreProtocol.loadConversation(_:caseVaultRoot:)` (§2.5/§2.6). It walks the
mirror tree, so the gate needs only the `conversationRef` + the injected case `vaultRoot` — **no** evidence
resolution (the evidence-scoped `loadConversation(_:for:vaultRoot:)` stays the hot resume path). A near-
mechanical one-line repoint. If R1+this story
land as one PR (recommended — both already want one PR with R1+R2), do the repoint there and this note closes.
Left unowned, it is a guaranteed build break at merge.
