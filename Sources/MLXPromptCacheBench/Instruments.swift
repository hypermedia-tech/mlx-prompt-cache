import Darwin
import Foundation
import MLX
import Synchronization

// MARK: - Verdicts

/// Three-state, not two. A timing gate whose run was thermally throttled or memory-pressured is
/// `invalid` — neither pass nor fail — because reporting a green on noise is worse than no gate.
enum Verdict: String, Sendable, Codable {
    case pass = "PASS"
    case fail = "FAIL"
    case invalid = "INVALID"   // measured, but the run conditions void the number
    case skip = "SKIP"         // not applicable to this model

    var glyph: String {
        switch self {
        case .pass: "✅"
        case .fail: "❌"
        case .invalid: "⚠️ "
        case .skip: "—"
        }
    }
}

struct GateResult: Sendable, Codable {
    let id: String
    let verdict: Verdict
    let detail: String
}

// MARK: - Wall clock

/// `ContinuousClock` — monotonic, immune to wall-clock adjustment. The existing harness uses `Date()`
/// (main.swift:91), which is not.
@inline(__always)
func timed<T>(_ body: () throws -> T) rethrows -> (value: T, ms: Double) {
    let t0 = ContinuousClock.now
    let v = try body()
    return (v, Double((ContinuousClock.now - t0).components.attoseconds) / 1e15
        + Double((ContinuousClock.now - t0).components.seconds) * 1000)
}

extension Duration {
    var ms: Double {
        Double(components.seconds) * 1000 + Double(components.attoseconds) / 1e15
    }
}

// MARK: - Process instruments

enum Proc {
    /// Physical footprint — the number Activity Monitor shows. Unlike `Memory.activeMemory`
    /// (Memory.swift:175, which explicitly EXCLUDES the buffer cache) this counts everything,
    /// including compressed and swapped pages. Never subtract one from the other: on unified
    /// memory the two ledgers overlap in an unspecified way.
    static func physFootprint() -> Int? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ip in
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), ip, &count)
            }
        }
        return kr == KERN_SUCCESS ? Int(info.phys_footprint) : nil
    }

    struct DiskIO: Sendable, Codable {
        var bytesRead: Int
        var bytesWritten: Int
        var pageins: Int
        static let zero = DiskIO(bytesRead: 0, bytesWritten: 0, pageins: 0)
        func delta(_ o: DiskIO) -> DiskIO {
            DiskIO(bytesRead: o.bytesRead - bytesRead,
                   bytesWritten: o.bytesWritten - bytesWritten,
                   pageins: o.pageins - pageins)
        }
    }

    /// DEVICE-level I/O for this process. This is a *secondary*, deliberately-labelled instrument:
    /// because nothing in mlx-swift's writer fsyncs (io/load.h has no fsync/F_FULLFSYNC/F_NOCACHE),
    /// writes land in the unified page cache and re-reads of a just-written file are served from
    /// RAM. The DIVERGENCE between logical bytes and these counters is itself a finding — it says
    /// how much of the "disk" cost the machine was never actually paying.
    static func diskIO() -> DiskIO? {
        var info = rusage_info_v4()
        let rc = withUnsafeMutablePointer(to: &info) { p -> Int32 in
            p.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { ip in
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, ip)
            }
        }
        guard rc == 0 else { return nil }
        return DiskIO(bytesRead: Int(info.ri_diskio_bytesread),
                      bytesWritten: Int(info.ri_diskio_byteswritten),
                      pageins: Int(info.ri_pageins))
    }

    /// Force a file's dirty pages all the way to the device. `write()` alone only reaches the page
    /// cache, so any save timing NOT followed by this is memcpy speed, not disk speed.
    static func fullFsync(_ url: URL) -> Double? {
        let fd = open(url.path, O_RDONLY)
        guard fd >= 0 else { return nil }
        defer { close(fd) }
        let t0 = ContinuousClock.now
        guard fcntl(fd, F_FULLFSYNC) != -1 else { return nil }
        return (ContinuousClock.now - t0).ms
    }

    /// Compressor / pageout activity. Non-zero deltas mean the run was under memory pressure and
    /// every timing and memory number from that phase is `invalid`.
    static func vmPressure() -> (compressions: Int, pageouts: Int)? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) { p in
            p.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { ip in
                host_statistics64(mach_host_self(), HOST_VM_INFO64, ip, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return (Int(stats.compressions), Int(stats.pageouts))
    }

    static var thermalNominal: Bool { ProcessInfo.processInfo.thermalState == .nominal }

    static func freeDiskBytes(at url: URL) -> Int? {
        (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
            .volumeAvailableCapacityForImportantUsage.map(Int.init)
    }
}

// MARK: - MLX memory

struct MemShot: Sendable, Codable {
    var active: Int
    var cache: Int
    var peak: Int
    var footprint: Int

    static func now() -> MemShot {
        let s = Memory.snapshot()
        return MemShot(active: s.activeMemory, cache: s.cacheMemory,
                       peak: s.peakMemory, footprint: Proc.physFootprint() ?? 0)
    }

    /// Reset MLX's peak high-water mark so the NEXT phase's peak is that phase's own.
    /// The setter ignores its value and calls `mlx_reset_peak_memory` (Memory.swift:209-212).
    static func resetPeak() { Memory.peakMemory = 0 }

    func delta(_ o: MemShot) -> MemShot {
        MemShot(active: o.active - active, cache: o.cache - cache,
                peak: o.peak - peak, footprint: o.footprint - footprint)
    }
}

// MARK: - Store log sink probe

/// Classified events parsed from `PromptCacheStore`'s diagnostic sink. The sink already reports
/// exact byte counts and tier discrimination (PromptCacheStore.swift:75, :129, :145, :148, :154);
/// the existing harness discards it by never passing `log:` (main.swift:154, :206, :257).
enum StoreEvent: Sendable, Equatable {
    case saved(bytes: Int, file: String)
    case loaded(file: String)
    case skip(String)             // "record: SKIP —" : a hybrid/pure-SSM refusal, NOT a data point
    case nothingStored(String)    // "record: NOTHING STORED"
    case other

    /// Keyed on the two-token prefix, never a bare index: `reuse: catalog matched 512 tokens in f`
    /// also has a number at index 2 and must NOT parse as a save.
    static func parse(_ line: String) -> StoreEvent {
        let t = line.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
        guard t.count >= 2 else { return .other }
        switch (t[0], t[1]) {
        case ("record:", "saved"):
            guard t.count >= 3, let b = Int(t[2]) else { return .other }
            return .saved(bytes: b, file: t.count >= 6 ? t[5] : "?")
        case ("reuse:", "loaded"):                                  // legacy whole-snapshot load
            guard t.count >= 3 else { return .other }
            return .loaded(file: t[2])
        case ("reuse:", "reassembled"):                             // delta chain reassembly = one load
            return .loaded(file: "chain")
        case ("record:", "SKIP"): return .skip(line)
        case ("record:", "NOTHING"): return .nothingStored(line)
        default: return .other
        }
    }
}

/// Sendable collector for one workload's store. Each workload gets its OWN store and OWN probe —
/// a shared store would attribute an interactive turn's `reuse: loaded` to the warm arm's byte
/// accounting and silently corrupt every ratio.
final class IOProbe: Sendable {
    private let state = Mutex<([StoreEvent], [String])>(([], []))

    /// Captures `self` (a `Sendable` class reference), not the `Mutex` — `Mutex` is non-copyable
    /// and cannot be pulled into a capture list.
    var sink: @Sendable (String) -> Void {
        { [self] line in note(line) }
    }

    private func note(_ line: String) {
        state.withLock { $0.0.append(StoreEvent.parse(line)); $0.1.append(line) }
    }

    var events: [StoreEvent] { state.withLock { $0.0 } }
    var lines: [String] { state.withLock { $0.1 } }
    func reset() { state.withLock { $0 = ([], []) } }

    var savedBytes: Int { events.reduce(0) { if case let .saved(b, _) = $1 { $0 + b } else { $0 } } }
    var saveCount: Int { events.reduce(0) { if case .saved = $1 { $0 + 1 } else { $0 } } }
    var loadCount: Int { events.reduce(0) { if case .loaded = $1 { $0 + 1 } else { $0 } } }
    var loadedFiles: [String] { events.compactMap { if case let .loaded(f) = $0 { f } else { nil } } }
    var refusals: [String] {
        events.compactMap {
            switch $0 { case let .skip(s): s; case let .nothingStored(s): s; default: nil }
        }
    }

    /// Aborts on log-format drift. The parser depends on format strings in PromptCacheStore; if
    /// those change, every byte number silently becomes zero and every gate goes green on nothing.
    static func selfTest() throws {
        func want(_ line: String, _ expect: StoreEvent) throws {
            let got = StoreEvent.parse(line)
            guard got == expect else {
                throw BenchError.logFormatDrift("parsed \(got) from: \(line)")
            }
        }
        try want("record: saved 12345 bytes → snap-abc.safetensors",
                 .saved(bytes: 12345, file: "snap-abc.safetensors"))
        try want("reuse: loaded snap-abc.safetensors — snapshot offset 512",
                 .loaded(file: "snap-abc.safetensors"))
        // Negative case: a number at index 2 that is NOT a byte count.
        try want("reuse: catalog matched 512 tokens in snap-abc.safetensors", .other)
        try want("reuse: MISS — no matching prefix in catalog (x)", .other)
        guard case .skip = StoreEvent.parse("record: SKIP — hybrid cache, can't record") else {
            throw BenchError.logFormatDrift("SKIP not recognised")
        }
        guard case .nothingStored = StoreEvent.parse("record: NOTHING STORED — planRecord nil") else {
            throw BenchError.logFormatDrift("NOTHING STORED not recognised")
        }
    }
}

enum BenchError: Error, CustomStringConvertible {
    case logFormatDrift(String)
    case silentRefusal(String)
    case armDidNoWork(String)
    case outOfDisk(String)
    case shapeUnexpected(String)

    var description: String {
        switch self {
        case let .logFormatDrift(s): "log-format drift — parser is stale: \(s)"
        case let .silentRefusal(s): "store refused to record but reported success: \(s)"
        case let .armDidNoWork(s): "arm produced no work — a green here would be meaningless: \(s)"
        case let .outOfDisk(s): "refusing to continue: \(s)"
        case let .shapeUnexpected(s): "cache shape not as predicted: \(s)"
        }
    }
}

// MARK: - Scratch root with real cleanup

/// One parent directory, removed by `atexit` AND by signal handlers. `defer` is not enough: the
/// harness deliberately has abort paths that skip it, and the existing harness leaks every store
/// dir it makes (main.swift:153, :204, :256 — no `defer`, no `removeItem`).
enum Scratch {
    nonisolated(unsafe) private static var root: URL?

    static func begin() -> URL {
        let parent = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxpc-bench", isDirectory: true)
        // Sweep stale runs from earlier aborted invocations.
        if let old = try? FileManager.default.contentsOfDirectory(at: parent,
                                                                  includingPropertiesForKeys: nil) {
            for d in old where d.lastPathComponent.hasPrefix("run-") {
                try? FileManager.default.removeItem(at: d)
            }
        }
        let dir = parent.appendingPathComponent("run-\(getpid())", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        root = dir
        atexit { Scratch.cleanup() }
        for sig in [SIGINT, SIGTERM, SIGHUP] {
            signal(sig) { _ in Scratch.cleanup(); exit(130) }
        }
        return dir
    }

    static func cleanup() {
        if let r = root { try? FileManager.default.removeItem(at: r); root = nil }
    }

    static func store(_ tag: String) -> URL {
        let d = (root ?? FileManager.default.temporaryDirectory)
            .appendingPathComponent("\(tag)-\(UUID().uuidString.prefix(8))", isDirectory: true)
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }
}

// MARK: - Ballast

/// Squeezes free memory so a large-RAM host behaves like a smaller one.
///
/// Every re-read in the baseline arm is currently served from the page cache — device reads are
/// ~0 on a 137 GB box. That makes the read half of the disease invisible. The question residency's
/// value on a 32–64 GB machine turns on is whether those re-reads still hit RAM when a model's
/// weights and a multi-gigabyte snapshot are competing for it. Rather than argue about it, allocate
/// and fault in enough anonymous memory that the page cache has to give ground.
enum Ballast {
    nonisolated(unsafe) private static var block: UnsafeMutableRawPointer?
    nonisolated(unsafe) private static var size = 0

    /// Hold `physicalMemory - targetFreeBytes`, clamped so we never take more than 70% of RAM.
    /// Returns the bytes actually held.
    @discardableResult
    static func squeeze(toFreeBytes target: Int) -> Int {
        let physical = Int(ProcessInfo.processInfo.physicalMemory)
        let want = min(max(0, physical - target), Int(Double(physical) * 0.70))
        guard want > 0 else { return 0 }
        guard let p = malloc(want) else { return 0 }
        // Fault every page in — an untouched allocation costs nothing and squeezes nothing.
        let stride = 16384
        var i = 0
        while i < want { p.advanced(by: i).storeBytes(of: UInt8(1), as: UInt8.self); i += stride }
        block = p
        size = want
        return want
    }

    static func release() {
        if let b = block { free(b); block = nil; size = 0 }
    }

    static var heldBytes: Int { size }
}

// MARK: - Formatting

func fmtBytes(_ b: Int) -> String {
    let x = Double(b)
    if x >= 1e9 { return String(format: "%.2f GB", x / 1e9) }
    if x >= 1e6 { return String(format: "%.1f MB", x / 1e6) }
    if x >= 1e3 { return String(format: "%.1f kB", x / 1e3) }
    return "\(b) B"
}

func fmtMs(_ m: Double) -> String {
    m >= 1000 ? String(format: "%.2f s", m / 1000) : String(format: "%.0f ms", m)
}

func pad(_ s: String, _ w: Int) -> String {
    s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
}

func median(_ xs: [Double]) -> Double {
    guard !xs.isEmpty else { return 0 }
    let s = xs.sorted()
    return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
}

func iqr(_ xs: [Double]) -> Double {
    guard xs.count >= 4 else { return 0 }
    let s = xs.sorted()
    return s[(s.count * 3) / 4] - s[s.count / 4]
}
