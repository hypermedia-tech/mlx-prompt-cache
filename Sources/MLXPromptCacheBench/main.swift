import Foundation
import HuggingFace
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXPromptCache
import Synchronization
import Tokenizers

// ═══════════════════════════════════════════════════════════════════════════════════════════════
// MLXPromptCacheBench — proves or refutes the warm-loop I/O hypothesis on every model.
//
//   H1  bytes(N) = A·N + M, with A and M PREDICTED from config.json before any weights load.
//   H2  the current warm-with-pause loop is quadratic in both directions.
//   H3  holding the live cache across pauses makes total I/O independent of the resume count.
//   H4  residency is lossless — including generating straight off the held cache.
//   H5  what a persist actually costs: memory peak, wall clock, and the process-global eval stall.
//
// Deliberate design decisions, each answering a specific way this could have lied:
//   · A and M are analytic, not fitted. Comparing a fit over file sizes against a fit over the
//     same file sizes is an arithmetic identity, not a measurement.
//   · File sizes are reported as "logical" bytes and are NEVER the headline. The primary
//     observables are wall clock, forced-fsync time, and rusage device counters.
//   · Nothing in mlx-swift's writer fsyncs, so a re-read seconds later is served from the page
//     cache. The divergence between logical and device bytes is reported as its own finding.
//   · Both arms run with prefillStepSize == blockSize so they issue the same GPU op sequence;
//     otherwise H4 would be testing bf16 numerics rather than residency.
//   · Every arm asserts it did work. "No I/O" must never be indistinguishable from "no work".
//   · Every AsyncStream is drained inside the perform that created it (see `generateInside`).
// ═══════════════════════════════════════════════════════════════════════════════════════════════

// MARK: - CLI

struct Options {
    var models: [ModelSpec] = ModelSpec.all
    var tokens = 16_384
    var resumes = 8
    var jsonPath: String?
    var skipCanary = false
    /// Simulate a smaller machine by holding anonymous memory, so the page cache cannot absorb
    /// every snapshot. 0 = no ballast.
    var freeRamBytes = 0
    /// A config.json path or HF model id — print KV-cache economics and exit, no weights loaded.
    var predictArgs: [String] = []

    static func parse() -> Options {
        var o = Options()
        var it = CommandLine.arguments.dropFirst().makeIterator()
        while let a = it.next() {
            switch a {
            case "--model":
                if let v = it.next(), let m = ModelSpec.named(v) { o.models = [m] }
                else { print("unknown --model; known: \(ModelSpec.all.map(\.short))"); exit(2) }
            case "--tokens": o.tokens = it.next().flatMap { Int($0) } ?? o.tokens
            case "--resumes": o.resumes = it.next().flatMap { Int($0) } ?? o.resumes
            case "--json": o.jsonPath = it.next()
            case "--no-canary": o.skipCanary = true
            case "--free-ram":
                o.freeRamBytes = (it.next().flatMap { Int($0) } ?? 0) * 1_000_000_000
            case "--predict":
                while let v = it.next() { o.predictArgs.append(v) }   // consumes the rest
            case "--help", "-h":
                print("""
                MLXPromptCacheBench — warm-loop I/O proof harness

                  --model <35B|9B|1.7B|GLM-32B>   run one model (default: all, one at a time)
                  --tokens N              prefix length to warm (default 16384)
                                          use 183296 to reproduce the production measurement
                  --resumes R             number of pause/resume rounds (default 8)
                  --json <path>           write the machine-readable artifact
                  --no-canary             skip the process-global eval-lock probe
                  --free-ram G            simulate a G-gigabyte machine by holding ballast memory
                  --predict <cfg|id>...   print KV-cache economics from config.json — NO weights
                                          loaded, NO GPU. Screen a candidate before pulling it.
                                          Accepts config.json paths or HF model ids (resolved from
                                          the local cache). Consumes all trailing arguments.
                """)
                exit(0)
            default: print("unknown argument \(a) — try --help"); exit(2)
            }
        }
        return o
    }
}

let opts = Options.parse()
let blockSize = 256

// MARK: - Predict mode (config-only, no MLX)

if !opts.predictArgs.isEmpty {
    func fmtG(_ b: Int) -> String { String(format: "%.2f GB", Double(b) / 1e9) }
    print("═══ KV-cache economics (from config.json — no weights loaded) ═══")
    print("KV dtype assumed fp16/bf16 = 2 bytes/element (weight quant does NOT change this).\n")
    for arg in opts.predictArgs {
        guard let p = ConfigPrediction.load(arg) else {
            print("  ✗ \(arg): no readable config.json (not a file, and not in the HF cache)\n")
            continue
        }
        let split = p.isHybrid
            ? "\(p.attentionLayers) attention + \(p.linearLayers) linear of \(p.totalLayers)"
            : "\(p.totalLayers) attention (dense)"
        print("  \(arg)  [\(p.modelType)]")
        print("    layers: \(split) · kv_heads \(p.kvHeads) · head_dim \(p.headDim) · context \(p.maxContext)"
            + (p.slidingWindow.map { " · SWA window \($0)" } ?? ""))
        print("    A = \(p.perTokenBytes) B/tok (\(String(format: "%.1f", Double(p.perTokenBytes) / 1024)) KiB/tok)"
            + "   M = \(p.fixedBytes > 0 ? fmtG(p.fixedBytes) : "0")\(p.isHybrid ? "  (SSM term; conv adds a few %)" : "")")
        let ladder = [8_192, 32_768, 131_072, 262_144, 524_288].filter { $0 <= max(p.maxContext, 8_192) }
        let cells = ladder.map { "\($0 / 1024)K→\(fmtG(p.snapshotBytes($0)))" }.joined(separator: " · ")
        print("    snapshot: \(cells)")
        if p.slidingWindow != nil {
            print("    note: sliding-window layers use RotatingKVCache in MLXPromptCache — cold-only (no RAM tier, no sub-slice)")
        } else if p.isHybrid {
            print("    note: hybrid — RAM tier stays cold-only; reusable only at a captured boundary")
        } else {
            print("    note: dense full-attention — RAM hot tier eligible, freely sub-sliceable")
        }
        print("")
    }
    exit(0)
}

// MARK: - Report model

struct ModelReport: Codable {
    var model: String
    var predictedA: Int
    var predictedMLowerBound: Int
    var measuredA: Double = 0
    var measuredM: Double = 0
    var fitPoints: [[Int]] = []
    var shape: String = ""
    var statusQuo: ArmResult?
    var residency: ArmResult?
    var gates: [GateResult] = []
    var persistRecordMs: [Double] = []
    var persistFsyncMs: [Double] = []
    var persistPeakDelta: Int = 0
    var evalStallMaxMs: Double = 0
    var statusQuoPeakBytes: Int = 0
    var residencyPeakBytes: Int = 0
    var ballastBytes: Int = 0
    var heldVsColdDiff: StateDiff?
    var contentionWaitMs: [Double] = []
    var notes: [String] = []
}

struct BenchReport: Codable {
    var host: String
    var ramBytes: Int
    var freeDiskBytes: Int
    var tokens: Int
    var resumes: Int
    var models: [ModelReport] = []
}

// MARK: - Helpers

/// Every `AsyncStream` from `generateTokens` is fully drained INSIDE the `perform` that made it,
/// and we synchronize before returning.
///
/// This is a RULE, not a style preference. `generateTokens` does not run in your `perform`: it
/// wraps the `[KVCache]` in a `SendableBox` and launches an unstructured detached `Task`
/// (Evaluate.swift:1825-1829). `AsyncStream<TokenGeneration>` IS `Sendable`, so it satisfies
/// `perform`'s `R: Sendable` and the compiler will happily let you return it and drain it outside —
/// putting two live caches on MLX at once, which is the exact hazard the serialised domain exists
/// to prevent. The existing harness obeys this by convention only (main.swift:192-199, :290).
func generateInside(_ ctx: ModelContext, tokens: [Int], cache: [KVCache],
                    params: GenerateParameters) async throws -> [Int] {
    var ids: [Int] = []
    for await g in try generateTokens(input: LMInput(tokens: MLXArray(tokens)),
                                      cache: cache, parameters: params, context: ctx) {
        if case let .token(t) = g { ids.append(t) }
    }
    Stream.gpu.synchronize()
    return ids
}

func logLine(_ i: Int) -> String {
    func oct(_ n: Int) -> Int { (n % 254) + 1 }
    let ports = [22, 80, 443, 445, 3389, 8080, 53, 25]
    let procs = ["powershell.exe", "cmd.exe", "svchost.exe", "rundll32.exe", "wscript.exe", "mshta.exe"]
    func p2(_ n: Int) -> String { n < 10 ? "0\(n)" : "\(n)" }
    return "[2026-06-16T\(p2((i / 3600) % 24)):\(p2((i / 60) % 60)):\(p2(i % 60))Z] "
        + "host=10.\(oct(i)).\(oct(i * 7)).\(oct(i * 13)) pid=\(1000 + (i * 131) % 60000) "
        + "proc=\(procs[(i * 5) % procs.count]) dst_port=\(ports[(i * 3) % ports.count]) "
        + "bytes_out=\((i * 977) % 1_000_000) beacon_s=\(30 + (i * 17) % 600) verdict=review\n"
}

func makeParams(maxTokens: Int = 32) -> GenerateParameters {
    var p = GenerateParameters(maxTokens: maxTokens, temperature: 0)   // greedy ⇒ deterministic
    p.prefillStepSize = blockSize          // pin chunking so both arms issue the same GPU ops
    return p
}

func gate(_ id: String, _ ok: Bool, _ detail: String, invalid: Bool = false) -> GateResult {
    GateResult(id: id, verdict: invalid ? .invalid : (ok ? .pass : .fail), detail: detail)
}

// MARK: - Startup self-tests

print("═══ MLXPromptCacheBench ═══")
do { try IOProbe.selfTest() } catch {
    print("ABORT: \(error)")
    exit(3)
}
let scratchRoot = Scratch.begin()
let ram = Int(ProcessInfo.processInfo.physicalMemory)
let freeDisk = Proc.freeDiskBytes(at: scratchRoot) ?? 0
print("host: \(fmtBytes(ram)) RAM · \(fmtBytes(freeDisk)) free disk · thermal \(ProcessInfo.processInfo.thermalState == .nominal ? "nominal" : "NON-NOMINAL")")
print("plan: T=\(opts.tokens) tokens · R=\(opts.resumes) resumes · block=\(blockSize) · n=1 per cell")
if opts.freeRamBytes > 0 {
    print("ballast: squeezing to ~\(fmtBytes(opts.freeRamBytes)) free …")
    let held = Ballast.squeeze(toFreeBytes: opts.freeRamBytes)
    print("ballast: holding \(fmtBytes(held)) — page cache must now compete for what is left")
}
atexit { Ballast.release() }
print("log-sink parser self-test ✅\n")

var report = BenchReport(host: ProcessInfo.processInfo.hostName, ramBytes: ram,
                         freeDiskBytes: freeDisk, tokens: opts.tokens, resumes: opts.resumes)

// ═══════════════════════════════════════════════════════════════════════════════════════════════

for spec in opts.models {
    print("\n╔═══ \(spec.short) — \(spec.id)")
    var rep = ModelReport(model: spec.short, predictedA: spec.predictedA,
                          predictedMLowerBound: spec.predictedMLowerBound)

    // Disk guard: the largest single snapshot this model will write, times a safety factor.
    let worstSnapshot = spec.predictedBytes(opts.tokens)
    if freeDisk < worstSnapshot * 4 {
        print("  SKIP — needs ~\(fmtBytes(worstSnapshot * 4)) free, have \(fmtBytes(freeDisk))")
        rep.gates.append(gate("disk", false, "insufficient free disk", invalid: true))
        report.models.append(rep)
        continue
    }

    // Context clamp — the 1.7B's window is 40,960; overrunning it makes every arm degrade
    // identically and H4 would pass on out-of-context garbage.
    let tTarget = min(opts.tokens, Int(Double(spec.maxContext) * 0.8))
    if tTarget < opts.tokens {
        print("  note: T clamped \(opts.tokens) → \(tTarget) (0.8 × context \(spec.maxContext))")
        rep.notes.append("T clamped to \(tTarget) by context window")
    }

    print("  predicted A = \(spec.predictedA) B/tok (\(spec.attentionLayers) attn × 2 × \(spec.kvHeads) kvHeads × \(spec.headDim) headDim × 2 B)")
    print("  predicted M ≥ \(fmtBytes(spec.predictedMLowerBound)) (\(spec.linearLayers) linear × \(spec.valueHeads)×\(spec.keyHeadDim)×\(spec.valueHeadDim) × 4 B)")

    // ── Load ────────────────────────────────────────────────────────────────────────────────
    // One model per iteration; the container is dropped at the end of the loop body so its
    // weights are freed before the next load (ModelFactory does not memoize).
    print("  loading …")
    let mc = try await LLMModelFactory.shared.loadContainer(
        from: #hubDownloader(),
        using: #huggingFaceTokenizerLoader(),
        configuration: ModelConfiguration(id: spec.id))

    let sig = CacheSignature(modelId: spec.id, kvDType: "bf16", kvBits: nil, buildVersion: "bench-1")
    let params = makeParams()

    let corpus = (0 ..< max(1400, tTarget / 8)).map(logLine).joined()
    let corpusTokens = await mc.encode(corpus)
    guard corpusTokens.count > tTarget else {
        print("  SKIP — corpus \(corpusTokens.count) tokens < T \(tTarget)")
        rep.gates.append(gate("corpus", false, "corpus too small", invalid: true))
        report.models.append(rep)
        continue
    }
    let T = (tTarget / blockSize) * blockSize

    let tokens = Array(corpusTokens.prefix(T))
    let question = await mc.encode("\n\nQ: one sentence — most suspicious behaviour?\nA:")

    // Not every phase needs production scale. G2/G3 measure I/O volume, which is the whole point of
    // a large T; the isolated persist measurement needs it too, because persist cost scales with
    // snapshot size. But G4 is a correctness gate, G5's canary just needs SOME MLX work running to
    // sample the lock, and G6 is a boolean about whether a tier populates — running those at 183k
    // costs four extra full-length prefills and proves nothing extra. Pin them.
    let probeBoundary = min((T - 1) / blockSize * blockSize, (16_384 / blockSize) * blockSize)
    let probePrompt = Array(tokens.prefix(probeBoundary + 1))
    let probeRounds = 4
    let probeChunks = max(1, probeBoundary / (probeRounds * blockSize))
    if probeBoundary < (T - 1) / blockSize * blockSize {
        print("  scale: G2/G3/G5-persist at \((T - 1) / blockSize * blockSize) tokens · G4/G5-canary/G6 at \(probeBoundary) (correctness + boolean gates, scale-invariant)")
    }

    // Warm up Metal so the first timing is not one-time init.
    _ = try await mc.perform { ctx in
        try await generateInside(ctx, tokens: Array(tokens.prefix(8)),
                                 cache: makePromptCache(model: ctx.model, parameters: params),
                                 params: makeParams(maxTokens: 1))
    }

    // ── G0 · shape probe (diagnostic, not a gate) ───────────────────────────────────────────
    let shape = await mc.perform { ctx -> String in
        cacheShape(makePromptCache(model: ctx.model, parameters: params))
    }
    rep.shape = shape
    print("  cache shape: \(shape)")
    let hybridByShape = shape.contains("MambaCache") || shape.contains("ArraysCache")
    rep.gates.append(gate("G0.shape", hybridByShape == spec.isHybrid,
                          "shape says hybrid=\(hybridByShape), config says \(spec.isHybrid)"))

    // ── G1 · byte model: measured A, M vs analytic prediction ───────────────────────────────
    print("\n  ── G1 byte model ──")
    var pts: [(Int, Int)] = []
    let fitT = probeBoundary            // the fit is exact at probe scale; see the full-scale check below
    for frac in [4, 2, 1] {
        let n = (fitT / frac / blockSize) * blockSize
        guard n >= blockSize * 2 else { continue }
        let dir = Scratch.store("g1-\(spec.short)-\(n)")
        let probe = IOProbe()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 40, signature: sig,
                                         blockSize: blockSize, hotBudgetBytes: 0, log: probe.sink)
        let coord = PromptCacheCoordinator(store: store)
        _ = await mc.perform { ctx in
            coord.warm(promptTokens: Array(tokens.prefix(n + 1)), model: ctx.model, parameters: params)
        }
        // A warm that reports success but silently refused to record would make every later gate
        // green on nothing. `record` swallows hybridNotAtBoundary/noSliceableLayer internally
        // (PromptCacheStore.swift:141-149) and `warm` discards the throw with try? (:148).
        guard probe.refusals.isEmpty, probe.saveCount == 1 else {
            throw BenchError.silentRefusal("n=\(n) saves=\(probe.saveCount) refusals=\(probe.refusals)")
        }
        let bytes = Census.of(dir).total
        // x must be the count `warm` ACTUALLY recorded, not the count we asked for. `warm` snapshots
        // the last FULL block of the prompt — (len-1)/blockSize*blockSize — and `prefix(n+1)` silently
        // clamps when n+1 exceeds the corpus. Labelling the point with the requested n instead of the
        // recorded one biases the two-point fit by exactly (n1'-n0)/(n1-n0).
        let promptLen = min(n + 1, tokens.count)
        let recorded = (promptLen - 1) / blockSize * blockSize
        pts.append((recorded, bytes))
        try? FileManager.default.removeItem(at: dir)
        print("    n=\(pad("\(recorded)", 7)) bytes=\(pad(fmtBytes(bytes), 10)) → \(String(format: "%.0f", Double(bytes) / Double(recorded))) B/tok")
    }
    rep.fitPoints = pts.map { [$0.0, $0.1] }
    if pts.count >= 2 {
        // Two-point solve on the extremes — no least-squares theatre.
        let (n0, b0) = pts.first!, (n1, b1) = pts.last!
        let a = Double(b1 - b0) / Double(n1 - n0)
        let m = Double(b0) - a * Double(n0)
        rep.measuredA = a
        rep.measuredM = m
        let aErr = abs(a - Double(spec.predictedA)) / Double(spec.predictedA)
        print(String(format: "    measured A = %.0f B/tok (predicted %d, %+.2f%%)", a, spec.predictedA, aErr * 100))
        print("    measured M = \(fmtBytes(Int(m))) (predicted ≥ \(fmtBytes(spec.predictedMLowerBound)))")
        rep.gates.append(gate("G1.A", aErr < 0.02,
                              String(format: "A within %.2f%% of analytic prediction", aErr * 100)))
        let mOK = spec.isHybrid
            ? (m >= Double(spec.predictedMLowerBound) * 0.95 && m <= Double(spec.predictedMLowerBound) * 1.25)
            : (abs(m) < Double(spec.predictedA) * Double(blockSize))   // pure attention ⇒ M ≈ 0
        rep.gates.append(gate("G1.M", mOK,
                              "measured M \(fmtBytes(Int(m))) vs predicted ≥ \(fmtBytes(spec.predictedMLowerBound))"))
    }

    // ── Shared arm plumbing ─────────────────────────────────────────────────────────────────
    let chunksPerRound = max(1, T / (opts.resumes * blockSize))
    let boundary = (T - 1) / blockSize * blockSize


    /// Status-quo arm: today's `warm`, called once per resume. Each call re-enters `store.reuse`
    /// (loading the whole previous snapshot) and ends in a whole-prefix `record`.
    func runStatusQuo() async throws -> ArmResult {
        let dir = Scratch.store("sq-\(spec.short)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let probe = IOProbe()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 40, signature: sig,
                                         blockSize: blockSize, hotBudgetBytes: 0, log: probe.sink)
        let coord = PromptCacheCoordinator(store: store)
        var arm = ArmResult(name: "status-quo", samples: [])
        var round = 0
        var reached = 0

        while reached < boundary && round < opts.resumes * 3 {
            round += 1
            probe.reset()
            let before = Census.of(dir)
            let io0 = Proc.diskIO() ?? .zero
            let vm0 = Proc.vmPressure()
            MemShot.resetPeak()
            let m0 = MemShot.now()

            let t0 = ContinuousClock.now
            // Counters live INSIDE the perform closure: its action is @Sendable, so capturing a
            // mutable local from the enclosing scope would not compile under Swift 6.
            let got: Int = await mc.perform { ctx in
                var chunks = 0
                let outcome = coord.warm(promptTokens: tokens, model: ctx.model, parameters: params,
                                         shouldPause: { chunks += 1; return chunks >= chunksPerRound })
                switch outcome {
                case let .paused(c): return c
                case let .complete(c, _): return c
                case .uncacheable: return -1
                }
            }
            let wall = (ContinuousClock.now - t0).ms
            let m1 = MemShot.now()
            let io1 = Proc.diskIO() ?? .zero
            let vm1 = Proc.vmPressure()

            guard got > reached else {
                throw BenchError.armDidNoWork("status-quo round \(round): \(reached) → \(got)")
            }
            guard probe.refusals.isEmpty else {
                throw BenchError.silentRefusal("status-quo round \(round): \(probe.refusals)")
            }
            let after = Census.of(dir)
            // The snapshot `reuse` actually read, sized before the call overwrote the directory.
            let readBytes = probe.loadedFiles.reduce(0) { $0 + before.size($1) }
            // Force the write to the device and time it separately — `write()` alone only reaches
            // the page cache, so any save timing without this is memcpy speed, not disk speed.
            var fsyncMs = 0.0
            for f in after.newFiles(vs: before) {
                fsyncMs += Proc.fullFsync(dir.appendingPathComponent(f)) ?? 0
            }
            let pressured = !Proc.thermalNominal
                || (vm0 != nil && vm1 != nil && (vm1!.pageouts > vm0!.pageouts))

            arm.samples.append(ResumeSample(
                round: round, reachedTokens: got,
                logicalWriteBytes: probe.savedBytes, logicalReadBytes: readBytes,
                saveCount: probe.saveCount, loadCount: probe.loadCount,
                wallMs: wall, fsyncMs: fsyncMs,
                deviceReadBytes: max(0, io1.bytesRead - io0.bytesRead),
                deviceWriteBytes: max(0, io1.bytesWritten - io0.bytesWritten),
                memPeakDelta: m1.peak, memActiveDelta: m1.active - m0.active,
                footprint: m1.footprint, pressured: pressured))
            reached = got
        }
        return arm
    }

    /// Residency arm — drives the SHIPPED API (`WarmStore` + `coordinator.warm(_:id:…)`), not a
    /// harness-side simulation. That distinction is the whole point: this measures the library you
    /// would publish, so a green gate here is evidence about MLXPromptCache rather than about the
    /// benchmark.
    ///
    /// `generateFromHeld` stops one round short of the boundary, leaving the warm PAUSED with its
    /// cache resident, and generates directly off it. That is the state residency actually creates,
    /// and it is the only arm that exercises the held cache with no disk round trip at all.
    func runResidency(persist: WarmPersistence = .onCompletion,
                      generateFromHeld: Bool = false,
                      prompt: [Int]? = nil,
                      upTo: Int? = nil,
                      chunks: Int? = nil) async throws -> (arm: ArmResult, ids: [Int]) {
        let toks = prompt ?? tokens
        let bnd = upTo ?? boundary
        let cpr = chunks ?? chunksPerRound
        let dir = Scratch.store("res-\(spec.short)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let probe = IOProbe()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 40, signature: sig,
                                         blockSize: blockSize, hotBudgetBytes: 0, log: probe.sink)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        var arm = ArmResult(name: "residency", samples: [])
        var reached = 0
        var round = 0
        // Stop one full round short so the cache is still held when we generate off it.
        let target = generateFromHeld ? max(blockSize, bnd - cpr * blockSize) : bnd

        while reached < target && round < opts.resumes * 4 {
            round += 1
            probe.reset()
            let before = Census.of(dir)
            let io0 = Proc.diskIO() ?? .zero
            MemShot.resetPeak()
            let m0 = MemShot.now()

            let t0 = ContinuousClock.now
            let got: Int = await mc.perform { ctx in
                var fired = 0
                let outcome = coord.warm(warms, id: id, promptTokens: toks, model: ctx.model,
                                         parameters: params, persist: persist,
                                         shouldPause: { fired += 1; return fired >= cpr })
                switch outcome {
                case let .paused(c): return c
                case let .complete(c, _): return c
                case .uncacheable: return -1
                }
            }
            let wall = (ContinuousClock.now - t0).ms
            let m1 = MemShot.now()
            let io1 = Proc.diskIO() ?? .zero

            // Anti-tautology: "performed no I/O" must never be indistinguishable from "did nothing".
            guard got > reached else {
                throw BenchError.armDidNoWork("residency round \(round): \(reached) → \(got)")
            }
            guard probe.refusals.isEmpty else {
                throw BenchError.silentRefusal("residency round \(round): \(probe.refusals)")
            }
            let after = Census.of(dir)
            var fs = 0.0
            for f in after.newFiles(vs: before) {
                fs += Proc.fullFsync(dir.appendingPathComponent(f)) ?? 0
            }
            arm.samples.append(ResumeSample(
                round: round, reachedTokens: got,
                logicalWriteBytes: probe.savedBytes,
                logicalReadBytes: probe.loadedFiles.reduce(0) { $0 + before.size($1) },
                saveCount: probe.saveCount, loadCount: probe.loadCount,
                wallMs: wall, fsyncMs: fs,
                deviceReadBytes: max(0, io1.bytesRead - io0.bytesRead),
                deviceWriteBytes: max(0, io1.bytesWritten - io0.bytesWritten),
                memPeakDelta: m1.peak, memActiveDelta: m1.active - m0.active,
                footprint: m1.footprint, pressured: !Proc.thermalNominal))
            reached = got
        }

        var ids: [Int] = []
        if generateFromHeld {
            let held = reached
            // Compare the HELD cache against one prefilled in a single pass to the same offset,
            // using the comparator the sensitivity control validates. Token IDs are a weaker probe:
            // 32 greedy samples taken after the state is already fixed can miss a small divergence
            // that a tensor-level diff cannot.
            rep.heldVsColdDiff = try await mc.perform { ctx -> StateDiff in
                guard let heldCache = coord.heldCache(warms, id: id, model: ctx.model) else {
                    throw BenchError.armDidNoWork("nothing held for state diff")
                }
                let cold = makePromptCache(model: ctx.model, parameters: params)
                try Prefill.chunk(Array(toks[0 ..< held]), into: cold,
                                  model: ctx.model, stepSize: blockSize)
                return diffStates(heldCache, cold)
            }
            let tail = Array(toks[held...]) + question
            ids = try await mc.perform { ctx in
                guard let cache = coord.heldCache(warms, id: id, model: ctx.model) else {
                    throw BenchError.armDidNoWork("nothing held after \(held) tokens")
                }
                guard attnOffset(cache) == held else {
                    throw BenchError.armDidNoWork(
                        "held cache at \(attnOffset(cache).map(String.init) ?? "nil"), expected \(held)")
                }
                return try await generateInside(ctx, tokens: tail, cache: cache, params: params)
            }
            // Abandonment path: persist what is held and free it. Exercises the real finishWarm.
            probe.reset()
            _ = await mc.perform { ctx in
                coord.finishWarm(warms, id: id, model: ctx.model)
            }
            guard warms.isEmpty else {
                throw BenchError.armDidNoWork("finishWarm did not release the held cache")
            }
        } else {
            // A completed warm must have persisted exactly once and released its cache.
            guard probe.saveCount == 1 else {
                throw BenchError.silentRefusal("completed warm saved \(probe.saveCount) times, expected 1")
            }
            guard warms.isEmpty else {
                throw BenchError.armDidNoWork("completed warm left \(warms.residentBytes) bytes resident")
            }
        }
        return (arm, ids)
    }

    // ── G2 · the disease ────────────────────────────────────────────────────────────────────
    print("\n  ── G2 status quo (R=\(opts.resumes)) ──")
    let sq = try await runStatusQuo()
    rep.statusQuo = sq
    for s in sq.samples {
        print("    r\(pad("\(s.round)", 3)) reached=\(pad("\(s.reachedTokens)", 7)) write=\(pad(fmtBytes(s.logicalWriteBytes), 10)) read=\(pad(fmtBytes(s.logicalReadBytes), 10)) wall=\(pad(fmtMs(s.wallMs), 9)) fsync=\(pad(fmtMs(s.fsyncMs), 8)) devW=\(fmtBytes(s.deviceWriteBytes)) devR=\(fmtBytes(s.deviceReadBytes))")
    }
    // With delta writes, each status-quo resume writes only its NEW range — the per-resume `write`
    // column is a delta, not a whole snapshot, so it can't be checked against a whole-snapshot
    // prediction here. The whole-snapshot byte-model check moves to after G3, using the residency
    // arm's single full-prefix snapshot (see `G0.modelAtScale`).
    let writeRounds = sq.samples.filter { $0.logicalWriteBytes > 0 }.count
    print("    delta writes: attention written once (~\(fmtBytes(spec.predictedA * boundary)))"
        + (spec.isHybrid
           ? " + \(writeRounds)× recurrent (~\(fmtBytes(Int(rep.measuredM))) each = \(fmtBytes(writeRounds * Int(rep.measuredM))) — the hybrid per-checkpoint tax, intrinsic)"
           : " ⇒ ideal, no recurrent tax"))
    let sqRoundTrip = sq.totalLogicalWrite + sq.totalLogicalRead
    print("    TOTAL logical write \(fmtBytes(sq.totalLogicalWrite)) + read \(fmtBytes(sq.totalLogicalRead)) = \(fmtBytes(sqRoundTrip))")
    print("    TOTAL device  write \(fmtBytes(sq.totalDeviceWrite)) + read \(fmtBytes(sq.totalDeviceRead))")
    // The disease is a COUNT, not a curve: R loads and R saves of the whole prefix.
    rep.gates.append(gate("G2.loads", sq.samples.dropFirst().allSatisfy { $0.loadCount == 1 },
                          "every resume after the first reloads the whole prefix"))
    rep.gates.append(gate("G2.saves", sq.samples.allSatisfy { $0.saveCount == 1 },
                          "every resume rewrites a whole-prefix snapshot"))
    if sq.totalLogicalRead > 0, sq.totalDeviceRead * 4 < sq.totalLogicalRead {
        let note = "page cache served the re-reads: logical \(fmtBytes(sq.totalLogicalRead)) vs device \(fmtBytes(sq.totalDeviceRead)) — the read half of the disease is largely absorbed by RAM on this host"
        print("    ⚠️  \(note)")
        rep.notes.append(note)
    }

    // ── G3 · the cure ───────────────────────────────────────────────────────────────────────
    print("\n  ── G3 residency ──")
    let (res, _) = try await runResidency()
    rep.residency = res
    for s in res.samples {
        print("    r\(pad("\(s.round)", 3)) reached=\(pad("\(s.reachedTokens)", 7)) write=\(pad(fmtBytes(s.logicalWriteBytes), 10)) read=\(pad(fmtBytes(s.logicalReadBytes), 10)) wall=\(pad(fmtMs(s.wallMs), 9)) fsync=\(pad(fmtMs(s.fsyncMs), 8)) devW=\(fmtBytes(s.deviceWriteBytes))")
    }
    print("    rounds=\(res.samples.count) · logical write \(fmtBytes(res.totalLogicalWrite)) · read \(fmtBytes(res.totalLogicalRead))")
    // Wall clock is the number that survives the page-cache caveat: a re-read served from RAM still
    // costs loadFull + eval, so the time saved is real even where the device bytes were not.
    let wallRatio: Double = res.totalWallMs > 0 ? sq.totalWallMs / res.totalWallMs : 0
    print(String(format: "    WALL: status-quo %@ → residency %@  (%.2f×)  ·  logical writes %@ → %@",
                 fmtMs(sq.totalWallMs), fmtMs(res.totalWallMs), wallRatio,
                 fmtBytes(sq.totalLogicalWrite), fmtBytes(res.totalLogicalWrite)))
    // The per-round wall curve rises in BOTH arms even though residency does no I/O at all — that
    // growth is attention cost scaling with context, not I/O. The I/O cost is the DIFFERENCE, and on
    // a machine with enough free RAM to absorb every write into the page cache it is small.
    rep.statusQuoPeakBytes = sq.peakMem
    rep.residencyPeakBytes = res.peakMem
    rep.ballastBytes = Ballast.heldBytes
    // Residency's cost: what it holds in unified memory for the duration of the warm. The cache
    // grows by concatenation with both buffers briefly live, so peak exceeds the settled size.
    print("    MLX peak: status-quo \(fmtBytes(sq.peakMem)) → residency \(fmtBytes(res.peakMem))"
        + "   (one settled snapshot = \(fmtBytes(spec.predictedBytes(boundary))))")
    print(String(format: "    I/O share of warm time: %.1f%% (%@ of %@)",
                 sq.totalWallMs > 0 ? 100 * (sq.totalWallMs - res.totalWallMs) / sq.totalWallMs : 0,
                 fmtMs(max(0, sq.totalWallMs - res.totalWallMs)), fmtMs(sq.totalWallMs)))
    let prefillRounds = res.samples.dropLast()
    rep.gates.append(gate("G3.zeroRead", prefillRounds.allSatisfy { $0.loadCount == 0 },
                          "residency performs no snapshot reads at all"))
    rep.gates.append(gate("G3.oneWrite", res.samples.filter { $0.saveCount > 0 }.count == 1,
                          "residency writes exactly one snapshot"))
    let ratio = res.totalLogicalWrite + res.totalLogicalRead > 0
        ? Double(sqRoundTrip) / Double(res.totalLogicalWrite + res.totalLogicalRead) : 0
    print(String(format: "    logical round-trip ratio: %.1f×", ratio))
    rep.gates.append(gate("G3.linear",
                          res.totalLogicalWrite <= Int(Double(spec.predictedBytes(boundary)) * 1.05),
                          "residency total write ≈ one snapshot"))
    // Byte model at full scale, off the residency arm's single WHOLE-prefix snapshot (the status-quo
    // arm no longer produces one — it writes deltas). snapshot(boundary) = A·boundary + M.
    if let snap = res.samples.map(\.logicalWriteBytes).max(), snap > 0 {
        let predicted = Double(spec.predictedA * boundary) + rep.measuredM
        let err = abs(Double(snap) - predicted) / predicted
        print(String(format: "    full-scale check: whole snapshot at %d tokens = %@, model predicts %@ (%+.3f%%)",
                     boundary, fmtBytes(snap), fmtBytes(Int(predicted)), err * 100))
        rep.gates.append(gate("G0.modelAtScale", err < 0.01,
                              String(format: "byte model holds at %d tokens (%+.3f%%)", boundary, err * 100)))
    }

    // ── G4 · losslessness ───────────────────────────────────────────────────────────────────
    print("\n  ── G4 losslessness ──")
    let genParams = makeParams(maxTokens: 32)

    let coldIDs = try await mc.perform { ctx in
        try await generateInside(ctx, tokens: probePrompt + question,
                                 cache: makePromptCache(model: ctx.model, parameters: genParams),
                                 params: genParams)
    }
    let (_, heldIDs) = try await runResidency(persist: .never, generateFromHeld: true,
                                              prompt: probePrompt, upTo: probeBoundary, chunks: probeChunks)
    print("    cold=\(coldIDs.prefix(6))… held=\(heldIDs.prefix(6))…")
    let lossless = !coldIDs.isEmpty && coldIDs == heldIDs
    if let d = rep.heldVsColdDiff {
        print("    held vs single-pass cold cache: \(d.summary)")
        rep.gates.append(gate("G4.stateIdentical", d.identical,
                              d.identical
                                  ? "cache held across pause/resume is BIT-IDENTICAL to a single-pass prefill"
                                  : "held cache diverged from single-pass: \(d.summary)"))
    }
    rep.gates.append(gate("G4.heldEqCold", lossless,
                          lossless ? "generation off the HELD cache is token-identical to cold"
                                   : "DIVERGED at index \(zip(coldIDs, heldIDs).enumerated().first { $0.element.0 != $0.element.1 }?.offset ?? min(coldIDs.count, heldIDs.count))"))

    // Two separate controls. The first run conflated them and reported a real invariance as an
    // instrument failure, so they are now distinct:
    //
    //  · SENSITIVITY — diff two caches built over DIFFERENT token content. This must be non-zero,
    //    or the comparator is broken and G4.heldEqCold's pass carries no information. This is the
    //    gate.
    //  · CHUNK INVARIANCE — diff chunk 256 vs 512 over the SAME content. Measured zero on both
    //    hybrids, i.e. the recurrent scan is bit-exact regardless of chunk boundary. That is a
    //    finding about the model, not a defect in the harness, so it is reported and not gated.
    do {
        let ctrlEnd = min(boundary, 16 * blockSize)         // small: two caches are live at once
        let (sensitivity, invariance) = try await mc.perform { ctx -> (StateDiff, StateDiff) in
            func build(chunk: Int, from origin: Int) throws -> [KVCache] {
                let c = makePromptCache(model: ctx.model, parameters: params)
                var o = 0
                while o < ctrlEnd {
                    let e = min(o + chunk, ctrlEnd)
                    try Prefill.chunk(Array(tokens[(origin + o) ..< (origin + e)]), into: c,
                                      model: ctx.model, stepSize: blockSize)
                    o = e
                }
                return c
            }
            let a = try build(chunk: blockSize, from: 0)
            let sens = diffStates(a, try build(chunk: blockSize, from: ctrlEnd))
            let inv = diffStates(a, try build(chunk: blockSize * 2, from: 0))
            return (sens, inv)
        }
        print("    sensitivity  (different content, \(ctrlEnd) tok): \(sensitivity.summary)")
        print("    chunk invariance (\(blockSize) vs \(blockSize * 2)): \(invariance.summary)")
        let sensitive = sensitivity.attnMaxAbs > 0
            || (spec.isHybrid && sensitivity.recurrentMaxAbs > 0)
        rep.gates.append(gate("G4.sensitivity", sensitive,
                              sensitive
                                  ? "comparator detects differing content ⇒ G4.heldEqCold is informative"
                                  : "comparator is blind ⇒ G4.heldEqCold proves nothing"))
        rep.notes.append("chunk invariance (\(blockSize) vs \(blockSize * 2)): \(invariance.summary)")
        rep.notes.append("sensitivity control: \(sensitivity.summary)")
    }

    // ── G5 · what a persist costs ───────────────────────────────────────────────────────────
    print("\n  ── G5 persist cost ──")
    // Isolate the write from the prefill: warm to a pause, then time `finishWarm` ALONE. That is
    // also the real abandonment path, so this measures shipped behaviour rather than a proxy.
    do {
        let dir = Scratch.store("persist-\(spec.short)")
        defer { try? FileManager.default.removeItem(at: dir) }
        let probe = IOProbe()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 40, signature: sig,
                                         blockSize: blockSize, hotBudgetBytes: 0, log: probe.sink)
        let coord = PromptCacheCoordinator(store: store)
        let warms = WarmStore()
        let id = UUID()
        var reached = 0
        while reached < boundary {
            let got: Int = await mc.perform { ctx in
                var chunks = 0
                let out = coord.warm(warms, id: id, promptTokens: tokens, model: ctx.model,
                                     parameters: params, persist: .never,
                                     shouldPause: { chunks += 1; return chunks >= chunksPerRound })
                if case let .paused(c) = out { return c }
                if case let .complete(c, _) = out { return c }
                return -1
            }
            guard got > reached else { throw BenchError.armDidNoWork("persist-arm stalled") }
            reached = got
            if reached >= boundary { break }
        }
        probe.reset()
        let before = Census.of(dir)
        let t0 = ContinuousClock.now
        _ = await mc.perform { ctx in coord.finishWarm(warms, id: id, model: ctx.model) }
        let recMs = (ContinuousClock.now - t0).ms
        guard probe.saveCount == 1, probe.refusals.isEmpty else {
            throw BenchError.silentRefusal("finishWarm saved \(probe.saveCount) \(probe.refusals)")
        }
        var fs = 0.0
        for f in Census.of(dir).newFiles(vs: before) {
            fs += Proc.fullFsync(dir.appendingPathComponent(f)) ?? 0
        }
        rep.persistRecordMs.append(recMs)
        rep.persistFsyncMs.append(fs)
    }
    if let rec = rep.persistRecordMs.first {
        let fs = rep.persistFsyncMs.first ?? 0
        print("    record \(fmtMs(rec)) (page-cache write) + F_FULLFSYNC \(fmtMs(fs)) (device flush)")
        print("    ⇒ the process-global eval lock is held for the record portion: \(fmtMs(rec))")
        rep.gates.append(gate("G5.measured", rec > 0, "persist cost measured"))
    }
    if !opts.skipCanary {
        // Separate phase, nothing else running. The canary allocates ONCE outside its loop so it
        // times eval() only and does not churn the Metal buffer cache.
        let stop = Mutex(false)
        let canary = Task.detached { () -> [Double] in
            let a = MLXArray([1.0 as Float])
            eval(a)
            var gaps: [Double] = []
            while !stop.withLock({ $0 }) {
                let t = ContinuousClock.now
                eval(a)
                gaps.append((ContinuousClock.now - t).ms)
            }
            return gaps
        }
        let dir = Scratch.store("canary-\(spec.short)")   // probe scale: the canary only needs MLX work running
        let probe = IOProbe()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 40, signature: sig,
                                         blockSize: blockSize, hotBudgetBytes: 0, log: probe.sink)
        let coord = PromptCacheCoordinator(store: store)
        _ = await mc.perform { ctx in
            coord.warm(promptTokens: probePrompt, model: ctx.model, parameters: params)
        }
        stop.withLock { $0 = true }
        let gaps = await canary.value
        rep.evalStallMaxMs = gaps.max() ?? 0
        print(String(format: "    eval-lock canary: %d samples, max stall %.0f ms, median %.2f ms",
                     gaps.count, rep.evalStallMaxMs, median(gaps)))
        rep.gates.append(gate("G5.evalStall", !gaps.isEmpty,
                              String(format: "max process-global MLX stall during a save: %.0f ms", rep.evalStallMaxMs)))
        try? FileManager.default.removeItem(at: dir)
    }

    // ── G6 · hot tier expectation, asserted on EVERY model ──────────────────────────────────
    // The existing harness only runs the hot gate on the attention model (main.swift:164-168) and
    // its `ok` formula goes vacuously true for hybrids (:173-175). Assert the correct per-arch
    // expectation instead of skipping: HotCodec.extract returns nil for a MambaCache
    // (HotCodec.swift:26-40), so hybrids must never populate the RAM tier.
    print("\n  ── G6 hot tier ──")
    do {
        let dir = Scratch.store("hot-\(spec.short)")
        let probe = IOProbe()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 40, signature: sig,
                                         blockSize: blockSize, hotBudgetBytes: 1 << 34, log: probe.sink)
        let coord = PromptCacheCoordinator(store: store)
        _ = await mc.perform { ctx in
            coord.warm(promptTokens: probePrompt, model: ctx.model, parameters: params)
        }
        // TWO reuses, not one. `record` never warms the RAM tier (warmHot: false —
        // PromptCacheStore.swift:96), and the FIRST reuse populates it but reports
        // `HIT (cold/disk, full)` (:81). Only the second can log `HIT (hot/RAM)` (:56).
        // The first run probed after one call and reported a false negative on the 1.7B.
        _ = await mc.perform { ctx in
            coord.prepare(promptTokens: probePrompt + question, model: ctx.model, parameters: params).suffixStart
        }
        probe.reset()
        _ = await mc.perform { ctx in
            coord.prepare(promptTokens: probePrompt + question, model: ctx.model, parameters: params).suffixStart
        }
        let hotHit = probe.lines.contains { $0.contains("HIT (hot/RAM)") }
        let expectHot = !spec.isHybrid
        print("    second reuse served from RAM=\(hotHit) · expected=\(expectHot) (\(spec.isHybrid ? "hybrid: HotCodec.extract returns nil for MambaCache" : "pure attention"))")
        rep.gates.append(gate("G6.hotTier", hotHit == expectHot,
                              expectHot ? "attention model serves the repeat from RAM"
                                        : "hybrid correctly never populates the RAM tier"))
        try? FileManager.default.removeItem(at: dir)
    }

    // ── Verdict block ───────────────────────────────────────────────────────────────────────
    print("\n  ── \(spec.short) gates ──")
    for g in rep.gates { print("    \(g.verdict.glyph) \(pad(g.id, 16)) \(g.detail)") }
    report.models.append(rep)
}

// ═══════════════════════════════════════════════════════════════════════════════════════════════

print("\n╔═══ SUMMARY ═══")
for m in report.models {
    let failed = m.gates.filter { $0.verdict == .fail }
    print("  \(pad(m.model, 9)) A=\(pad(String(format: "%.0f", m.measuredA), 8)) (pred \(m.predictedA))  M=\(pad(fmtBytes(Int(m.measuredM)), 10))  gates \(m.gates.count - failed.count)/\(m.gates.count)\(failed.isEmpty ? " ✅" : " ❌ \(failed.map(\.id))")")
    if let sq = m.statusQuo, let r = m.residency {
        let a = sq.totalLogicalWrite + sq.totalLogicalRead
        let b = r.totalLogicalWrite + r.totalLogicalRead
        print("            logical I/O: status-quo \(fmtBytes(a)) → residency \(fmtBytes(b))"
            + (b > 0 ? String(format: "  (%.1f×)", Double(a) / Double(b)) : ""))
        print("            device write: \(fmtBytes(sq.totalDeviceWrite)) → \(fmtBytes(r.totalDeviceWrite))"
            + (r.totalDeviceWrite > 0 ? String(format: "  (%.1f×)", Double(sq.totalDeviceWrite) / Double(r.totalDeviceWrite)) : "")
            + "   [device READ was \(fmtBytes(sq.totalDeviceRead)) — page cache absorbed the rest]")
        print("            MLX peak:    \(fmtBytes(m.statusQuoPeakBytes)) → \(fmtBytes(m.residencyPeakBytes))"
            + (m.ballastBytes > 0 ? "   [ballast \(fmtBytes(m.ballastBytes)) held]" : ""))
        print("            wall clock:  \(fmtMs(sq.totalWallMs)) → \(fmtMs(r.totalWallMs))"
            + (r.totalWallMs > 0 ? String(format: "  (%.2f×)", sq.totalWallMs / r.totalWallMs) : ""))
    }
    for n in m.notes { print("            note: \(n)") }
}

if let p = opts.jsonPath {
    let enc = JSONEncoder()
    enc.outputFormatting = [.prettyPrinted, .sortedKeys]
    if let data = try? enc.encode(report) {
        try? data.write(to: URL(fileURLWithPath: p))
        print("\n  artifact → \(p)")
    }
}

let anyFail = report.models.flatMap(\.gates).contains { $0.verdict == .fail }
Scratch.cleanup()
exit(anyFail ? 1 : 0)
