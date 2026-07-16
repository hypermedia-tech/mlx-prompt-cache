import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import HuggingFace
import Tokenizers
import MLXPromptCache

// ───────────────────────── config ─────────────────────────

let blockSize = 256
let sizes = [2048, 8192, 16384, 24576]                 // cached-document token counts (block-aligned; within Qwen3's 32k)

struct ModelSpec { let id: String; let testHot: Bool }
let models = [
    ModelSpec(id: "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit",    testHot: false),   // PRODUCTION deep tenant (hybrid Mamba) — the numbers §0 wants
    ModelSpec(id: "mlx-community/Qwen3-1.7B-4bit",          testHot: true),    // attention → cold / disk / RAM
    ModelSpec(id: "lmstudio-community/Qwen3.5-9B-MLX-4bit", testHot: false),   // hybrid → cold / disk only (RAM tier is attention-only)
]
let question = "\n\nQuestion: In one sentence, what is the single most suspicious behaviour in this log?\nAnswer:"
let questionB = "\n\nQuestion: Name the single host that appears most often in this log.\nAnswer:"

// ───────────────────────── helpers ─────────────────────────

func pad2(_ n: Int) -> String { n < 10 ? "0\(n)" : "\(n)" }

/// One deterministic, varied synthetic cyber-log line (no randomness → runs are comparable).
func logLine(_ i: Int) -> String {
    func oct(_ n: Int) -> Int { (n % 254) + 1 }
    let ports = [22, 80, 443, 445, 3389, 8080, 53, 25]
    let procs = ["powershell.exe", "cmd.exe", "svchost.exe", "rundll32.exe", "wscript.exe", "mshta.exe"]
    return "[2026-06-16T\(pad2((i / 3600) % 24)):\(pad2((i / 60) % 60)):\(pad2(i % 60))Z] "
        + "host=10.\(oct(i)).\(oct(i * 7)).\(oct(i * 13)) pid=\(1000 + (i * 131) % 60000) "
        + "proc=\(procs[(i * 5) % procs.count]) dst_port=\(ports[(i * 3) % ports.count]) "
        + "bytes_out=\((i * 977) % 1_000_000) beacon_s=\(30 + (i * 17) % 600) verdict=review\n"
}

func ms(_ x: Double) -> String { String(format: "%.0f ms", x) }
func col(_ s: String, _ w: Int) -> String { s.count >= w ? s : s + String(repeating: " ", count: w - s.count) }

func deleteSnapshots(_ dir: URL) -> Int {
    let snaps = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
        .filter { $0.pathExtension == "safetensors" } ?? []
    for f in snaps { try? FileManager.default.removeItem(at: f) }
    return snaps.count
}

/// Bytes of the recorded snapshot(s) in a store directory — the on-disk cost of a warmed prefix.
func snapshotBytes(in dir: URL) -> Int64 {
    let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
    return files.filter { $0.pathExtension == "safetensors" }.reduce(Int64(0)) {
        $0 + Int64((try? $1.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
    }
}

struct RunResult: Sendable {
    var text = ""
    var prepareMs = 0.0        // cache contribution: cold = prefill+record, warm/hot = load
    var ttftMs = 0.0           // prompt-ready → first generated token (the felt latency §0 measures)
    var genTokPerSec = 0.0     // decode throughput AFTER the first token (cache must NOT move this)
    var boundary = 0           // captured block-aligned prefix (cold); 0 otherwise
    var matched = 0            // tokens served from cache (hit); 0 on a cold capture
    var suffixStart = 0
    var outcome = ""
}

struct ConvOut: Sendable {
    let d1: Int;
    let d2: Int;
    let a1: [Int];
    let a2Held: [Int];
    let ttft1: Double;
    let ttft2: Double;
}
struct Row {
    let size: Int
    let coldTtft, warmTtft, newQTtft: Double
    let hotTtft: Double?
    let prefillTokPerSec, kibPerTok, coldGenTps, warmGenTps: Double
    let ok: Bool
}

/// One run through the real coordinator path: time `prepare` (the cache's contribution) AND the felt TTFT
/// (prompt-ready → first token), then generate the suffix, reading decode throughput from `.info`.
/// `prepare` on a miss = prefill + save (first-run cost); on a hit = disk- or RAM-load.
func measure(mc: ModelContainer, coordinator: PromptCacheCoordinator,
             prompt: [Int], params: GenerateParameters) async throws -> RunResult {
    try await mc.perform { context in
        var r = RunResult()
        let t0 = Date()                                            // prompt-ready (tokens precomputed)
        let prepared = coordinator.prepare(promptTokens: prompt, model: context.model, parameters: params)
        r.prepareMs = Date().timeIntervalSince(t0) * 1000
        r.suffixStart = prepared.suffixStart
        r.outcome = "\(prepared.outcome)"
        if case let .hit(m)      = prepared.outcome { r.matched = m }
        if case let .captured(b) = prepared.outcome { r.boundary = b }
        let genTokens = prepared.suffixStart > 0 ? Array(prompt[prepared.suffixStart...]) : prompt
        let stream = try MLXLMCommon.generate(
            input: LMInput(tokens: MLXArray(genTokens)), cache: prepared.cache, parameters: params, context: context)
        var firstAt: Date?
        for await g in stream {
            if case .chunk(let s) = g {
                if firstAt == nil { firstAt = Date(); r.ttftMs = firstAt!.timeIntervalSince(t0) * 1000 }
                r.text += s
            }
            if case .info(let info) = g {                          // same fields the applied engine reads
                r.genTokPerSec = info.generateTime > 0 ? Double(info.generationTokenCount) / info.generateTime : 0
            }
        }
        return r
    }
}

// ───────────────────────── run ─────────────────────────

for model in models {
    print("\n=================== \(model.id) ===================")
    print("Loading …")
    let mc = try await LLMModelFactory.shared.loadContainer(
        from: #hubDownloader(),
        using: #huggingFaceTokenizerLoader(),
        configuration: ModelConfiguration(id: model.id))

    let params = GenerateParameters(maxTokens: 48, temperature: 0)   // greedy ⇒ deterministic ⇒ cold == warm == hot
    let sig = CacheSignature(modelId: model.id, kvDType: "bf16", kvBits: nil, buildVersion: "sweep-1")

    // Build one corpus big enough for the largest size; tokenise once and slice per size.
    let corpus = (0 ..< 1400).map(logLine).joined()
    let corpusTokens = await mc.encode(corpus)
    let qTokens = await mc.encode(question)
    let qbTokens = await mc.encode(questionB)
    print("corpus: \(corpusTokens.count) tokens available · question: \(qTokens.count) tokens")
    guard corpusTokens.count >= (sizes.max() ?? 0) else {
        print("⚠️ corpus too small (\(corpusTokens.count) < \(sizes.max()!)) — raise the line count"); continue
    }

    // Warm up Metal/model so the first cold timing isn't inflated by one-time init.
    _ = try await mc.perform { context -> Int in
        let s = try MLXLMCommon.generate(
            input: LMInput(tokens: MLXArray(Array(corpusTokens.prefix(8)))),
            cache: makePromptCache(model: context.model, parameters: params),
            parameters: GenerateParameters(maxTokens: 1, temperature: 0), context: context)
        for await _ in s {}
        return 0
    }

    var rows: [Row] = []
    for size in sizes {
        let document = Array(corpusTokens.prefix(size))     // block-aligned (sizes are 256-multiples)
        let fullA = document + qTokens
        let fullB = document + qbTokens
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxpc-sweep-\(UUID().uuidString)")
        let store = try PromptCacheStore(directory: dir, budgetBytes: 16_000_000_000,
                                         signature: sig, blockSize: blockSize, hotBudgetBytes: 16_000_000_000)
        let coordinator = PromptCacheCoordinator(store: store)

        let cold  = try await measure(mc: mc, coordinator: coordinator, prompt: fullA, params: params)  // miss → capture
        let bytes = snapshotBytes(in: dir)                                                              // the recorded prefix
        store.clearHot()
        let warm  = try await measure(mc: mc, coordinator: coordinator, prompt: fullA, params: params)  // disk, same Q
        let newQ  = try await measure(mc: mc, coordinator: coordinator, prompt: fullB, params: params)  // disk, DIFFERENT Q
        var hot: RunResult? = nil
        if model.testHot {
            let n = deleteSnapshots(dir)
            hot = try await measure(mc: mc, coordinator: coordinator, prompt: fullA, params: params)     // RAM only
            if n == 0 { print("  ⚠️ [\(size)] no snapshot file to delete before HOT") }
        }

        let prefillTokPerSec = cold.prepareMs > 0 ? Double(cold.boundary) / (cold.prepareMs / 1000) : 0
        let kibPerTok        = cold.boundary  > 0 ? Double(bytes) / 1024 / Double(cold.boundary) : 0
        // Correctness: same-Q reuse byte-identical; the different-Q ask still hit the doc blocks; hot matches warm.
        let ok = cold.text == warm.text
            && newQ.matched > 0
            && (hot == nil || warm.text == hot!.text)
        rows.append(Row(size: size,
                        coldTtft: cold.ttftMs, warmTtft: warm.ttftMs, newQTtft: newQ.ttftMs, hotTtft: hot?.ttftMs,
                        prefillTokPerSec: prefillTokPerSec, kibPerTok: kibPerTok,
                        coldGenTps: cold.genTokPerSec, warmGenTps: warm.genTokPerSec, ok: ok))
        print("  [\(size)] TTFT cold \(ms(cold.ttftMs)) → warm \(ms(warm.ttftMs)) (\(String(format: "%.1f×", cold.ttftMs / max(warm.ttftMs, 0.001)))) · new-Q \(ms(newQ.ttftMs)) (matched \(newQ.matched)) · prefill \(String(format: "%.0f", prefillTokPerSec)) tok/s · \(String(format: "%.0f", kibPerTok)) KiB/tok · gen \(String(format: "%.0f", warm.genTokPerSec)) tok/s · \(ok ? "✅" : "❌")")
        store.clearHot()
    }

    // ── K2a gate: paused→resumed warm must be byte-identical to an uninterrupted warm (and to cold) ──
    do {
        let doc = Array(corpusTokens.prefix(2048))
        let full = doc + qTokens
        let gateParams = GenerateParameters(maxTokens: 48, temperature: 0)   // greedy ⇒ deterministic

        func generateOverWarmed(_ store: PromptCacheStore) async throws -> String {
            let coordinator = PromptCacheCoordinator(store: store)
            return try await mc.perform { context in
                let prepared = coordinator.prepare(promptTokens: full, model: context.model, parameters: gateParams)
                let gen = prepared.suffixStart > 0 ? Array(full[prepared.suffixStart...]) : full
                let stream = try MLXLMCommon.generate(
                    input: LMInput(tokens: MLXArray(gen)), cache: prepared.cache,
                    parameters: gateParams, context: context)
                var t = ""; for await g in stream { if case .chunk(let s) = g { t += s } }
                return t
            }
        }

        func freshStore(_ tag: String) throws -> PromptCacheStore {
            let dir = FileManager.default.temporaryDirectory
                .appendingPathComponent("mlxpc-warmgate-\(tag)-\(UUID().uuidString)")
            return try PromptCacheStore(directory: dir, budgetBytes: 16_000_000_000, signature: sig, blockSize: blockSize)
        }

        // A — pause once mid-prefill, then resume.
        let storeA = try freshStore("A")
        let (pausedOK, completeOK): (Bool, Bool) = await mc.perform { context in
            let coordA = PromptCacheCoordinator(store: storeA)
            var fired = false
            let pauseOnce: () -> Bool = { if fired { return false }; fired = true; return true }
            let p = coordA.warm(promptTokens: full, model: context.model, parameters: gateParams, shouldPause: pauseOnce)
            guard case let .paused(cached) = p else { print("  ⚠️ warm did not pause: \(p)"); return (false, false) }
            let c = coordA.warm(promptTokens: full, model: context.model, parameters: gateParams)
            guard case let .complete(total, prefilled) = c else { print("  ⚠️ resume did not complete: \(c)"); return (true, false) }
            print("  warm gate: paused@\(cached) → complete(total \(total), prefilled \(prefilled) = remainder)")
            return (true, true)
        }
        let textA = try await generateOverWarmed(storeA)

        // B — uninterrupted warm, fresh store.
        let storeB = try freshStore("B")
        _ = await mc.perform { context -> Int in
            _ = PromptCacheCoordinator(store: storeB).warm(promptTokens: full, model: context.model, parameters: gateParams)
            return 0
        }
        let textB = try await generateOverWarmed(storeB)

        // COLD — no cache at all.
        let textCold = try await mc.perform { context in
            let stream = try MLXLMCommon.generate(
                input: LMInput(tokens: MLXArray(full)),
                cache: makePromptCache(model: context.model, parameters: gateParams),
                parameters: gateParams, context: context)
            var t = ""; for await g in stream { if case .chunk(let s) = g { t += s } }
            return t
        }

        let ok = pausedOK && completeOK && textA == textB && textB == textCold
        print("  K2a warm gate [\(model.id)]: paused-resume == uninterrupted == cold → \(ok ? "✅" : "❌")")
        if !ok {
            print("    A   = \(textA.prefix(80))")
            print("    B   = \(textB.prefix(80))")
            print("    cold= \(textCold.prefix(80))")
        }
    }
    
    // ── Conversation gate: a HELD SessionCache. Turn 2 prefills only the new question, and the
    //    held-cache answer is token-identical to a cold full-prefill of the whole conversation. ──
    do {
        let document = Array(corpusTokens.prefix(2048))                                   // block-aligned root
        let convParams = GenerateParameters(maxTokens: 48, temperature: 0)                // greedy ⇒ held == cold
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxpc-conv-\(UUID().uuidString)")
        let store = try PromptCacheStore(directory: dir, budgetBytes: 16_000_000_000,
                                         signature: sig, blockSize: blockSize)
        let coordinator = PromptCacheCoordinator(store: store)

        // Warm the root to disk so the session SEEDS from it (not a cold prefill).
        _ = await mc.perform { context -> Int in
            _ = coordinator.warm(promptTokens: document, model: context.model, parameters: convParams); return 0
        }

        // Both turns inside ONE perform so the non-Sendable [KVCache] never crosses the boundary.
        let held: ConvOut = try await mc.perform { context in
            let session = SessionCache(warmRoot: store.reuse(forTokens: document),
                                       makeCache: { makePromptCache(model: context.model, parameters: convParams) })
            // Turn 1 — whole document resident, only the question prefills.
            let d1in = session.advance(fullPromptTokens: document + qTokens)
            let d1 = d1in.text.tokens.shape.last ?? 0
            let t1 = Date(); var a1: [Int] = []; var ttft1 = 0.0
            for await g in try generateTokens(input: d1in, cache: session.cache, parameters: convParams, context: context) {
                if case .token(let tok) = g { if a1.isEmpty { ttft1 = Date().timeIntervalSince(t1) * 1000 }; a1.append(tok) }
            }
            // Turn 2 — held cache already carries document+Q1+A1; only Q2 prefills.
            let d2in = session.advance(fullPromptTokens: document + qTokens + a1 + qbTokens)
            let d2 = d2in.text.tokens.shape.last ?? 0
            let t2 = Date(); var a2: [Int] = []; var ttft2 = 0.0
            for await g in try generateTokens(input: d2in, cache: session.cache, parameters: convParams, context: context) {
                if case .token(let tok) = g { if a2.isEmpty { ttft2 = Date().timeIntervalSince(t2) * 1000 }; a2.append(tok) }
            }
            session.release()
            return ConvOut(d1: d1, d2: d2, a1: a1, a2Held: a2, ttft1: ttft1, ttft2: ttft2)
        }

        // Cold: prefill the ENTIRE conversation from scratch; its answer must equal the held turn-2 answer.
        let a2Cold: [Int] = try await mc.perform { context in
            var ids: [Int] = []
            for await g in try generateTokens(input: LMInput(tokens: MLXArray(document + qTokens + held.a1 + qbTokens)),
                                              cache: makePromptCache(model: context.model, parameters: convParams),
                                              parameters: convParams, context: context) {
                if case .token(let tok) = g { ids.append(tok) }
            }
            return ids
        }

        let deltaOnly = held.d1 == qTokens.count && held.d2 == qbTokens.count             // only the new turn prefilled
        let heldEqCold = held.a2Held == a2Cold                                            // held == cold, token-identical
        print("  conversation gate [\(model.id)]​: d1=\(held.d1)(Q \(qTokens.count)) d2=\(held.d2)(Q2 \(qbTokens.count)) · "
            + "TTFT turn1 \(ms(held.ttft1)) → turn2 \(ms(held.ttft2)) · delta-only \(deltaOnly ? "✅" : "❌") · "
            + "held==cold \(heldEqCold ? "✅" : "❌")")
        if !heldEqCold { print("    held a2=\(Array(held.a2Held.prefix(12))) · cold a2=\(Array(a2Cold.prefix(12)))") }
    }

    print("\n  \(col("doc tok", 8))| \(col("TTFT cold", 11))| \(col("TTFT warm", 11))| \(col("TTFT new-Q", 11))| \(col("TTFT hot", 10))| \(col("prefill tok/s", 14))| \(col("KiB/tok", 8))| \(col("gen tok/s", 10))| ok")
    for r in rows {
        print("  \(col("\(r.size)", 8))| \(col(ms(r.coldTtft), 11))| \(col(ms(r.warmTtft), 11))| \(col(ms(r.newQTtft), 11))| \(col(r.hotTtft.map(ms) ?? "n/a", 10))| \(col(String(format: "%.0f", r.prefillTokPerSec), 14))| \(col(String(format: "%.0f", r.kibPerTok), 8))| \(col(String(format: "%.0f", r.warmGenTps), 10))| \(r.ok ? "✅" : "❌")")
    }
}
