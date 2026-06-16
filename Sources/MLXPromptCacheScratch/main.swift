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
    ModelSpec(id: "mlx-community/Qwen3-1.7B-4bit",          testHot: true),    // attention → cold / disk / RAM
    ModelSpec(id: "lmstudio-community/Qwen3.5-9B-MLX-4bit", testHot: false),   // hybrid → cold / disk only (RAM tier is attention-only)
]
let question = "\n\nQuestion: In one sentence, what is the single most suspicious behaviour in this log?\nAnswer:"

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

struct RunResult: Sendable { var text = ""; var prepareMs = 0.0; var suffixStart = 0; var outcome = "" }
struct Row { let size: Int; let cold: Double; let warm: Double; let hot: Double?; let ok: Bool }

/// One run through the real coordinator path: time `prepare` (the cache's contribution), then generate the
/// suffix. `prepare` on a miss = prefill + save (first-run cost); on a hit = disk- or RAM-load.
func measure(mc: ModelContainer, coordinator: PromptCacheCoordinator,
             prompt: [Int], params: GenerateParameters) async throws -> RunResult {
    try await mc.perform { context in
        var r = RunResult()
        let t0 = Date()
        let prepared = coordinator.prepare(promptTokens: prompt, model: context.model, parameters: params)
        r.prepareMs = Date().timeIntervalSince(t0) * 1000
        r.suffixStart = prepared.suffixStart
        r.outcome = "\(prepared.outcome)"
        let genTokens = prepared.suffixStart > 0 ? Array(prompt[prepared.suffixStart...]) : prompt
        let stream = try MLXLMCommon.generate(
            input: LMInput(tokens: MLXArray(genTokens)), cache: prepared.cache, parameters: params, context: context)
        for await g in stream { if case .chunk(let s) = g { r.text += s } }
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
        let document = Array(corpusTokens.prefix(size))
        let full = document + qTokens
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxpc-sweep-\(UUID().uuidString)")
        let store = try PromptCacheStore(directory: dir, budgetBytes: 16_000_000_000,
                                         signature: sig, blockSize: blockSize, hotBudgetBytes: 16_000_000_000)
        let coordinator = PromptCacheCoordinator(store: store)

        // COLD — first run: miss → capture (prefill + save). The honest first-run cost.
        let cold = try await measure(mc: mc, coordinator: coordinator, prompt: full, params: params)
        // WARM — disk reuse: clear RAM first so the hit comes from disk.
        store.clearHot()
        let warm = try await measure(mc: mc, coordinator: coordinator, prompt: full, params: params)
        // HOT — RAM reuse: WARM already promoted into RAM; delete the on-disk file so a hit can only be RAM.
        var hot: RunResult? = nil
        if model.testHot {
            let n = deleteSnapshots(dir)
            hot = try await measure(mc: mc, coordinator: coordinator, prompt: full, params: params)
            if n == 0 { print("  ⚠️ [\(size)] no snapshot file found to delete before HOT") }
        }

        let ok = cold.text == warm.text && (hot == nil || warm.text == hot!.text)
        rows.append(Row(size: size, cold: cold.prepareMs, warm: warm.prepareMs, hot: hot?.prepareMs, ok: ok))
        print("  [\(size)] cold \(ms(cold.prepareMs)) (\(cold.outcome)) · warm \(ms(warm.prepareMs)) · hot \(hot.map { ms($0.prepareMs) } ?? "n/a") · suffix \(full.count - warm.suffixStart) tok · outputs \(ok ? "✅" : "❌")")
        store.clearHot()
    }

    print("\n  \(col("doc tokens", 11))| \(col("cold (prefill+save)", 20))| \(col("warm (disk)", 13))| \(col("hot (RAM)", 11))| \(col("cold/warm", 10))| outputs")
    for r in rows {
        let speed = r.warm > 0 ? String(format: "%.0f×", r.cold / r.warm) : "—"
        print("  \(col("\(r.size)", 11))| \(col(ms(r.cold), 20))| \(col(ms(r.warm), 13))| \(col(r.hot.map(ms) ?? "n/a", 11))| \(col(speed, 10))| \(r.ok ? "✅" : "❌")")
    }
}
