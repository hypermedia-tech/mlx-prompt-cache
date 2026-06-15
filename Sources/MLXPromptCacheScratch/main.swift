import Foundation
import MLX
import MLXLMCommon
import MLXLLM
import MLXHuggingFace
import HuggingFace
import Tokenizers
import MLXPromptCache

let modelID = "mlx-community/Qwen3-1.7B-4bit"
let blockSize = 256

print("Loading \(modelID) …")
let mc = try await LLMModelFactory.shared.loadContainer(
    from: #hubDownloader(),                       // default HubClient — uses your HF cache, downloads if missing
    using: #huggingFaceTokenizerLoader(),
    configuration: ModelConfiguration(id: modelID)
)

// A long, stable "document" + a short question. The document is the reusable prefix.
let document = String(
    repeating: "Cyber threat report: an unusual outbound connection was seen from host 10.2.4.18 to a known C2 endpoint over port 443; the process tree shows powershell spawning a child that beacons every 60 seconds. ",
    count: 40)
let question = "\n\nQuestion: In one sentence, what is the suspicious behaviour?\nAnswer:"

let fullTokens = await mc.encode(document + question)
let prefixLen = max(blockSize, (fullTokens.count - 64) / blockSize * blockSize)   // block-aligned; leave a suffix
let prefixTokens = Array(fullTokens.prefix(prefixLen))
print("tokens — full: \(fullTokens.count), cached prefix: \(prefixLen) (\(prefixLen / blockSize) blocks), suffix: \(fullTokens.count - prefixLen)")

let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mlxpc-itest-\(UUID().uuidString)")
let signature = CacheSignature(modelId: modelID, kvDType: "bf16", kvBits: nil, buildVersion: "itest-1")
let store = try PromptCacheStore(directory: dir, budgetBytes: 8_000_000_000, signature: signature,
                                 blockSize: blockSize, hotBudgetBytes: 8_000_000_000)   // RAM hot tier on
let params = GenerateParameters(maxTokens: 48, temperature: 0)   // greedy ⇒ deterministic ⇒ cold output must == warm

struct RunResult: Sendable { var text = ""; var promptTime = 0.0; var prefilled = 0; var loadMs = 0.0 }

func oneRun(reuse: Bool) async throws -> RunResult {
    try await mc.perform { context in
        var out = RunResult()
        var inputTokens = fullTokens
        let cache: [KVCache]
        if reuse {
            let t0 = Date()
            let reused = store.reuse(forTokens: fullTokens)
            out.loadMs = Date().timeIntervalSince(t0) * 1000
            if let reused {
                inputTokens = Array(fullTokens[reused.matchedTokens...])
                cache = reused.cache
                print("  reuse HIT: loaded \(reused.matchedTokens) tokens in \(String(format: "%.1f", out.loadMs)) ms; prefilling \(inputTokens.count)")
            } else {
                cache = makePromptCache(model: context.model, parameters: params)
                print("  reuse MISS")
            }
        } else {
            cache = makePromptCache(model: context.model, parameters: params)
        }
        let stream = try MLXLMCommon.generate(
            input: LMInput(tokens: MLXArray(inputTokens)), cache: cache, parameters: params, context: context)
        for await g in stream {
            switch g {
            case .chunk(let s): out.text += s
            case .info(let i): out.promptTime = i.promptTime; out.prefilled = i.promptTokenCount
            case .toolCall: break
            }
        }
        if !reuse { try store.record(prefixTokens: prefixTokens, cache: cache) }
        return out
    }
}

// Warm up Metal/model so the cold timing isn't inflated by one-time init (doesn't touch the store).
_ = try await mc.perform { context -> Int in
    let s = try MLXLMCommon.generate(
        input: LMInput(tokens: MLXArray(Array(fullTokens.prefix(8)))),
        cache: makePromptCache(model: context.model, parameters: params),
        parameters: GenerateParameters(maxTokens: 1, temperature: 0), context: context)
    for await _ in s {}
    return 0
}

print("\n— COLD (no cache; records the prefix) —")
let cold = try await oneRun(reuse: false)
print("  prefilled \(cold.prefilled) tokens, prompt \(String(format: "%.0f", cold.promptTime * 1000)) ms")

print("\n— WARM (disk reuse; promotes the snapshot into the RAM hot tier) —")
let warm = try await oneRun(reuse: true)
print("  prefilled \(warm.prefilled) tokens, prompt \(String(format: "%.0f", warm.promptTime * 1000)) ms")

// HOT: delete every on-disk snapshot first, so a reuse hit can ONLY come from the in-RAM bytes that the
// WARM run promoted. A HIT here (same short suffix prefilled, not a full re-prefill) proves the hot tier
// served it without touching disk.
print("\n— HOT (RAM bytes; on-disk snapshots deleted first) —")
let snaps = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
    .filter { $0.pathExtension == "safetensors" } ?? []
for f in snaps { try? FileManager.default.removeItem(at: f) }
print("  deleted \(snaps.count) on-disk snapshot file(s)")
let hot = try await oneRun(reuse: true)
print("  prefilled \(hot.prefilled) tokens, prompt \(String(format: "%.0f", hot.promptTime * 1000)) ms")

let coldReady = cold.promptTime * 1000
let warmReady = warm.loadMs + warm.promptTime * 1000
let hotReady = hot.loadMs + hot.promptTime * 1000
let outputsMatch = cold.text == warm.text && warm.text == hot.text
let hotFromRAM = hot.prefilled == warm.prefilled   // reuse hit (short suffix), not a full re-prefill from a miss
print("""

================ RESULT ================
outputs identical : \(outputsMatch ? "✅ YES (cold == warm == hot)" : "❌ NO")
hot from RAM      : \(hotFromRAM ? "✅ YES — served after on-disk snapshots were deleted" : "❌ NO — fell back to full prefill")
cold : prefill \(String(format: "%.0f", coldReady)) ms over \(cold.prefilled) tokens
warm : load \(String(format: "%.1f", warm.loadMs)) ms + prefill \(String(format: "%.0f", warm.promptTime * 1000)) ms over \(warm.prefilled) tokens
hot  : load \(String(format: "%.1f", hot.loadMs)) ms + prefill \(String(format: "%.0f", hot.promptTime * 1000)) ms over \(hot.prefilled) tokens
ready : cold \(String(format: "%.0f", coldReady)) → warm \(String(format: "%.0f", warmReady)) → hot \(String(format: "%.0f", hotReady)) ms
========================================
""")
if !outputsMatch { print("\nCOLD:\n\(cold.text)\n\nWARM:\n\(warm.text)\n\nHOT:\n\(hot.text)") }
