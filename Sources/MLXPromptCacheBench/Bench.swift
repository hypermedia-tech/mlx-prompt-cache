import Foundation
import MLX
import MLXLMCommon
import MLXPromptCache

// MARK: - Model specs with an ANALYTICAL byte model

/// The snapshot byte model, predicted from `config.json` BEFORE any weights load:
///
///   bytes(N) = A·N + M
///
///   A = attentionLayers × 2 (K,V) × kvHeads × headDim × 2 (bf16)      [per token]
///   M = linearLayers × (valueHeads × keyHeadDim × valueHeadDim × 4)   [fixed, fp32 recurrent]
///       plus a small per-layer conv state, reported as the measured residual.
///
/// Deriving A and M analytically is what turns H1 from a curve fit into a falsifiable prediction:
/// a fit over file sizes compared against a fit over the same file sizes proves nothing.
struct ModelSpec: Sendable {
    let id: String
    let short: String
    let layers: Int
    let attentionLayers: Int
    let kvHeads: Int
    let headDim: Int
    let linearLayers: Int
    let valueHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let maxContext: Int

    /// Per-token attention KV cost, bytes. bf16 ⇒ 2 bytes/element, K and V.
    var predictedA: Int { attentionLayers * 2 * kvHeads * headDim * 2 }

    /// Fixed recurrent-state cost, bytes — the SSM term only (fp32). The conv state is a small
    /// additional per-layer constant; measured M should exceed this slightly.
    var predictedMLowerBound: Int { linearLayers * valueHeads * keyHeadDim * valueHeadDim * 4 }

    var isHybrid: Bool { linearLayers > 0 }
    func predictedBytes(_ n: Int) -> Int { predictedA * n + predictedMLowerBound }

    // Values read from each model's config.json in the local HuggingFace cache.
    // 35B: 40 layers, full_attention_interval 4 ⇒ 10 attention / 30 linear_attention.
    static let qwen35B = ModelSpec(
        id: "unsloth/Qwen3.6-35B-A3B-UD-MLX-4bit", short: "35B-A3B",
        layers: 40, attentionLayers: 10, kvHeads: 2, headDim: 256,
        linearLayers: 30, valueHeads: 32, keyHeadDim: 128, valueHeadDim: 128,
        maxContext: 262_144)

    // 9B: 32 layers, full_attention_interval 4 ⇒ 8 attention / 24 linear_attention.
    static let qwen9B = ModelSpec(
        id: "lmstudio-community/Qwen3.5-9B-MLX-4bit", short: "9B",
        layers: 32, attentionLayers: 8, kvHeads: 4, headDim: 256,
        linearLayers: 24, valueHeads: 32, keyHeadDim: 128, valueHeadDim: 128,
        maxContext: 262_144)

    // 1.7B: 28 layers, pure attention. No recurrent state ⇒ M must measure ~0.
    static let qwen17B = ModelSpec(
        id: "mlx-community/Qwen3-1.7B-4bit", short: "1.7B",
        layers: 28, attentionLayers: 28, kvHeads: 8, headDim: 128,
        linearLayers: 0, valueHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        maxContext: 40_960)

    // GLM-4-32B: dense pure-attention transformer, ~same parameter scale as the 35B hybrid — the
    // controlled comparison. 61 uniform KVCacheSimple layers (glm4 conforms to
    // KVCacheDimensionProvider with no newCache override). partial_rotary_factor 0.5 rotates half the
    // head dims but the cache still stores full head_dim=128, so A ignores it; G1 confirms empirically.
    // Predicted A = 61 × 2 × 2 × 128 × 2 = 62,464 B/tok (61.0 KiB) — 3.05× the hybrid's 20 KiB/tok at
    // equal scale, and M = 0. Context is only 32,768, so compare both at ≤24,576 tokens.
    static let glm32B = ModelSpec(
        id: "mlx-community/GLM-4-32B-0414-4bit", short: "GLM-32B",
        layers: 61, attentionLayers: 61, kvHeads: 2, headDim: 128,
        linearLayers: 0, valueHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        maxContext: 32_768)

    static let all = [qwen35B, qwen9B, qwen17B, glm32B]

    static func named(_ s: String) -> ModelSpec? {
        all.first { $0.short.lowercased() == s.lowercased() || $0.id == s }
    }
}

// MARK: - Config-driven prediction (no weights, no GPU)

/// The KV-cache byte model computed straight from a model's `config.json`, so any candidate can be
/// screened for cache cost before its weights are pulled. Same A/M formula the G1 gate validates to
/// 0.00% on loaded models — this is that formula, decoupled from a hardcoded `ModelSpec`.
struct ConfigPrediction {
    let source: String
    let modelType: String
    let totalLayers: Int
    let attentionLayers: Int
    let linearLayers: Int
    let kvHeads: Int
    let headDim: Int
    let maxContext: Int
    let slidingWindow: Int?
    let recurrentStateBytes: Int   // per-linear-layer SSM term (fp32); 0 for non-hybrids
    let kvDtypeBytes: Int

    var isHybrid: Bool { linearLayers > 0 }
    /// Per-token attention KV cost. KV is fp16/bf16 = 2 bytes REGARDLESS of weight quantization —
    /// a 4-bit model still holds a 2-byte KV cache.
    var perTokenBytes: Int { attentionLayers * 2 * kvHeads * headDim * kvDtypeBytes }
    var fixedBytes: Int { linearLayers * recurrentStateBytes }
    func snapshotBytes(_ n: Int) -> Int { perTokenBytes * n + fixedBytes }

    /// Load from a `config.json` path, or from a HuggingFace model id resolved against the local cache.
    static func load(_ arg: String) -> ConfigPrediction? {
        let path = FileManager.default.fileExists(atPath: arg) ? arg : hfCachePath(forModelID: arg)
        guard let path,
              let data = FileManager.default.contents(atPath: path),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        // Multimodal configs nest the language fields under text_config.
        let c = (root["text_config"] as? [String: Any]) ?? root
        func int(_ k: String) -> Int? { (c[k] as? NSNumber)?.intValue }
        let layers = int("num_hidden_layers") ?? 0
        let heads = int("num_attention_heads") ?? 0
        let hidden = int("hidden_size") ?? 0
        let kvHeads = int("num_key_value_heads") ?? heads                 // absent ⇒ MHA
        let headDim = int("head_dim") ?? (heads > 0 ? hidden / heads : 0)
        let types = c["layer_types"] as? [String]                        // hybrid split, Qwen3.5/3.6 style
        let attn = types?.filter { $0 == "full_attention" }.count ?? layers
        let linear = types?.filter { $0 == "linear_attention" }.count ?? 0
        let recur = (int("linear_num_value_heads") ?? 0)
            * (int("linear_key_head_dim") ?? 0) * (int("linear_value_head_dim") ?? 0) * 4
        return ConfigPrediction(
            source: path, modelType: (c["model_type"] as? String) ?? "?",
            totalLayers: layers, attentionLayers: attn, linearLayers: linear,
            kvHeads: kvHeads, headDim: headDim,
            maxContext: int("max_position_embeddings") ?? 0,
            slidingWindow: int("sliding_window"),
            recurrentStateBytes: recur, kvDtypeBytes: 2)
    }

    /// `~/.cache/huggingface/hub/models--org--name/snapshots/<hash>/config.json`
    static func hfCachePath(forModelID id: String) -> String? {
        let hub = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub", isDirectory: true)
            .appendingPathComponent("models--" + id.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)
        guard let snaps = try? FileManager.default.contentsOfDirectory(
            at: hub, includingPropertiesForKeys: nil) else { return nil }
        for snap in snaps {
            let cfg = snap.appendingPathComponent("config.json")
            if FileManager.default.fileExists(atPath: cfg.path) { return cfg.path }
        }
        return nil
    }
}

// MARK: - Cache shape helpers (mirrors PromptCacheIO's internal predicates)

/// Same predicate as `PromptCacheIO.isSliceableLayer` (PromptCacheIO.swift:15). `PromptCacheIO` is
/// internal to the package, so the bench target re-derives it rather than reaching in.
func isSliceableLayer(_ c: KVCache) -> Bool { c is QuantizedKVCache || c is KVCacheSimple }

/// Token length from a sliceable layer's offset — NEVER `cache.first`, which is a `MambaCache`
/// reporting 0 on a hybrid (PromptCacheIO.swift:22-25).
func attnOffset(_ cache: [KVCache]) -> Int? { cache.first(where: isSliceableLayer)?.offset }

func cacheShape(_ cache: [KVCache]) -> String {
    var counts: [String: Int] = [:]
    for c in cache { counts[String(describing: type(of: c)), default: 0] += 1 }
    return counts.sorted { $0.key < $1.key }.map { "\($0.value)×\($0.key)" }.joined(separator: " + ")
}

// MARK: - Prefill primitive

enum Prefill {
    /// One chunk into `cache`. Mirrors `PromptCacheCoordinator.prefillChunked`'s per-chunk mechanic
    /// (PromptCacheCoordinator.swift:171-180) — but THROWS where the original does `try?`.
    /// A silent return would make "the arm did no I/O" indistinguishable from "the arm did nothing",
    /// and both of this harness's residency gates would pass on an empty body.
    static func chunk(_ tokens: [Int], into cache: [KVCache],
                      model: any LanguageModel, stepSize: Int) throws {
        let piece = LMInput(tokens: MLXArray(tokens))
        let result = try model.prepare(piece, cache: cache, windowSize: stepSize)
        if case let .tokens(remaining) = result {
            _ = model(remaining[text: .newAxis], cache: cache, state: nil)
        }
        eval(cache.flatMap { $0.state })
    }
}

// MARK: - The residency box

/// Holds the live `[KVCache]` between `perform` calls — the thing `PromptCacheCoordinator.warm`
/// cannot do today, because it returns only `PromptWarmOutcome` and drops the cache it built
/// (PromptCacheCoordinator.swift:130, :152).
///
/// `@unchecked Sendable` invariant, verbatim the discipline documented at SessionStore.swift:9-17:
/// `live` is only ever read or mutated inside `ModelContainer.perform`, which serialises all model
/// access, so the race `Sendable` guards against cannot occur. A `Mutex` is deliberately NOT used —
/// it would create a second access path reachable off `perform` (session-store-reshape.md:108-115).
///
/// `guardHash` closes the hazard a `UUID`-keyed holder introduces: resuming the wrong id would
/// extend the wrong cache and then record it under a chain hash that does not describe its
/// contents, poisoning the catalog. Cheap to check — SHA-256 over tokens, no GPU, no I/O.
final class HeldCache: @unchecked Sendable {
    private var live: [KVCache]?
    private var guardHash: BlockHash?

    init() {}

    func resume(expecting hash: BlockHash?) -> [KVCache]? {
        guard let live else { return nil }
        guard guardHash == hash else { return nil }   // divergence ⇒ decline, never extend
        return live
    }

    func hold(_ cache: [KVCache], frontier hash: BlockHash?) {
        live = cache
        guardHash = hash
    }

    func drop() { live = nil; guardHash = nil }
    var isHeld: Bool { live != nil }
}

/// Chain hash of the block-aligned prefix `tokens[0..<n]` — the divergence guard's key.
/// Uses the package's own public `BlockHasher`, so the guard is the same hash the catalog keys on.
func frontierHash(_ tokens: [Int], upTo n: Int, blockSize: Int, signature: CacheSignature) -> BlockHash? {
    BlockHasher.boundaries(for: Array(tokens[0 ..< n]), blockSize: blockSize, signature: signature).last
}

// MARK: - Per-resume sample

struct ResumeSample: Sendable, Codable {
    var round: Int
    var reachedTokens: Int
    var logicalWriteBytes: Int      // from the store's own sink: exact, but page-cache-absorbed
    var logicalReadBytes: Int       // size of the snapshot `reuse` loaded whole
    var saveCount: Int
    var loadCount: Int
    var wallMs: Double              // whole warm call, inside perform
    var fsyncMs: Double             // forced device flush AFTER the call — the deferred cost
    var deviceReadBytes: Int        // rusage delta: expect ≈0 when the page cache serves the re-read
    var deviceWriteBytes: Int
    var memPeakDelta: Int
    var memActiveDelta: Int
    var footprint: Int
    var pressured: Bool             // compressor/pageout activity or non-nominal thermals
}

struct ArmResult: Sendable, Codable {
    var name: String
    var samples: [ResumeSample]
    var totalLogicalWrite: Int { samples.reduce(0) { $0 + $1.logicalWriteBytes } }
    var totalLogicalRead: Int { samples.reduce(0) { $0 + $1.logicalReadBytes } }
    var totalDeviceWrite: Int { samples.reduce(0) { $0 + $1.deviceWriteBytes } }
    var totalDeviceRead: Int { samples.reduce(0) { $0 + $1.deviceReadBytes } }
    var totalWallMs: Double { samples.reduce(0) { $0 + $1.wallMs } }
    var totalFsyncMs: Double { samples.reduce(0) { $0 + $1.fsyncMs } }
    var peakMem: Int { samples.map(\.memPeakDelta).max() ?? 0 }
    var anyPressured: Bool { samples.contains { $0.pressured } }

    enum CodingKeys: String, CodingKey { case name, samples }
}

// MARK: - Directory census

struct Census: Sendable {
    var sizes: [String: Int]
    var total: Int { sizes.values.reduce(0, +) }

    static func of(_ dir: URL) -> Census {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        var m: [String: Int] = [:]
        for f in files where f.pathExtension == "safetensors" {
            m[f.lastPathComponent] = (try? f.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0
        }
        return Census(sizes: m)
    }

    func size(_ name: String) -> Int { sizes[name] ?? 0 }
    func newFiles(vs old: Census) -> [String] { sizes.keys.filter { old.sizes[$0] == nil }.sorted() }
}

// MARK: - State comparison (the sensitive H4 observable)

/// Compares two caches tensor by tensor. For a hybrid the quantity that accumulates error along
/// the sequence is the RECURRENT state, not the sampled token — 48 greedy tokens sampled after the
/// state is already fixed is close to the least sensitive probe available. This is the sensitive one.
struct StateDiff: Sendable, Codable {
    var attnMaxAbs: Double
    var recurrentMaxAbs: Double
    var attnTensors: Int
    var recurrentTensors: Int
    var shapeMismatch: String?
    var identical: Bool { shapeMismatch == nil && attnMaxAbs == 0 && recurrentMaxAbs == 0 }

    var summary: String {
        if let m = shapeMismatch { return "SHAPE MISMATCH: \(m)" }
        return String(format: "attn maxΔ=%.3g (%d tensors) · recurrent maxΔ=%.3g (%d tensors)",
                      attnMaxAbs, attnTensors, recurrentMaxAbs, recurrentTensors)
    }
}

func diffStates(_ a: [KVCache], _ b: [KVCache]) -> StateDiff {
    var d = StateDiff(attnMaxAbs: 0, recurrentMaxAbs: 0, attnTensors: 0, recurrentTensors: 0,
                      shapeMismatch: nil)
    guard a.count == b.count else {
        d.shapeMismatch = "layer count \(a.count) vs \(b.count)"
        return d
    }
    for (i, (la, lb)) in zip(a, b).enumerated() {
        let sa = la.state, sb = lb.state
        guard sa.count == sb.count else {
            d.shapeMismatch = "layer \(i) tensor count \(sa.count) vs \(sb.count)"
            return d
        }
        let sliceable = isSliceableLayer(la)
        for (ta, tb) in zip(sa, sb) {
            guard ta.shape == tb.shape else {
                d.shapeMismatch = "layer \(i) shape \(ta.shape) vs \(tb.shape)"
                return d
            }
            // max|a−b| on GPU, one scalar back. For non-NaN floats maxAbs == 0 IS bit identity,
            // and it is far cheaper than materialising multi-GB tensors to compare bytes.
            let m = MLX.abs(ta.asType(.float32) - tb.asType(.float32)).max()
            eval(m)
            let v = Double(m.item(Float.self))
            if sliceable {
                d.attnTensors += 1
                d.attnMaxAbs = Swift.max(d.attnMaxAbs, v)
            } else {
                d.recurrentTensors += 1
                d.recurrentMaxAbs = Swift.max(d.recurrentMaxAbs, v)
            }
        }
    }
    return d
}
