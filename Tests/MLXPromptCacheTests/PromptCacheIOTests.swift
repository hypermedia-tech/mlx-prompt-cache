import Foundation
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite struct PromptCacheIOTests {
    let sig = Fixture.signature

    @Test func saveLoadRoundTrip() throws {
        let url = Fixture.tempDir().appendingPathComponent("snap.safetensors")
        let live = Fixture.syntheticCache(tokens: 8)
        try PromptCacheIO.save(prefixTokenCount: 8, liveCache: live, url: url, signature: sig)
        let loaded = PromptCacheIO.loadFull(url: url, signature: sig)
        #expect(loaded?.count == live.count)
        #expect(loaded?.first?.offset == 8)
    }

    @Test func trimsToMatched() throws {
        let url = Fixture.tempDir().appendingPathComponent("snap.safetensors")
        try PromptCacheIO.save(prefixTokenCount: 8, liveCache: Fixture.syntheticCache(tokens: 8), url: url, signature: sig)
        let loaded = PromptCacheIO.loadFull(url: url, signature: sig)
        #expect(loaded?.first?.offset == 8)                                                  // full length
        #expect(loaded.flatMap { PromptCacheIO.trim($0, toMatched: 4) }?.first?.offset == 4)  // 8 → 4
    }

    @Test func saveCopiesAndTrims() throws {
        let url = Fixture.tempDir().appendingPathComponent("snap.safetensors")
        let live = Fixture.syntheticCache(tokens: 12)
        try PromptCacheIO.save(prefixTokenCount: 8, liveCache: live, url: url, signature: sig)
        #expect(live.first?.offset == 12)
        #expect(PromptCacheIO.loadFull(url: url, signature: sig)?.first?.offset == 8)
    }

    @Test func signatureMismatchMisses() throws {
        let url = Fixture.tempDir().appendingPathComponent("snap.safetensors")
        try PromptCacheIO.save(prefixTokenCount: 8, liveCache: Fixture.syntheticCache(tokens: 8), url: url, signature: sig)
        let other = CacheSignature(modelId: "other", kvDType: "bf16", kvBits: nil, buildVersion: "t1")
        #expect(PromptCacheIO.loadFull(url: url, signature: other) == nil)
    }

    @Test func missingFileMisses() {
        let url = Fixture.tempDir().appendingPathComponent("nope.safetensors")
        #expect(PromptCacheIO.loadFull(url: url, signature: sig) == nil)
    }

    // MARK: - ChunkedKVCache is NOT sliceable, despite subclassing KVCacheSimple (regression guard)

    /// `ChunkedKVCache` is-a `KVCacheSimple`, so a naive `c is KVCacheSimple` gate reports it
    /// sliceable — contradicting `isSliceableLayer`'s own doc and mis-keying its `startPosition`-based
    /// trim. Pin the exclusion so a reordered type check can't silently reintroduce it. Headless: no
    /// model needed, because the correct behaviour is "don't slice it", which is a classification, not
    /// a generation result. (A cold==warm test would need a real chunked model; none exists in
    /// mlx-swift-lm today.)
    @Test func chunkedLayerIsNotSliceable() {
        #expect(PromptCacheIO.isSliceableLayer(Fixture.chunkedCache(tokens: 8)) == false)
        #expect(PromptCacheIO.isSliceable(Fixture.attentionPlusChunked(slice: 8)) == false)
    }

    /// A chunked layer makes the whole cache boundary-only — exactly like a Mamba layer. The attention
    /// layer is still what `tokenLength` keys on (never the chunked one, whose offset semantics differ).
    @Test func chunkedTokenLengthReadsAttentionLayer() {
        #expect(PromptCacheIO.tokenLength(Fixture.attentionPlusChunked(slice: 8)) == 8)   // the attn layer
        #expect(PromptCacheIO.tokenLength([Fixture.chunkedCache(tokens: 8)]) == nil)      // pure chunked → uncacheable
    }

    /// Non-sliceable ⇒ valid only at a captured boundary: a sub-prefix save is refused, not trimmed.
    @Test func chunkedSaveRejectsSubPrefixAcceptsBoundary() throws {
        let url = Fixture.tempDir().appendingPathComponent("chunk.safetensors")
        #expect(throws: PromptCacheIO.Failure.hybridNotAtBoundary) {
            try PromptCacheIO.save(prefixTokenCount: 4, liveCache: Fixture.attentionPlusChunked(slice: 8),
                                   url: url, signature: sig)
        }
        // At the boundary it saves and round-trips through the disk path (which has an explicit
        // ChunkedKVCache reconstruction case), the attention layer keying the reload.
        try PromptCacheIO.save(prefixTokenCount: 8, liveCache: Fixture.attentionPlusChunked(slice: 8),
                               url: url, signature: sig)
        #expect(PromptCacheIO.tokenLength(PromptCacheIO.loadFull(url: url, signature: sig) ?? []) == 8)
    }
}
