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
}
