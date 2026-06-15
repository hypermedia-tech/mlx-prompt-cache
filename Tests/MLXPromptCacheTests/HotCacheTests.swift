import Foundation
import MLX          // DType only — no MLXArray, so these stay pure (run under `swift test`)
import Testing
@testable import MLXPromptCache

@Suite struct HotCacheTests {
    /// A fake snapshot of exactly `bytes` bytes — no MLX arrays, just value bytes, so footprint == `bytes`.
    private func blob(_ bytes: Int) -> [CacheBytes] {
        [CacheBytes(className: "KVCacheSimple", metaState: [],
                    tensors: [TensorBytes(data: Data(count: bytes), dtype: .float16, shape: [bytes])])]
    }

    @Test func insertThenResident() {
        var hot = HotCache(budgetBytes: 1000)
        hot.insert("a", blobs: blob(100), fullTokens: 8)
        let e = hot.resident("a")
        #expect(e?.fullTokens == 8)
        #expect(e?.bytes == 100)
        #expect(hot.totalBytes == 100)
        #expect(hot.resident("missing") == nil)
    }

    @Test func budgetEvictsLRU() {
        var hot = HotCache(budgetBytes: 250)
        hot.insert("a", blobs: blob(100), fullTokens: 1)
        hot.insert("b", blobs: blob(100), fullTokens: 1)
        hot.insert("c", blobs: blob(100), fullTokens: 1)   // 300 > 250 ⇒ evict LRU ("a")
        #expect(hot.resident("a") == nil)
        #expect(hot.resident("b") != nil)
        #expect(hot.resident("c") != nil)
        #expect(hot.totalBytes == 200)
    }

    @Test func residentBumpsLRU() {
        var hot = HotCache(budgetBytes: 250)
        hot.insert("a", blobs: blob(100), fullTokens: 1)
        hot.insert("b", blobs: blob(100), fullTokens: 1)
        _ = hot.resident("a")                              // touch "a" ⇒ "b" becomes LRU
        hot.insert("c", blobs: blob(100), fullTokens: 1)   // evict LRU ("b"), not "a"
        #expect(hot.resident("a") != nil)
        #expect(hot.resident("b") == nil)
        #expect(hot.resident("c") != nil)
    }

    @Test func oversizedIsSkipped() {
        var hot = HotCache(budgetBytes: 100)
        hot.insert("big", blobs: blob(200), fullTokens: 1)   // larger than the whole budget ⇒ skip (no thrash)
        #expect(hot.resident("big") == nil)
        #expect(hot.totalBytes == 0)
    }

    @Test func reinsertReplacesBytes() {
        var hot = HotCache(budgetBytes: 1000)
        hot.insert("a", blobs: blob(100), fullTokens: 1)
        hot.insert("a", blobs: blob(50), fullTokens: 2)      // replace, not accumulate
        #expect(hot.totalBytes == 50)
        #expect(hot.resident("a")?.fullTokens == 2)
    }

    @Test func dropAndClear() {
        var hot = HotCache(budgetBytes: 1000)
        hot.insert("a", blobs: blob(100), fullTokens: 1)
        hot.insert("b", blobs: blob(100), fullTokens: 1)
        hot.drop("a")
        #expect(hot.resident("a") == nil)
        #expect(hot.totalBytes == 100)
        hot.clear()
        #expect(hot.resident("b") == nil)
        #expect(hot.totalBytes == 0)
    }

    @Test func disabledNeverCaches() {
        var hot = HotCache(budgetBytes: 0)
        #expect(hot.isEnabled == false)
        hot.insert("a", blobs: blob(100), fullTokens: 1)
        #expect(hot.resident("a") == nil)
        #expect(hot.totalBytes == 0)
    }
}
