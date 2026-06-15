import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

@Suite struct HotCodecTests {
    /// Byte-for-byte equality of two caches' `state` (shape + raw bytes per tensor).
    private func statesByteEqual(_ a: [KVCache], _ b: [KVCache]) -> Bool {
        guard a.count == b.count else { return false }
        for (ca, cb) in zip(a, b) {
            let sa = ca.state, sb = cb.state
            guard sa.count == sb.count else { return false }
            for (xa, xb) in zip(sa, sb) {
                guard xa.shape == xb.shape,
                      xa.asData(access: .copy).data == xb.asData(access: .copy).data else { return false }
            }
        }
        return true
    }

    @Test func roundTripIsByteIdentical() {
        let original = Fixture.syntheticCache(tokens: 8)
        let blobs = HotCodec.extract(original)
        #expect(blobs != nil)
        let restored = HotCodec.reconstruct(blobs!)
        #expect(restored.count == original.count)
        #expect(restored.first?.offset == 8)
        #expect(statesByteEqual(restored, original))
    }

    @Test func metaStatePreserved() {
        let original = Fixture.syntheticCache(tokens: 8)
        let restored = HotCodec.reconstruct(HotCodec.extract(original)!)
        #expect(restored.first?.metaState == original.first?.metaState)
    }

    @Test func reconstructionsHaveIndependentBuffers() {
        // Two caches from the same bytes must not share storage: writing one leaves the other — and a
        // fresh reconstruct — byte-for-byte untouched. This is the §3.4 "no aliasing" property.
        let blobs = HotCodec.extract(Fixture.syntheticCache(tokens: 8))!
        let a = HotCodec.reconstruct(blobs)
        let b = HotCodec.reconstruct(blobs)
        let bBefore = b.first!.state.first!.asData(access: .copy).data
        let append = MLXArray.zeros([1, 2, 4, 8], dtype: .bfloat16)   // matches Fixture's [B,kvHeads,_,headDim]
        _ = a.first?.update(keys: append, values: append)            // mutate a (realloc-on-append write path)
        #expect(a.first?.offset == 12)                                          // a grew
        #expect(b.first?.offset == 8)                                           // b untouched
        #expect(b.first!.state.first!.asData(access: .copy).data == bBefore)    // b's bytes unchanged
        #expect(HotCodec.reconstruct(blobs).first?.offset == 8)                 // blobs immutable
    }

    @Test func footprintMatchesByteCount() {
        let original = Fixture.syntheticCache(tokens: 8)
        let blobs = HotCodec.extract(original)!
        let expected = original.reduce(0) { $0 + $1.state.reduce(0) { $0 + $1.asData(access: .copy).data.count } }
        #expect(HotCodec.footprint(blobs) == expected)
    }

    @Test func unsupportedTypeReturnsNil() {
        // RotatingKVCache is neither KVCacheSimple nor QuantizedKVCache ⇒ not hot-cacheable (stays cold).
        #expect(HotCodec.extract([RotatingKVCache(maxSize: 16)]) == nil)
    }
}
