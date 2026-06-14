import Testing
@testable import MLXPromptCache

@Suite struct BlockHasherTests {
    let sig = Fixture.signature
    
    @Test func deterministic() {
        #expect(BlockHasher.hash(parent: nil, blockTokens: [1, 2, 3], signature: sig)
                == BlockHasher.hash(parent: nil, blockTokens: [1, 2, 3], signature: sig))
    }
    
    @Test func differentTokensDiffer() {
        #expect(BlockHasher.hash(parent: nil, blockTokens: [1, 2, 3], signature: sig)
            != BlockHasher.hash(parent: nil, blockTokens: [1, 2, 4], signature: sig))
    }
    
    @Test func signatureIsolates() {
        let base = BlockHasher.hash(parent: nil, blockTokens: [1, 2, 3], signature: sig)
        let other = CacheSignature(modelId: "other", kvDType: "bf16", kvBits: nil, buildVersion: "t1")
        let quant = CacheSignature(modelId: "test-model", kvDType: "bf16", kvBits: 4, buildVersion: "t1")
        #expect(base != BlockHasher.hash(parent: nil, blockTokens: [1, 2, 3], signature: other))
        #expect(base != BlockHasher.hash(parent: nil, blockTokens: [1, 2, 3], signature: quant))
    }
    
    @Test func parentChains() {
        let root = BlockHasher.hash(parent: nil, blockTokens: [1, 2, 3], signature: sig)
        #expect(BlockHasher.hash(parent: root, blockTokens: [4, 5, 6], signature: sig)
            != BlockHasher.hash(parent: nil, blockTokens: [4, 5, 6], signature: sig))
    }
}

extension BlockHasherTests {
    @Test func fullBlocksOnly() {
        #expect(BlockHasher.boundaries(for: Fixture.tokens(10), blockSize: 4, signature: sig).count == 2)
        #expect(BlockHasher.boundaries(for: Fixture.tokens(8), blockSize: 4, signature: sig).count == 2)
        #expect(BlockHasher.boundaries(for:Fixture.tokens(3), blockSize: 4, signature: sig).count == 0)
        #expect(BlockHasher.boundaries(for: [], blockSize: 4, signature: sig).count == 0)
    }
    
    @Test func isTheChainOverHash() {
        let toks = Fixture.tokens(8)
        let h0 = BlockHasher.hash(parent: nil, blockTokens: Array(toks[0..<4]), signature: sig)
        let h1 = BlockHasher.hash(parent: h0, blockTokens: Array(toks[4..<8]), signature: sig)
        #expect(BlockHasher.boundaries(for: toks, blockSize: 4, signature: sig) == [h0, h1])
    }
    
    @Test func sharedPrefixSharesLeadingHashes() {
        let a = BlockHasher.boundaries(for: [1, 2, 3, 4, 5, 6, 7, 8], blockSize: 4, signature: sig)
        let b = BlockHasher.boundaries(for: [1, 2, 3, 4, 9, 9, 9, 9], blockSize: 4, signature: sig)
        #expect(a[0] == b[0])
        #expect(a[1] != b[1])
    }
    
    @Test func changePropagatesForwardOnly() {
        let base = Fixture.tokens(12)
        var changed = base; changed[5] = 999
        let a = BlockHasher.boundaries(for: base, blockSize: 4, signature: sig)
        let b = BlockHasher.boundaries(for: changed, blockSize: 4, signature: sig)
        #expect(a[0] == b[0])
        #expect(a[1] != b[1])
        #expect(a[2] != b[2])
    }
}
