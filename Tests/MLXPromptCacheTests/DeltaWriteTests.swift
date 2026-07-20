import Foundation
import MLX
import MLXLMCommon
import Testing
@testable import MLXPromptCache

/// The delta-write path: `planRecord(delta:)` plans only new blocks, `commit` no longer orphans,
/// `chain` walks the ordered files, and `reuse` reassembles them byte-identically to a whole snapshot.
/// The catalog tests are pure logic; the store tests are MLX-backed (real `KVCacheSimple` buffers) and
/// run wherever the other MLX-backed unit tests do.
@Suite struct DeltaWriteTests {
    let bs = 4
    let sig = Fixture.signature
    func cat() -> Catalog {
        Catalog(header: .init(signature: sig, blockSize: bs, indexLayout: Catalog.currentIndexLayout))
    }
    func hashes(_ t: [Int]) -> [BlockHash] { BlockHasher.boundaries(for: t, blockSize: bs, signature: sig) }

    // MARK: - Catalog delta logic (pure)

    @Test func deltaPlansOnlyNewBlocks() {
        var c = cat()
        let toks = Fixture.tokens(16)                                   // 4 blocks
        // First persist: 8 tokens (blocks 0-1).
        let p1 = c.planRecord(hashes(Array(toks.prefix(8))), blockSize: bs, delta: true)!
        _ = c.commit(p1, byteSize: 100, budgetBytes: 1 << 30)
        // Second persist: 16 tokens — delta must plan ONLY blocks 2-3, range [8,16).
        let p2 = c.planRecord(hashes(toks), blockSize: bs, delta: true)!
        #expect(p2.delta)
        #expect(p2.boundaries.count == 2)                              // only the new blocks
        #expect(p2.fromToken == 8 && p2.toToken == 16)
        #expect(p2.fileName != p1.fileName)
    }

    @Test func deltaCommitDoesNotOrphan() {
        var c = cat()
        let toks = Fixture.tokens(16)
        let p1 = c.planRecord(hashes(Array(toks.prefix(8))), blockSize: bs, delta: true)!
        _ = c.commit(p1, byteSize: 100, budgetBytes: 1 << 30)
        let p2 = c.planRecord(hashes(toks), blockSize: bs, delta: true)!
        let deleted = c.commit(p2, byteSize: 100, budgetBytes: 1 << 30)
        #expect(deleted.isEmpty)                                       // the delta write orphans nothing
        #expect(c.files.count == 2)                                    // BOTH files retained
    }

    @Test func chainReturnsOrderedFilesToMatch() {
        var c = cat()
        let toks = Fixture.tokens(16)
        let p1 = c.planRecord(hashes(Array(toks.prefix(8))), blockSize: bs, delta: true)!
        _ = c.commit(p1, byteSize: 100, budgetBytes: 1 << 30)
        let p2 = c.planRecord(hashes(toks), blockSize: bs, delta: true)!
        _ = c.commit(p2, byteSize: 100, budgetBytes: 1 << 30)
        let chain = c.chain(hashes(toks))
        #expect(chain.map { $0.fileName } == [p1.fileName, p2.fileName])
        #expect(chain.last?.tokenCount == 16)
        // A shorter match resolves the same first file, capped at its boundary.
        #expect(c.chain(hashes(Array(toks.prefix(8)))).map { $0.fileName } == [p1.fileName])
    }

    @Test func chainStopsAtAHole() {
        var c = cat()
        let toks = Fixture.tokens(16)
        let p1 = c.planRecord(hashes(Array(toks.prefix(8))), blockSize: bs, delta: true)!
        _ = c.commit(p1, byteSize: 100, budgetBytes: 1 << 30)
        let p2 = c.planRecord(hashes(toks), blockSize: bs, delta: true)!
        _ = c.commit(p2, byteSize: 100, budgetBytes: 1 << 30)
        c.evict(p1.fileName)                                          // punch a hole at the front
        #expect(c.chain(hashes(toks)).isEmpty)                       // block 0 gone ⇒ nothing resolves
    }

    // MARK: - Store round-trip (MLX-backed) — the correctness + amplification proof

    private func totalBytes(_ dir: URL) -> Int {
        let files = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
        return files.filter { $0.pathExtension == "safetensors" }.reduce(0) {
            $0 + ((try? $1.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
    }
    private func stateBytesEqual(_ a: [KVCache], _ b: [KVCache]) -> Bool {
        guard a.count == b.count else { return false }
        for (la, lb) in zip(a, b) {
            let sa = la.state, sb = lb.state
            guard sa.count == sb.count else { return false }
            for (ta, tb) in zip(sa, sb) {
                guard ta.shape == tb.shape else { return false }
                if ta.asData(access: .copy).data != tb.asData(access: .copy).data { return false }
            }
        }
        return true
    }

    /// The lossless proof: a prefix persisted in TWO delta writes, then reused (chain + reassemble),
    /// is byte-identical to the source cache. Both persists slice the SAME live cache, so the
    /// reassembled `[0,512) ++ [512,1024)` must reconstruct it exactly.
    @Test func deltaReassemblyIsByteIdentical() throws {
        let dir = Fixture.tempDir()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 34, signature: sig, blockSize: 256)
        let tokens = Fixture.tokens(1024)                             // 4 × 256-blocks
        let live = Fixture.patternedCache(tokens: 1024)               // non-zero, position-dependent
        try store.record(prefixTokens: Array(tokens.prefix(512)), cache: live)   // delta [0,512)
        try store.record(prefixTokens: tokens, cache: live)                      // delta [512,1024)
        let reused = store.reuse(forTokens: tokens)
        #expect(reused?.matchedTokens == 1024)
        #expect(stateBytesEqual(reused?.cache ?? [], live))          // reassembled == source
    }

    /// The amplification proof: the second persist writes only its DELTA (~512 tokens), and the first
    /// file is NOT deleted. Two files retained; the second is far smaller than a whole 1024-token
    /// snapshot would be (which the old path wrote and then orphaned the first).
    @Test func deltaWriteVolumeIsDeltaNotWhole() throws {
        let dir = Fixture.tempDir()
        let store = try PromptCacheStore(directory: dir, budgetBytes: 1 << 34, signature: sig, blockSize: 256)
        let tokens = Fixture.tokens(1024)
        let live = Fixture.patternedCache(tokens: 1024)
        try store.record(prefixTokens: Array(tokens.prefix(512)), cache: live)
        let afterFirst = totalBytes(dir)
        try store.record(prefixTokens: tokens, cache: live)
        let afterSecond = totalBytes(dir)
        let secondWrote = afterSecond - afterFirst
        let files = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil))?
            .filter { $0.pathExtension == "safetensors" }.count ?? 0
        #expect(files == 2)                                          // no orphan — both kept
        // The delta (512 tokens) is ~the first file's size; a WHOLE 1024 write would be ~2×. Well under.
        #expect(secondWrote < afterFirst * 3 / 2)
    }
}
