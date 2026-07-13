import Foundation
import Testing
@testable import MLXPromptCache

@Suite struct CatalogTests {
    let bs = 4
    let sig = Fixture.signature
    func emptyCatalog() -> Catalog { Catalog(header: .init(signature: sig, blockSize: bs)) }
    func hashes(_ tokens: [Int]) -> [BlockHash] { BlockHasher.boundaries(for: tokens, blockSize: bs, signature: sig) }
    
    @Test func emptyLookupMisses() {
        var cat = emptyCatalog()
        #expect(cat.lookup([]) == nil)
        #expect(cat.lookup(hashes(Fixture.tokens(8))) == nil)
    }
    
    @Test func recordThenExactHit() {
        var cat = emptyCatalog()
        let toks = Fixture.tokens(8)
        let plan = cat.planRecord(hashes(toks), blockSize: bs)!
        _ = cat.commit(plan, byteSize: 1000, budgetBytes: 1_000_000)
        let hit = cat.lookup(hashes(toks))
        #expect(hit?.matchedTokens == 8)
        #expect(hit?.fileName == plan.fileName)
    }
    
    @Test func crossPromptPartialMatch() {
        var cat = emptyCatalog()
        let aPlan = cat.planRecord(hashes([1, 2, 3, 4, 5, 6, 7, 8]), blockSize: bs)!
        _ = cat.commit(aPlan, byteSize: 1000, budgetBytes: 1_000_000)
        let hit = cat.lookup(hashes([1, 2, 3, 4, 9, 9, 9, 9]))
        #expect(hit?.matchedTokens == 4)
        #expect(hit?.fileName == aPlan.fileName)
    }
    
    @Test func planRecordSkipsFullyCached() {
        var cat = emptyCatalog()
        let toks = Fixture.tokens(8)
        _ = cat.commit(cat.planRecord(hashes(toks), blockSize: bs)!, byteSize: 1000, budgetBytes: 1_000_000)
        #expect(cat.planRecord(hashes(toks), blockSize: bs) == nil)
    }
    
    @Test func evictUntilBudget() {
        var cat = emptyCatalog()
        for seed in [0, 100, 200] {
            let p = cat.planRecord(hashes(Fixture.tokens(8, seed: seed)), blockSize: bs)!
            _ = cat.commit(p, byteSize: 1000, budgetBytes: 2500)
        }
        #expect(cat.totalBytes <= 2500)
        #expect(cat.files.count == 2)
    }
    
    @Test func lookupTouchPromotes() {
        var cat = emptyCatalog()
        let a = Fixture.tokens(8, seed: 0), b = Fixture.tokens(8, seed: 100)
        _ = cat.commit(cat.planRecord(hashes(a), blockSize: bs)!, byteSize: 1000, budgetBytes: 2500)
        let bPlan = cat.planRecord(hashes(b), blockSize: bs)!
        _ = cat.commit(bPlan, byteSize: 1000, budgetBytes: 2500)
        _ = cat.lookup(hashes(a))
        let cPlan = cat.planRecord(hashes(Fixture.tokens(8, seed: 200)), blockSize: bs)!
        let evicted = cat.commit(cPlan, byteSize: 1000, budgetBytes: 2500)
        #expect(evicted == [bPlan.fileName])
        #expect(cat.lookup(hashes(a)) != nil)
    }
    
    @Test func ownershipOrphanEviction() {
        var cat = emptyCatalog()
        let aPlan = cat.planRecord(hashes(Fixture.tokens(8)),  blockSize: bs)!
        _ = cat.commit(aPlan, byteSize: 1000, budgetBytes: 1_000_000)
        let bPlan = cat.planRecord(hashes(Fixture.tokens(12)), blockSize: bs)!
        let evicted = cat.commit(bPlan, byteSize: 1000, budgetBytes: 1_000_000)
        #expect(evicted == [aPlan.fileName])
        #expect(cat.lookup(hashes(Fixture.tokens(8)))?.fileName == bPlan.fileName)
        #expect(cat.files.count == 1)
    }
    
    @Test func codableRoundTrip() throws {
        var cat = emptyCatalog()
        let toks = Fixture.tokens(8)
        _ = cat.commit(cat.planRecord(hashes(toks), blockSize: bs)!, byteSize: 1000, budgetBytes: 1_000_000)
        var restored = try JSONDecoder().decode(Catalog.self, from: try JSONEncoder().encode(cat))
        #expect(restored.lookup(hashes(toks))?.matchedTokens == 8)
        #expect(restored.totalBytes == cat.totalBytes)
    }

    // MARK: - probe (read-only twin of lookup, K0)

    @Test func probeDoesNotBumpClockOrLastAccess() {
        var cat = emptyCatalog()
        let toks = Fixture.tokens(8)
        let plan = cat.planRecord(hashes(toks), blockSize: bs)!
        _ = cat.commit(plan, byteSize: 100, budgetBytes: 1_000)

        let clockBefore = cat.clock
        let accessBefore = cat.files[plan.fileName]!.lastAccess
        #expect(cat.probe(hashes(toks))?.matchedTokens == 8)            // read-only twin
        #expect(cat.clock == clockBefore)
        #expect(cat.files[plan.fileName]!.lastAccess == accessBefore)

        _ = cat.lookup(hashes(toks))                                    // the mutating twin still touches
        #expect(cat.clock == clockBefore + 1)
        #expect(cat.files[plan.fileName]!.lastAccess == clockBefore + 1)
    }

    @Test func probedEntryStillEvictsFirst() {
        // The LRU-corruption scenario the probe exists to avoid: A is oldest, B newer; probing A
        // repeatedly must NOT save it — the next over-budget commit still evicts A.
        var cat = emptyCatalog()
        let a = hashes(Fixture.tokens(8, seed: 0))
        let b = hashes(Fixture.tokens(8, seed: 100))
        let c = hashes(Fixture.tokens(8, seed: 200))

        let planA = cat.planRecord(a, blockSize: bs)!
        _ = cat.commit(planA, byteSize: 100, budgetBytes: 1_000)        // A: lastAccess 1
        _ = cat.commit(cat.planRecord(b, blockSize: bs)!, byteSize: 100, budgetBytes: 1_000)  // B: 2

        _ = cat.probe(a); _ = cat.probe(a); _ = cat.probe(a)            // probe A hard

        // Committing C at a 250-byte budget (3 × 100 on disk) must evict exactly the LRU entry —
        // and that is still A, because probes left its lastAccess alone.
        let evicted = cat.commit(cat.planRecord(c, blockSize: bs)!, byteSize: 100, budgetBytes: 250)
        #expect(evicted == [planA.fileName])
    }
}
