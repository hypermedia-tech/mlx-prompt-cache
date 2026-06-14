import Foundation
import CryptoKit

/// A block's chain digest - hex so it's Codable and dictionary-key friendly.
public struct BlockHash: Hashable, Sendable, Codable {
    public let hex: String
    public init(hex: String) { self.hex = hex }
}

public enum BlockHasher {
    static let rootSeed = Data("mlx-prompt-cache-root".utf8)
    
    /// `SHA256( signature ++ (parent ?? root) ++ packed little-endian Int32 tokens )`. Folding the
    /// signature in first makes a different model / dtype / quant structurally unable to collide.
    public static func hash(
        parent: BlockHash?,
        blockTokens: [Int],
        signature: CacheSignature
    ) -> BlockHash {
        var sha = SHA256()
        sha.update(data: Data(signature.canonical.utf8))
        sha.update(data: parent.map { Data($0.hex.utf8) } ?? rootSeed)
        var packed = Data(capacity: blockTokens.count * 4)
        for token in blockTokens {
            var le = Int32(truncatingIfNeeded: token).littleEndian
            withUnsafeBytes(of: &le) { packed.append(contentsOf: $0) }
        }
        sha.update(data: packed)
        return BlockHash(hex: sha.finalize().map { String(format: "%02x", $0) }.joined())
    }
    
    /// Full-block chain hashes for a token sequence; trailing partial block omitted
    public static func boundaries(for tokens: [Int], blockSize: Int, signature: CacheSignature) -> [BlockHash] {
        let blocks = tokens.count / blockSize
        var result: [BlockHash] = []; result.reserveCapacity(blocks)
        var parent: BlockHash?
        for i in 0..<blocks {
            let h = hash(parent: parent,
                         blockTokens: Array(tokens[(i * blockSize)..<((i + 1) * blockSize)]),
                         signature: signature)
            result.append(h); parent = h
        }
        return result
    }
}
