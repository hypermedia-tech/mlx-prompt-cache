import Foundation
@testable import MLXPromptCache

enum Fixture {
    static let signature = CacheSignature(
        modelId: "test-model",
        kvDType: "bf16",
        kvBits: nil,
        buildVersion: "t1"
    )
    
    static func tokens(_ n: Int, seed: Int = 0) -> [Int] {
        (0..<n).map { ($0 + seed) % 50_000 }
    }
    
    static func tempDir() -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxpc-\(UUID().uuidString)")
        try! FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
