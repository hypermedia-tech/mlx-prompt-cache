
/// Invalidation key - a cached prefix is only valid for the same model + params + quant.
public struct CacheSignature: Hashable, Sendable, Codable {
    public let modelId: String
    public let kvDType: String
    public let kvBits: Int?
    public let buildVersion: String
    
    public init(modelId: String, kvDType: String, kvBits: Int?, buildVersion: String) {
        self.modelId = modelId
        self.kvDType = kvDType
        self.kvBits = kvBits
        self.buildVersion = buildVersion
    }
    
    /// Canonical bytes folded into the block hash and stored in snapshot metadata
    var canonical: String { "\(modelId)|\(kvDType)|\(kvBits.map(String.init) ?? "-")|\(buildVersion)" }
}
