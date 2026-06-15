// swift-tools-version: 6.3
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MLXPromptCache",
    platforms: [
        .macOS(.v26),
        .iOS(.v26),
        .tvOS(.v26),
        .visionOS(.v26),
    ],
    products: [
        .library(
            name: "MLXPromptCache",
            targets: ["MLXPromptCache"]
        ),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.2"),
        .package(url: "https://github.com/mattt/EventSource.git", "1.3.0" ..< "1.4.0"),
    ],
    targets: [
        .target(
            name: "MLXPromptCache",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
        .executableTarget(
            name: "MLXPromptCacheScratch",
            dependencies: [
                "MLXPromptCache",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "EventSource", package: "EventSource"),
            ]
        ),
        .testTarget(
            name: "MLXPromptCacheTests",
            dependencies: [
                "MLXPromptCache",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
