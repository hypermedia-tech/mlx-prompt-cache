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
            dependencies: ["MLXPromptCache"]
        ),
        .testTarget(
            name: "MLXPromptCacheTests",
            dependencies: [
                "MLXPromptCache",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ]
        ),
    ],
    swiftLanguageModes: [.v6]
)
