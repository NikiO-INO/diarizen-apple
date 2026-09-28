// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DiariZen",
    platforms: [
        // CoreML ML Program + ANE features target macOS 14+.
        .macOS(.v14)
    ],
    products: [
        .library(name: "DiariZen", targets: ["DiariZen"]),
        .executable(name: "diarizen-cli", targets: ["diarizen-cli"]),
    ],
    targets: [
        .target(
            name: "DiariZen",
            path: "Sources/DiariZen"
        ),
        .executableTarget(
            name: "diarizen-cli",
            dependencies: ["DiariZen"],
            path: "Sources/diarizen-cli"
        ),
        .testTarget(
            name: "DiariZenTests",
            dependencies: ["DiariZen"],
            path: "Tests/DiariZenTests"
        ),
    ]
)
