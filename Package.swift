// swift-tools-version:5.5
// The swift-tools-version declares the minimum version of Swift required to build this package.
//
// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import PackageDescription
import Foundation

// MARK: - Configuration

/// Detect if we're using pre-built PJRT plugins or building from source
let usePrebuiltPJRT = ProcessInfo.processInfo.environment["X10_USE_PREBUILT_PJRT"] != "false"

/// Get the X10 library path from environment or use default search paths
func getX10LibraryPath() -> String? {
    // Check environment variable first
    if let path = ProcessInfo.processInfo.environment["X10_LIBRARY_PATH"] {
        return path
    }

    // Check for pre-built library in common locations
    let searchPaths = [
        "/usr/local/lib",
        "/usr/lib",
        "/opt/x10/lib",
        NSHomeDirectory() + "/.local/lib",
        ProcessInfo.processInfo.environment["CONDA_PREFIX"].map { $0 + "/lib" },
    ].compactMap { $0 }

    for path in searchPaths {
        if FileManager.default.fileExists(atPath: path + "/libx10.so") ||
           FileManager.default.fileExists(atPath: path + "/libx10.dylib") {
            return path
        }
    }

    return nil
}

/// Get include path for X10 headers
func getX10IncludePath() -> String? {
    if let path = ProcessInfo.processInfo.environment["X10_INCLUDE_PATH"] {
        return path
    }

    let searchPaths = [
        "/usr/local/include",
        "/usr/include",
        "/opt/x10/include",
        NSHomeDirectory() + "/.local/include",
        ProcessInfo.processInfo.environment["CONDA_PREFIX"].map { $0 + "/include" },
    ].compactMap { $0 }

    for path in searchPaths {
        if FileManager.default.fileExists(atPath: path + "/x10") {
            return path
        }
    }

    return nil
}

// MARK: - Linker Settings

/// Build linker settings based on configuration
func buildLinkerSettings() -> [LinkerSetting] {
    var settings: [LinkerSetting] = []

    if let libPath = getX10LibraryPath() {
        settings.append(.unsafeFlags(["-L\(libPath)"]))
    }

    // Link against X10 library
    settings.append(.linkedLibrary("x10"))

    // Link against system libraries needed by PJRT
    #if os(Linux)
    settings.append(.linkedLibrary("dl"))
    settings.append(.linkedLibrary("pthread"))
    settings.append(.linkedLibrary("stdc++"))
    #elseif os(macOS)
    settings.append(.linkedLibrary("c++"))
    #endif

    return settings
}

/// Build C settings based on configuration
func buildCSettings() -> [CSetting] {
    var settings: [CSetting] = []

    if let includePath = getX10IncludePath() {
        settings.append(.unsafeFlags(["-I\(includePath)"]))
    }

    // Define for OpenXLA/PJRT build
    settings.append(.define("X10_OPENXLA"))

    if usePrebuiltPJRT {
        settings.append(.define("X10_USE_PREBUILT_PJRT"))
    }

    return settings
}

/// Build Swift settings
func buildSwiftSettings() -> [SwiftSetting] {
    var settings: [SwiftSetting] = []

    if let includePath = getX10IncludePath() {
        settings.append(.unsafeFlags(["-I\(includePath)"]))
    }

    return settings
}

// MARK: - Package Definition

let package = Package(
    name: "TensorFlow",
    platforms: [
        .macOS(.v11),
        .iOS(.v14),
    ],
    products: [
        // Main TensorFlow library
        .library(
            name: "TensorFlow",
            type: .dynamic,
            targets: ["TensorFlow"]),

        // Tensor library (no X10 dependency)
        .library(
            name: "Tensor",
            type: .dynamic,
            targets: ["Tensor"]),

        // X10 optimizers
        .library(
            name: "x10_optimizers_optimizer",
            type: .dynamic,
            targets: ["x10_optimizers_optimizer"]),
        .library(
            name: "x10_optimizers_tensor_visitor_plan",
            type: .dynamic,
            targets: ["x10_optimizers_tensor_visitor_plan"]),

        // X10 with OpenXLA/PJRT (new)
        .library(
            name: "X10",
            type: .dynamic,
            targets: ["X10"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-numerics", from: "1.0.0"),
        .package(url: "https://github.com/pvieito/PythonKit.git", .branch("master")),
    ],
    targets: [
        // MARK: - Core Targets

        .target(
            name: "Tensor",
            dependencies: []),

        .target(
            name: "CTensorFlow",
            dependencies: []),

        // MARK: - X10/PJRT System Library

        // System library target for X10 with PJRT
        // This requires pre-built X10 library or PJRT plugins
        .systemLibrary(
            name: "CX10",
            path: "Sources/CX10",
            pkgConfig: "x10",
            providers: [
                .brew(["x10"]),
                .apt(["libx10-dev"]),
            ]),

        // C modules for X10 bindings
        .target(
            name: "CX10Modules",
            dependencies: [],
            cSettings: buildCSettings()),

        // MARK: - X10 Swift Bindings (OpenXLA/PJRT)

        .target(
            name: "X10",
            dependencies: [
                "Tensor",
                "CX10Modules",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            path: "Sources/x10/swift_bindings",
            exclude: [
                "optimizers",
                "training_loop.swift",
            ],
            swiftSettings: buildSwiftSettings(),
            linkerSettings: buildLinkerSettings()),

        // MARK: - TensorFlow Target

        .target(
            name: "TensorFlow",
            dependencies: [
                "Tensor",
                "PythonKit",
                "CTensorFlow",
                "CX10Modules",
                .product(name: "Numerics", package: "swift-numerics"),
            ],
            swiftSettings: [
                .define("DEFAULT_BACKEND_EAGER"),
            ]),

        // MARK: - Optimizers

        .target(
            name: "x10_optimizers_tensor_visitor_plan",
            dependencies: ["TensorFlow"],
            path: "Sources/x10",
            sources: [
                "swift_bindings/optimizers/TensorVisitorPlan.swift",
            ]),

        .target(
            name: "x10_optimizers_optimizer",
            dependencies: [
                "x10_optimizers_tensor_visitor_plan",
                "TensorFlow",
            ],
            path: "Sources/x10",
            sources: [
                "swift_bindings/optimizers/Optimizer.swift",
                "swift_bindings/optimizers/Optimizers.swift",
            ]),

        // MARK: - Experimental

        .target(
            name: "Experimental",
            dependencies: [],
            path: "Sources/third_party/Experimental"),

        // MARK: - Tests

        .testTarget(
            name: "ExperimentalTests",
            dependencies: ["Experimental"]),

        .testTarget(
            name: "TensorTests",
            dependencies: ["Tensor"]),

        .testTarget(
            name: "TensorFlowTests",
            dependencies: ["TensorFlow"]),

        .testTarget(
            name: "X10Tests",
            dependencies: ["X10"],
            path: "Tests/X10Tests"),
    ]
)
