licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# OpenXLA-based tensor library
# Migrated from TensorFlow XLA to standalone OpenXLA
cc_library(
    name = "tensor",
    srcs = glob(
        [
            "*.cpp",
            "ops/*.cpp",
        ],
        exclude = ["test.cpp"],
    ),
    hdrs = glob([
        "*.h",
        "ops/*.h",
    ]),
    deps = [
        # XLA client (use PJRT-based client for OpenXLA)
        "//xla_client:computation_client",
        "//xla_client:pjrt_computation_client",
        "//xla_client:xla_client_util",
        # Core XLA libraries (OpenXLA paths)
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        # XLA client libraries
        "@xla//xla/client:padding",
        "@xla//xla/client:xla_builder",
        "@xla//xla/client:xla_computation",
        "@xla//xla/client/lib:arithmetic",
        "@xla//xla/client/lib:comparators",
        "@xla//xla/client/lib:constants",
        "@xla//xla/client/lib:logdet",
        "@xla//xla/client/lib:math",
        "@xla//xla/client/lib:matrix",
        "@xla//xla/client/lib:pooling",
        "@xla//xla/client/lib:prng",
        "@xla//xla/client/lib:qr",
        "@xla//xla/client/lib:self_adjoint_eig",
        "@xla//xla/client/lib:slicing",
        "@xla//xla/client/lib:svd",
        # XLA service libraries
        "@xla//xla/service:hlo",
        "@xla//xla/service:hlo_proto_cc",
        # MLIR/HLO utilities (for StableHLO support)
        # "@xla//xla/mlir_hlo",
        # Abseil libraries
        "@com_google_absl//absl/base",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/debugging:stacktrace",
        "@com_google_absl//absl/debugging:symbolize",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:str_format",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_absl//absl/types:variant",
    ],
)

# Windows DLL import library
filegroup(
    name = "get_x10_dll_import_lib",
    srcs = [":x10.dll"],
    output_group = "interface_library",
)

genrule(
    name = "x10_dll_import_lib",
    srcs = [":get_x10_dll_import_lib"],
    outs = ["x10.lib"],
    cmd = select({
        "@platforms//os:windows": "cp -f $< $@",
        "//conditions:default": "touch $@",  # Placeholder for Unix
    }),
)

# Shared library target
cc_shared_library(
    name = "x10",
    shared_lib_name = select({
        "@platforms//os:macos": "libx10.dylib",
        "@platforms//os:windows": "x10.dll",
        "//conditions:default": "libx10.so",
    }),
    deps = [
        "//swift_bindings:device_wrapper",
        "//swift_bindings:xla_tensor_wrapper",
        "//swift_bindings:xla_tensor_tf_ops",
        ":tensor",
    ],
)

# Alternative: cc_binary with linkshared for older Bazel versions
cc_binary(
    name = "libx10_shared",
    linkshared = True,
    linkstatic = False,
    linkopts = select({
        "@platforms//os:macos": [],
        "@platforms//os:windows": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script,$(location :tf_version_script.lds)",
        ],
    }),
    deps = [
        "//swift_bindings:device_wrapper",
        "//swift_bindings:xla_tensor_wrapper",
        "//swift_bindings:xla_tensor_tf_ops",
        ":tensor",
        ":tf_exported_symbols.lds",
        ":tf_version_script.lds",
    ],
)

# Export linker scripts
exports_files([
    "tf_exported_symbols.lds",
    "tf_version_script.lds",
])
