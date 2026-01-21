licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

# Device wrapper - provides Swift-to-C++ device abstraction
cc_library(
    name = "device_wrapper",
    srcs = ["device_wrapper.cc"],
    hdrs = ["device_wrapper.h"],
    deps = [
        # Use PJRT-based client for OpenXLA
        "//xla_client:pjrt_computation_client",
        "//xla_client:xla_client_util",
        "//xla_tensor:tensor",
        "@xla//xla:types",
    ],
)

# XLA tensor wrapper - main C interface for Swift bindings
cc_library(
    name = "xla_tensor_wrapper",
    srcs = [
        "functional_while.cc",
        "xla_tensor_wrapper.cc",
        "xla_tensor_ops_wrapper.cc",
        "xla_tensor_ops_wrapper_generated.cc.inc",
    ],
    hdrs = [
        "xla_tensor_wrapper.h",
        "tf_compat.h",
    ],
    deps = [
        ":device_wrapper",
        "//xla_tensor:tensor",
        "//xla_client:xla_client_util",
        "//xla_client:tf_compat",
        "@xla//xla:shape_util",
        "@xla//xla:types",
        "@com_google_absl//absl/types:span",
    ],
)

# TensorFlow ops wrapper for XLA tensor
cc_library(
    name = "xla_tensor_tf_ops",
    srcs = ["xla_tensor_tf_ops.cc"],
    hdrs = ["xla_tensor_tf_ops.h"],
    deps = [
        ":xla_tensor_wrapper",
        "//xla_tensor:tensor",
        "@xla//xla/client/lib:pooling",
        "@xla//xla:shape_util",
        "@com_google_absl//absl/types:span",
    ],
)
