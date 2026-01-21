licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load(
    "@xla//xla:xla.bzl",
    "xla_cc_library",
)

# Proto library for mesh service
proto_library(
    name = "mesh_service_proto",
    srcs = ["mesh_service.proto"],
    deps = [
        "@xla//xla:xla_data_proto",
    ],
)

cc_proto_library(
    name = "mesh_service_cc_proto",
    deps = [":mesh_service_proto"],
)

# Core XLA client utilities (backend-agnostic)
cc_library(
    name = "xla_client_util",
    srcs = [
        "device.cc",
        "env_vars.cc",
        "metrics.cc",
        "metrics_reader.cc",
        "multi_wait.cc",
        "sys_util.cc",
        "tf_logging.cc",
        "thread_pool.cc",
        "triggered_task.cc",
        "util.cc",
        "xla_util.cc",
    ],
    hdrs = [
        "async_task.h",
        "cache.h",
        "debug_macros.h",
        "device.h",
        "env_vars.h",
        "metrics.h",
        "metrics_reader.h",
        "multi_wait.h",
        "openxla_compat.h",
        "sys_util.h",
        "tf_logging.h",
        "thread_pool.h",
        "triggered_task.h",
        "types.h",
        "unique.h",
        "util.h",
        "xla_util.h",
    ],
    deps = [
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla/client:xla_computation",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# Computation client base interface
cc_library(
    name = "computation_client",
    srcs = ["computation_client.cc"],
    hdrs = ["computation_client.h"],
    deps = [
        ":xla_client_util",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@xla//xla:types",
        "@xla//xla/client:xla_computation",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# PJRT-based computation client (recommended for OpenXLA)
cc_library(
    name = "pjrt_computation_client",
    srcs = [
        "local_device.cc",
        "pjrt_computation_client.cc",
    ],
    hdrs = [
        "local_device.h",
        "openxla_compat.h",
        "pjrt_computation_client.h",
    ],
    defines = [
        "USE_PJRT=1",
        "XLA_CPU_BACKEND=1",
    ],
    deps = [
        ":computation_client",
        ":xla_client_util",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla/client:xla_computation",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/service:hlo_proto_cc",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/log:check",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# GPU-enabled PJRT client (optional)
cc_library(
    name = "pjrt_computation_client_gpu",
    srcs = [
        "local_device.cc",
        "pjrt_computation_client.cc",
    ],
    hdrs = [
        "local_device.h",
        "openxla_compat.h",
        "pjrt_computation_client.h",
    ],
    defines = [
        "USE_PJRT=1",
        "XLA_CPU_BACKEND=1",
        "XLA_GPU_BACKEND=1",
    ],
    deps = [
        ":computation_client",
        ":xla_client_util",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla/client:xla_computation",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/pjrt/gpu:gpu_client",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# TPU-enabled PJRT client (optional)
cc_library(
    name = "pjrt_computation_client_tpu",
    srcs = [
        "local_device.cc",
        "pjrt_computation_client.cc",
    ],
    hdrs = [
        "local_device.h",
        "openxla_compat.h",
        "pjrt_computation_client.h",
    ],
    defines = [
        "USE_PJRT=1",
        "XLA_CPU_BACKEND=1",
        "XLA_TPU_BACKEND=1",
    ],
    deps = [
        ":computation_client",
        ":xla_client_util",
        "@xla//xla:literal",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla/client:xla_computation",
        "@xla//xla/pjrt:pjrt_api",
        "@xla//xla/pjrt:pjrt_client",
        "@xla//xla/pjrt:pjrt_executable",
        "@xla//xla/pjrt/cpu:cpu_client",
        "@xla//xla/pjrt/tpu:tpu_client",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/log",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# Legacy XRT-based computation client (deprecated, for backwards compatibility)
cc_library(
    name = "xrt_computation_client",
    srcs = [
        "computation_client.cc",
        "device.cc",
        "env_vars.cc",
        "local_device.cc",
        "mesh_service.cc",
        "metrics.cc",
        "metrics_reader.cc",
        "multi_wait.cc",
        "nccl_distributed.cc",
        "sys_util.cc",
        "tf_logging.cc",
        "thread_pool.cc",
        "triggered_task.cc",
        "util.cc",
        "xla_util.cc",
        "xrt_computation_client.cc",
        "xrt_local_service.cc",
        "xrt_session.cc",
        "xrt_session_cache.cc",
    ],
    hdrs = [
        "async_task.h",
        "cache.h",
        "computation_client.h",
        "debug_macros.h",
        "device.h",
        "env_vars.h",
        "local_device.h",
        "mesh_service.h",
        "metrics.h",
        "metrics_reader.h",
        "multi_wait.h",
        "nccl_distributed.h",
        "sys_util.h",
        "tf_logging.h",
        "thread_pool.h",
        "triggered_task.h",
        "types.h",
        "unique.h",
        "util.h",
        "xla_util.h",
        "xrt_computation_client.h",
        "xrt_local_service.h",
        "xrt_session.h",
        "xrt_session_cache.h",
    ],
    defines = ["USE_XRT=1"],
    tags = ["deprecated"],
    deps = [
        ":mesh_service_cc_proto",
        "@xla//xla:literal_util",
        "@xla//xla:shape_util",
        "@xla//xla:status",
        "@xla//xla:statusor",
        "@xla//xla:types",
        "@xla//xla:util",
        "@xla//xla:xla_data_proto_cc",
        "@xla//xla/client:client_library",
        "@xla//xla/client:xla_computation",
        "@xla//xla/service:hlo",
        "@xla//xla/service:hlo_proto_cc",
        "@xla//xla/service:platform_util",
        # Note: XRT dependencies would need to be provided separately
        # or migrated from TensorFlow
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/numeric:int128",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

# Fake computation client for testing
cc_library(
    name = "fake_computation_client",
    srcs = ["fake_computation_client.cc"],
    hdrs = ["fake_computation_client.h"],
    deps = [
        ":computation_client",
        ":xla_client_util",
        "@xla//xla:literal",
        "@xla//xla:shape_util",
        "@com_google_absl//absl/types:span",
    ],
)
