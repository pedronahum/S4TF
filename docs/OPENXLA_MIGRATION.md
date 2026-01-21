# X10 OpenXLA Migration Guide

This document describes the migration of X10 from TensorFlow's XLA to standalone OpenXLA.

## Overview

The X10 tensor library has been migrated from using XLA through TensorFlow to using the standalone OpenXLA project. This migration provides:

1. **Cleaner dependencies**: No longer requires full TensorFlow build
2. **Modern runtime**: Uses PJRT instead of deprecated XRT
3. **Better portability**: Standalone XLA is easier to integrate
4. **Active development**: OpenXLA is actively maintained by Google and partners

## Key Changes

### 1. Include Path Changes

All XLA includes have been updated from TensorFlow paths to OpenXLA paths:

| Old Path | New Path |
|----------|----------|
| `tensorflow/compiler/xla/...` | `xla/...` |
| `tensorflow/compiler/xla/client/...` | `xla/client/...` |
| `tensorflow/compiler/xla/service/...` | `xla/service/...` |
| `tensorflow/compiler/xla/xla_client/...` | `xla/xla_client/...` |

### 2. Runtime Migration: XRT → PJRT

The XRT (XLA Runtime) based computation client has been replaced with PJRT (Platform-independent JIT Runtime):

**Old (XRT):**
- `xrt_computation_client.h/.cc`
- `xrt_session.h/.cc`
- `xrt_session_cache.h/.cc`
- `xrt_local_service.h/.cc`

**New (PJRT):**
- `pjrt_computation_client.h/.cc`
- Uses `xla/pjrt/pjrt_client.h`
- Uses `xla/pjrt/pjrt_executable.h`

### 3. Build System

The build system has been updated to use standalone OpenXLA:

- **WORKSPACE.openxla**: Traditional WORKSPACE-based build
- **MODULE.bazel**: Modern bzlmod-based build
- Updated BUILD files use `@xla//xla/...` dependencies

### 4. Compatibility Layer

A compatibility layer (`tf_compat.h`, `openxla_compat.h`) provides:
- TensorFlow type definitions (Padding, MirrorPadMode, TensorFormat)
- Profiler TraceMe compatibility
- GTL ArraySlice → absl::Span mapping

## Building with OpenXLA

### Prerequisites

- Bazel 6.0 or later
- C++17 compatible compiler
- For GPU: CUDA 11.8+ or ROCm 5.0+
- For TPU: Cloud TPU access

### Build Commands

```bash
# CPU-only build
bazel build --config=openxla //xla_tensor:x10

# GPU build
bazel build --config=openxla --config=cuda //xla_tensor:x10

# TPU build
bazel build --config=openxla --config=tpu //xla_tensor:x10
```

### Environment Variables

- `XLA_PLATFORM`: Set to "cpu", "gpu", or "tpu" (default: "cpu")
- `XLA_DEFAULT_DEVICE`: Override default device (e.g., "GPU:0")
- `XLA_CPU_DEVICE_COUNT`: Number of CPU devices to create

## API Changes

### Device Initialization

**Old (XRT):**
```cpp
// XRT required explicit service initialization
xla::XrtLocalService::Start();
auto client = xla::ComputationClient::Create();
```

**New (PJRT):**
```cpp
// PJRT handles initialization automatically
auto client = xla::ComputationClient::Create();  // Uses PJRT internally
```

### Compilation

The compilation API remains largely the same, but now uses PJRT's compilation pipeline:

```cpp
auto computation = client->Compile(xla_computation, device, devices, output_shape);
```

### Execution

Execution also maintains the same interface:

```cpp
auto results = device->ExecuteComputation(computation, arguments, options);
```

## Migration Checklist

- [x] Update include paths from `tensorflow/compiler/xla/` to `xla/`
- [x] Create PJRT-based computation client
- [x] Update BUILD files with OpenXLA dependencies
- [x] Create TF compatibility headers
- [x] Create WORKSPACE.openxla and MODULE.bazel
- [ ] Test CPU backend
- [ ] Test GPU backend
- [ ] Test TPU backend
- [ ] Update Swift bindings if needed
- [ ] Performance benchmarking

## Known Issues

1. **tf2xla utilities**: Some utilities from `tensorflow/compiler/tf2xla/` don't have direct equivalents in OpenXLA. These have been either ported or stubbed.

2. **XRT-specific features**: Some XRT-specific features (like XRT session caching) are handled differently in PJRT.

3. **Profiler integration**: TensorFlow profiler integration has been replaced with a simple TraceMe stub. For full profiling, integrate with OpenTelemetry or XLA's native profiling.

## Resources

- [OpenXLA Repository](https://github.com/openxla/xla)
- [PJRT Documentation](https://openxla.org/xla/pjrt_integration)
- [StableHLO](https://github.com/openxla/stablehlo)
- [OpenXLA Community](https://github.com/openxla/community)
