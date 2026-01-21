# X10 OpenXLA Migration Guide

This document describes the migration of X10 from TensorFlow's XLA to standalone OpenXLA.

## Overview

The X10 tensor library has been migrated from using XLA through TensorFlow to using the standalone OpenXLA project. This migration provides:

1. **Cleaner dependencies**: No longer requires full TensorFlow build
2. **Modern runtime**: Uses PJRT instead of deprecated XRT
3. **Better portability**: Standalone XLA is easier to integrate
4. **Active development**: OpenXLA is actively maintained by Google and partners

## Swift API Stability

**The Swift API is completely unchanged.** This migration only affects the C++ backend implementation. Your existing Swift code will work without any modifications:

```swift
import TensorFlow

// All existing APIs work exactly as before
let device = Device(kind: .GPU, ordinal: 0, backend: .XLA)
let tensor = Tensor<Float>(randomNormal: [1024, 1024], on: device)
let result = matmul(tensor, tensor)

// LazyTensorBarrier still works - now uses PJRT internally
LazyTensorBarrier()
```

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Your Swift Code                          │
│                      (unchanged)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Swift TensorFlow APIs                          │
│         (Tensor.swift, Layer.swift, etc.)                   │
│                      (unchanged)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Swift X10 Bindings                             │
│         (Device.swift, XLATensor.swift)                     │
│                      (unchanged)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              C Interface (FFI Boundary)                     │
│    (device_wrapper.h, xla_tensor_wrapper.h)                 │
│                      (unchanged)                            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              C++ Implementation                             │
│    (device_wrapper.cc, xla_tensor_wrapper.cc)               │
│                      (UPDATED)                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              XLA Client Layer                               │
│         (pjrt_computation_client.cc)                        │
│                       (NEW)                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    OpenXLA / PJRT                           │
│              (Replaces TensorFlow XLA/XRT)                  │
└─────────────────────────────────────────────────────────────┘
```

### What Changed vs What Stayed the Same

| Component | Status | Notes |
|-----------|--------|-------|
| Swift user code | ✅ Unchanged | No changes needed |
| Swift TensorFlow APIs | ✅ Unchanged | Same Tensor, Layer, etc. |
| Swift X10 bindings | ✅ Unchanged | Device.swift, XLATensor.swift |
| C interface | ✅ Unchanged | device_wrapper.h, xla_tensor_wrapper.h |
| C++ implementation | ⚠️ Updated | Internal changes for PJRT |
| XLA client | 🆕 New | PJRT replaces XRT |
| Build system | ⚠️ Updated | Uses standalone OpenXLA |

## Key Changes (C++ Backend Only)

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

## Using Pre-built PJRT Plugins

Instead of building from source, you can leverage pre-built PJRT plugins from
Python packages like JAX or TensorFlow. This is the easiest way to get started.

### Quick Start

```bash
# Install JAX (provides PJRT plugins)
pip install jax[cpu]       # For CPU
pip install jax[cuda12]    # For CUDA GPU

# X10 will automatically find and use JAX's PJRT plugin
export XLA_PLATFORM=cpu    # or "cuda" for GPU
```

### Finding Available Plugins

Use the included helper script:

```bash
# Find all available plugins
python scripts/find_pjrt_plugin.py

# Find plugin for specific platform
python scripts/find_pjrt_plugin.py cpu
python scripts/find_pjrt_plugin.py cuda
python scripts/find_pjrt_plugin.py tpu
```

### Plugin Environment Variables

| Variable | Description |
|----------|-------------|
| `XLA_USE_PREBUILT_PJRT` | Enable/disable plugin loading (default: true) |
| `PJRT_CPU_LIBRARY_PATH` | Override CPU plugin path |
| `PJRT_CUDA_LIBRARY_PATH` | Override CUDA plugin path |
| `PJRT_GPU_LIBRARY_PATH` | Override GPU plugin path |
| `PJRT_TPU_LIBRARY_PATH` | Override TPU plugin path |
| `PJRT_PLUGIN_LIBRARY_PATH` | Generic plugin override |

### Supported Python Packages

| Package | CPU | CUDA | ROCm | TPU |
|---------|-----|------|------|-----|
| JAX (`jax[cpu]`) | Yes | - | - | - |
| JAX (`jax[cuda12]`) | Yes | Yes | - | - |
| JAX (`jax[rocm]`) | Yes | - | Yes | - |
| JAX (`jax[tpu]`) | Yes | - | - | Yes |
| TensorFlow | Yes | - | - | - |
| TensorFlow (`tensorflow[and-cuda]`) | Yes | Yes | - | - |

### How Plugin Loading Works

When X10 initializes, it searches for PJRT plugins in this order:

1. Environment variable override (e.g., `PJRT_CPU_LIBRARY_PATH`)
2. Python site-packages (JAX, TensorFlow installations)
3. `LD_LIBRARY_PATH` directories
4. Fall back to compiled-in backends (if available)

To disable automatic plugin loading and force compiled-in backends:

```bash
export XLA_USE_PREBUILT_PJRT=false
```

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
- [x] Swift bindings verified unchanged (C interface stable)
- [x] Add pre-built PJRT plugin support
- [x] Create plugin discovery helper script
- [x] Update documentation (README, Development.md)
- [x] Create release notes
- [ ] Test CPU backend
- [ ] Test GPU backend
- [ ] Test TPU backend
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
