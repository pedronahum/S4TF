# Release Notes

## Version 0.13.0 - OpenXLA Migration

This release represents a major architectural change in the X10 tensor library,
migrating from TensorFlow's bundled XLA to standalone OpenXLA. Despite the
significant backend changes, **the Swift API remains completely unchanged**.

### Highlights

- **OpenXLA Migration**: X10 now uses standalone [OpenXLA](https://github.com/openxla/xla)
  instead of TensorFlow's bundled XLA
- **PJRT Runtime**: Replaced deprecated XRT with modern PJRT (Platform-independent JIT Runtime)
- **Pre-built Plugin Support**: Use PJRT plugins from JAX/TensorFlow without building from source
- **TensorFlow 2.20 Compatibility**: Updated compatibility layer for TF 2.20
- **Simplified Dependencies**: Cleaner build without full TensorFlow dependency

### Breaking Changes

**None for Swift users.** The Swift API is completely unchanged. This migration
only affects the C++ backend implementation.

### New Features

#### Pre-built PJRT Plugin Support

You can now use pre-built PJRT plugins from Python packages instead of building
the C++ library from source:

```bash
# Install JAX (provides PJRT plugins)
pip install jax[cpu]       # For CPU
pip install jax[cuda12]    # For CUDA GPU

# X10 automatically detects and uses the plugin
export XLA_PLATFORM=cpu    # or "cuda" for GPU
```

A helper script is included to find available plugins:

```bash
python scripts/find_pjrt_plugin.py
python scripts/find_pjrt_plugin.py cuda
```

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `XLA_PLATFORM` | Target platform: cpu, cuda, gpu, tpu, rocm | cpu |
| `XLA_USE_PREBUILT_PJRT` | Enable pre-built plugin loading | true |
| `PJRT_CPU_LIBRARY_PATH` | Override CPU plugin path | (auto-detected) |
| `PJRT_CUDA_LIBRARY_PATH` | Override CUDA plugin path | (auto-detected) |
| `PJRT_GPU_LIBRARY_PATH` | Override GPU plugin path | (auto-detected) |
| `PJRT_TPU_LIBRARY_PATH` | Override TPU plugin path | (auto-detected) |
| `XLA_CPU_DEVICE_COUNT` | Number of CPU devices | 1 |

### Architecture Changes (C++ Backend)

#### Include Path Migration

All XLA includes updated from TensorFlow paths to OpenXLA paths:

| Old Path | New Path |
|----------|----------|
| `tensorflow/compiler/xla/...` | `xla/...` |
| `tensorflow/compiler/xla/client/...` | `xla/client/...` |
| `tensorflow/compiler/xla/service/...` | `xla/service/...` |

#### Runtime Migration: XRT → PJRT

**Removed (XRT):**
- `xrt_computation_client.h/.cc`
- `xrt_session.h/.cc`
- `xrt_session_cache.h/.cc`
- `xrt_local_service.h/.cc`

**Added (PJRT):**
- `pjrt_computation_client.h/.cc` - New PJRT-based computation client
- `openxla_compat.h` - OpenXLA compatibility layer
- `tf_compat.h` - TensorFlow type compatibility (Padding, TensorFormat, etc.)

#### Build System Updates

- New `WORKSPACE.openxla` for traditional Bazel builds
- New `MODULE.bazel` for modern bzlmod builds
- Updated BUILD files use `@xla//xla/...` dependencies
- Support for CPU, CUDA, ROCm, and TPU backends

### New Files

| File | Description |
|------|-------------|
| `Sources/x10/xla_client/pjrt_computation_client.h` | PJRT client header |
| `Sources/x10/xla_client/pjrt_computation_client.cc` | PJRT client implementation |
| `Sources/x10/xla_client/openxla_compat.h` | OpenXLA compatibility layer |
| `Sources/x10/xla_client/tf_compat.h` | TensorFlow type compatibility |
| `Sources/CX10/tf_compat.h` | CX10 TensorFlow compatibility |
| `scripts/find_pjrt_plugin.py` | Helper to find PJRT plugins |
| `WORKSPACE.openxla` | OpenXLA Bazel workspace |
| `MODULE.bazel` | bzlmod configuration |
| `docs/OPENXLA_MIGRATION.md` | Migration guide |

### Updated Files

| File | Changes |
|------|---------|
| `Sources/x10/xla_client/BUILD` | OpenXLA dependencies, PJRT targets |
| `Sources/x10/xla_tensor/BUILD` | OpenXLA dependencies |
| `Sources/CX10/BUILD` | OpenXLA/PJRT dependencies |
| `Sources/x10/xla_client/tf_logging.h` | Use absl logging |
| `CMakeLists.txt` | TF 2.4 → 2.20 version update |
| `README.md` | OpenXLA documentation |
| `Documentation/Development.md` | Build instructions |
| 165+ C++ files | Include path updates |

### Build Instructions

#### Option 1: Using Pre-built Plugins (Easiest)

```bash
# Install JAX
pip install jax[cpu]       # CPU only
pip install jax[cuda12]    # CUDA GPU

# Find plugin (optional - auto-detected)
python scripts/find_pjrt_plugin.py

# Set platform
export XLA_PLATFORM=cpu
```

#### Option 2: Building from Source

```bash
# CPU build
bazel build --config=openxla //xla_tensor:x10

# CUDA GPU build
bazel build --config=openxla --config=cuda //xla_tensor:x10

# TPU build
bazel build --config=openxla --config=tpu //xla_tensor:x10
```

### Supported Platforms

| Platform | Build from Source | Pre-built Plugin |
|----------|------------------|------------------|
| CPU | Yes | Yes (JAX, TensorFlow) |
| CUDA GPU | Yes | Yes (JAX, TensorFlow) |
| ROCm GPU | Yes | Yes (JAX) |
| TPU | Yes | Yes (JAX) |

### Migration Guide

For detailed migration information, see [docs/OPENXLA_MIGRATION.md](docs/OPENXLA_MIGRATION.md).

### Known Issues

1. **tf2xla utilities**: Some TensorFlow utilities have been stubbed or ported
2. **XRT-specific features**: Session caching handled differently in PJRT
3. **Profiler integration**: Replaced with simple TraceMe stub

### Testing Status

- [x] Include path updates verified
- [x] PJRT client implementation complete
- [x] Pre-built plugin loading implemented
- [x] Documentation updated
- [ ] CPU backend testing
- [ ] GPU backend testing
- [ ] TPU backend testing
- [ ] Performance benchmarking

### Dependencies

| Dependency | Version | Notes |
|------------|---------|-------|
| OpenXLA | Latest | github.com/openxla/xla |
| Abseil | 20230125+ | Logging, strings, containers |
| Bazel | 6.0+ | Build system |
| CUDA | 11.8+ | Optional, for GPU |
| cuDNN | 8.6+ | Optional, for GPU |
| ROCm | 5.0+ | Optional, for AMD GPU |
| Python | 3.8+ | For pre-built plugins |
| JAX | 0.4+ | Optional, provides PJRT plugins |
| TensorFlow | 2.20+ | Optional, provides PJRT plugins |

### Contributors

Thanks to all contributors who made this migration possible.

### Resources

- [OpenXLA Repository](https://github.com/openxla/xla)
- [PJRT Documentation](https://openxla.org/xla/pjrt_integration)
- [StableHLO](https://github.com/openxla/stablehlo)
- [OpenXLA Community](https://github.com/openxla/community)
- [Swift for TensorFlow](https://github.com/tensorflow/swift)
