/*
 * Copyright 2024 OpenXLA Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// OpenXLA Compatibility Header
// This header provides compatibility definitions for migrating from
// TensorFlow's XLA (tensorflow/compiler/xla) to OpenXLA (xla/).
//
// The migration involves:
// 1. Header path changes: tensorflow/compiler/xla/ -> xla/
// 2. Runtime migration: XRT -> PJRT
// 3. Build system updates: TF workspace -> standalone XLA workspace

#ifndef X10_XLA_CLIENT_OPENXLA_COMPAT_H_
#define X10_XLA_CLIENT_OPENXLA_COMPAT_H_

// ============================================================================
// OpenXLA Include Path Compatibility
// ============================================================================
// The following macros can be used during migration to support both old
// (TensorFlow-based) and new (OpenXLA standalone) include paths.

#define OPENXLA_MIGRATION 1

// ============================================================================
// Type Compatibility
// ============================================================================

// XLA types that may have moved or changed
#include "xla/types.h"
#include "xla/shape.h"
#include "xla/literal.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {

// Compatibility type aliases for migration
// These help code that was written for older TF/XLA versions

// int64 is used extensively in X10, ensure it's available
using int64 = int64_t;
using uint64 = uint64_t;
using int32 = int32_t;
using uint32 = uint32_t;

}  // namespace xla

// ============================================================================
// PJRT Migration Helpers
// ============================================================================

namespace xla {
namespace pjrt_compat {

// Helper to determine if we're using PJRT or XRT backend
constexpr bool UsingPjRt() {
#ifdef USE_PJRT
  return true;
#else
  return false;
#endif
}

// Helper to get the platform name from environment
inline std::string GetPlatformFromEnv() {
  const char* platform = std::getenv("XLA_PLATFORM");
  if (platform != nullptr) {
    return std::string(platform);
  }
  // Default to CPU if not specified
  return "cpu";
}

}  // namespace pjrt_compat
}  // namespace xla

// ============================================================================
// Logging Compatibility
// ============================================================================

// TensorFlow logging macros may need to be replaced with standard logging
// For OpenXLA, we use absl logging

#ifndef TF_LOG
#include "absl/log/log.h"
#define TF_LOG(severity) LOG(severity)
#endif

#ifndef TF_CHECK
#include "absl/log/check.h"
#define TF_CHECK(condition) CHECK(condition)
#endif

// ============================================================================
// Build Configuration
// ============================================================================

// Define backend availability based on build configuration
#if defined(__APPLE__)
// macOS typically only has CPU backend
#ifndef XLA_CPU_BACKEND
#define XLA_CPU_BACKEND 1
#endif
#elif defined(_WIN32)
// Windows may have CPU and GPU
#ifndef XLA_CPU_BACKEND
#define XLA_CPU_BACKEND 1
#endif
#else
// Linux typically has all backends available
#ifndef XLA_CPU_BACKEND
#define XLA_CPU_BACKEND 1
#endif
// GPU and TPU are optional
#endif

#endif  // X10_XLA_CLIENT_OPENXLA_COMPAT_H_
