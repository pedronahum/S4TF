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

// TensorFlow Compatibility Layer for XLA Client
// Provides compatibility definitions for TF utilities used in X10

#ifndef XLA_CLIENT_TF_COMPAT_H_
#define XLA_CLIENT_TF_COMPAT_H_

#include <string>
#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "absl/hash/hash.h"

// ============================================================================
// Hashing Utilities
// ============================================================================
// Replace tensorflow::Hash64 with absl-based implementation

namespace tensorflow {

inline uint64_t Hash64(const char* data, size_t n, uint64_t seed) {
  // Use absl's hashing
  return absl::HashOf(absl::string_view(data, n), seed);
}

inline uint64_t Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

}  // namespace tensorflow

// ============================================================================
// Profiler TraceMe Compatibility
// ============================================================================
// Original: tensorflow/core/profiler/lib/traceme.h
// Provides a no-op implementation for builds without TF profiler

namespace tensorflow {
namespace profiler {

class TraceMe {
 public:
  explicit TraceMe(const char* name) : name_(name) {}
  explicit TraceMe(const std::string& name) : name_(name) {}

  template <typename NameT>
  explicit TraceMe(NameT&& name) : name_(std::forward<NameT>(name)) {}

  ~TraceMe() = default;

  // Disable copy
  TraceMe(const TraceMe&) = delete;
  TraceMe& operator=(const TraceMe&) = delete;

  // Allow move
  TraceMe(TraceMe&&) = default;
  TraceMe& operator=(TraceMe&&) = default;

  // Check if tracing is active (always false for stub)
  static bool Active() { return false; }

 private:
  std::string name_;
};

}  // namespace profiler
}  // namespace tensorflow

// ============================================================================
// GTL Compatibility
// ============================================================================
// Replace tensorflow::gtl types with absl equivalents

namespace tensorflow {
namespace gtl {

template <typename T>
using ArraySlice = absl::Span<T>;

template <typename T, size_t N>
using InlinedVector = absl::InlinedVector<T, N>;

}  // namespace gtl
}  // namespace tensorflow

// ============================================================================
// Error Utilities Compatibility
// ============================================================================

namespace tensorflow {
namespace errors {

// Common error creation functions - map to absl::Status
inline absl::Status InvalidArgument(const std::string& message) {
  return absl::InvalidArgumentError(message);
}

inline absl::Status NotFound(const std::string& message) {
  return absl::NotFoundError(message);
}

inline absl::Status Internal(const std::string& message) {
  return absl::InternalError(message);
}

inline absl::Status Unimplemented(const std::string& message) {
  return absl::UnimplementedError(message);
}

inline absl::Status FailedPrecondition(const std::string& message) {
  return absl::FailedPreconditionError(message);
}

inline absl::Status ResourceExhausted(const std::string& message) {
  return absl::ResourceExhaustedError(message);
}

}  // namespace errors
}  // namespace tensorflow

// ============================================================================
// Device Name Utilities
// ============================================================================
// Simplified version of tensorflow/core/util/device_name_utils.h

namespace tensorflow {

struct DeviceNameUtils {
  struct ParsedName {
    std::string type;
    int id = 0;
    bool has_type = false;
    bool has_id = false;
  };

  static bool ParseFullName(const std::string& fullname, ParsedName* parsed) {
    // Simple parsing for device names like "CPU:0" or "GPU:1"
    auto colon_pos = fullname.rfind(':');
    if (colon_pos != std::string::npos) {
      parsed->type = fullname.substr(0, colon_pos);
      parsed->has_type = true;
      try {
        parsed->id = std::stoi(fullname.substr(colon_pos + 1));
        parsed->has_id = true;
      } catch (...) {
        parsed->has_id = false;
      }
    } else {
      parsed->type = fullname;
      parsed->has_type = true;
    }
    return parsed->has_type;
  }

  static std::string ParsedNameToString(const ParsedName& parsed) {
    if (parsed.has_id) {
      return parsed.type + ":" + std::to_string(parsed.id);
    }
    return parsed.type;
  }
};

}  // namespace tensorflow

#endif  // XLA_CLIENT_TF_COMPAT_H_
