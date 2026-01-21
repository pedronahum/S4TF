/*
 * Copyright 2024 The TensorFlow Authors
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

// TensorFlow Compatibility Layer for OpenXLA Migration
// This header provides compatibility definitions for TensorFlow types
// that were used in the original X10 implementation but don't have
// direct equivalents in standalone OpenXLA.

#ifndef CX10_TF_COMPAT_H_
#define CX10_TF_COMPAT_H_

#include <string>

// ============================================================================
// Padding Types
// ============================================================================
// These were originally defined in tensorflow/core/util/padding.h

namespace tensorflow {

enum Padding {
  VALID = 1,     // No padding
  SAME = 2,      // Input and output layers have the same size
  EXPLICIT = 3,  // Padding is explicitly specified
};

inline std::string PaddingToString(Padding padding) {
  switch (padding) {
    case VALID: return "VALID";
    case SAME: return "SAME";
    case EXPLICIT: return "EXPLICIT";
    default: return "UNKNOWN";
  }
}

}  // namespace tensorflow

// ============================================================================
// Mirror Pad Mode
// ============================================================================
// These were originally defined in tensorflow/core/util/mirror_pad_mode.h

namespace tensorflow {

enum class MirrorPadMode {
  REFLECT = 1,
  SYMMETRIC = 2,
};

inline std::string MirrorPadModeToString(MirrorPadMode mode) {
  switch (mode) {
    case MirrorPadMode::REFLECT: return "REFLECT";
    case MirrorPadMode::SYMMETRIC: return "SYMMETRIC";
    default: return "UNKNOWN";
  }
}

}  // namespace tensorflow

// ============================================================================
// Tensor Format
// ============================================================================
// These were originally defined in tensorflow/core/util/tensor_format.h

namespace tensorflow {

enum TensorFormat {
  FORMAT_NHWC = 0,  // Batch, Height, Width, Channels
  FORMAT_NCHW = 1,  // Batch, Channels, Height, Width
  FORMAT_NCHW_VECT_C = 2,  // Vectorized channels
};

inline std::string TensorFormatToString(TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC: return "NHWC";
    case FORMAT_NCHW: return "NCHW";
    case FORMAT_NCHW_VECT_C: return "NCHW_VECT_C";
    default: return "UNKNOWN";
  }
}

// Get the dimension index for various components based on format
inline int GetTensorBatchDimIndex(int num_dims, TensorFormat format) {
  return 0;  // Batch is always first
}

inline int GetTensorFeatureDimIndex(int num_dims, TensorFormat format) {
  switch (format) {
    case FORMAT_NHWC: return num_dims - 1;
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C: return 1;
    default: return -1;
  }
}

inline int GetTensorSpatialDimIndex(int num_dims, TensorFormat format, int dim) {
  switch (format) {
    case FORMAT_NHWC: return dim + 1;
    case FORMAT_NCHW:
    case FORMAT_NCHW_VECT_C: return dim + 2;
    default: return -1;
  }
}

}  // namespace tensorflow

// ============================================================================
// Profiler TraceMe Compatibility
// ============================================================================
// Original: tensorflow/core/profiler/lib/traceme.h
// For OpenXLA, we provide a no-op implementation or use absl tracing

namespace tensorflow {
namespace profiler {

class TraceMe {
 public:
  explicit TraceMe(const char* name) : name_(name) {
    // In a full implementation, this would integrate with XLA profiling
    // or OpenTelemetry tracing
  }

  explicit TraceMe(const std::string& name) : name_(name) {}

  ~TraceMe() {
    // End trace span
  }

  // Disable copy
  TraceMe(const TraceMe&) = delete;
  TraceMe& operator=(const TraceMe&) = delete;

  // Allow move
  TraceMe(TraceMe&&) = default;
  TraceMe& operator=(TraceMe&&) = default;

 private:
  std::string name_;
};

}  // namespace profiler
}  // namespace tensorflow

// ============================================================================
// GTL ArraySlice Compatibility
// ============================================================================
// Original: tensorflow/core/lib/gtl/array_slice.h
// For OpenXLA, use absl::Span instead

#include "absl/types/span.h"

namespace tensorflow {
namespace gtl {

template <typename T>
using ArraySlice = absl::Span<T>;

template <typename T>
using MutableArraySlice = absl::Span<T>;

}  // namespace gtl
}  // namespace tensorflow

#endif  // CX10_TF_COMPAT_H_
