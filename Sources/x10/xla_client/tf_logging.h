/*
 * Copyright 2020 TensorFlow Authors
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

// Standalone logging header for OpenXLA migration
// Replaces tensorflow/core/platform/logging.h with absl logging

#ifndef XLA_CLIENT_TF_LOGGING_H_
#define XLA_CLIENT_TF_LOGGING_H_

#include <sstream>
#include <string>

#include "absl/log/log.h"
#include "absl/log/check.h"
#include "absl/base/optimization.h"
#include "xla/status.h"

namespace xla {
namespace internal {

// Logging macros that work with both OpenXLA and standalone builds
// These replace the TensorFlow-specific logging macros

#define TF_LOG(severity) LOG(severity)
#define TF_VLOG(level) VLOG(level)
#define TF_VLOG_IS_ON(level) VLOG_IS_ON(level)

// Prediction macros for branch optimization
#ifndef TF_PREDICT_FALSE
#define TF_PREDICT_FALSE(x) ABSL_PREDICT_FALSE(x)
#endif

#ifndef TF_PREDICT_TRUE
#define TF_PREDICT_TRUE(x) ABSL_PREDICT_TRUE(x)
#endif

// Attribute for functions that don't return
#ifndef TF_ATTRIBUTE_NORETURN
#if defined(__GNUC__)
#define TF_ATTRIBUTE_NORETURN __attribute__((noreturn))
#elif defined(_MSC_VER)
#define TF_ATTRIBUTE_NORETURN __declspec(noreturn)
#else
#define TF_ATTRIBUTE_NORETURN
#endif
#endif

struct ErrorSink : public std::basic_ostringstream<char> {};

class ErrorGenerator {
 public:
  ErrorGenerator(const char* file, int line) : file_(file), line_(line) {}

  // Use a dummy & operator as it has lower precedence WRT the streaming
  // operator, and hence allows collecting user error messages before we finally
  // throw.
  TF_ATTRIBUTE_NORETURN void operator&(
      const std::basic_ostream<char>& oss) const;

 private:
  const char* file_ = nullptr;
  int line_ = 0;
};

#define TF_ERROR_STREAM()                               \
  ::xla::internal::ErrorGenerator(__FILE__, __LINE__) & \
      ::xla::internal::ErrorSink()

#define TF_CHECK(condition)              \
  while (TF_PREDICT_FALSE(!(condition))) \
  TF_ERROR_STREAM() << "Check failed: " #condition " "

// Check macros using absl equivalents
#define TF_CHECK_EQ(val1, val2) CHECK_EQ(val1, val2)
#define TF_CHECK_NE(val1, val2) CHECK_NE(val1, val2)
#define TF_CHECK_LE(val1, val2) CHECK_LE(val1, val2)
#define TF_CHECK_LT(val1, val2) CHECK_LT(val1, val2)
#define TF_CHECK_GE(val1, val2) CHECK_GE(val1, val2)
#define TF_CHECK_GT(val1, val2) CHECK_GT(val1, val2)

#define TF_CHECK_NOTNULL(val) CHECK(val != nullptr)

}  // namespace internal
}  // namespace xla

#endif  // XLA_CLIENT_TF_LOGGING_H_
