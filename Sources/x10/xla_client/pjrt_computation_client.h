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

#ifndef X10_XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_
#define X10_XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "xla/xla_client/cache.h"
#include "xla/xla_client/computation_client.h"
#include "xla/xla_client/debug_macros.h"
#include "xla/xla_client/device.h"
#include "xla/xla_client/metrics.h"
#include "xla/xla_client/util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/literal.h"
#include "xla/shape.h"

namespace xla {

// PJRT-based computation client that replaces the XRT-based implementation.
// PJRT (Platform-independent JIT Runtime) is the modern execution interface
// for XLA that provides better portability and performance.
class PjRtComputationClient : public ComputationClient,
                              public ComputationClient::TransferManager {
 public:
  // PJRT-specific data wrapper that holds a PjRtBuffer
  struct PjRtData : public Data {
    PjRtData(Device* device, Shape device_shape,
             std::unique_ptr<PjRtBuffer> buffer)
        : Data(device, std::move(device_shape)),
          buffer(std::move(buffer)) {}

    PjRtData(Device* device, Shape device_shape)
        : Data(device, std::move(device_shape)) {}

    OpaqueHandle GetOpaqueHandle() override {
      return reinterpret_cast<OpaqueHandle>(buffer.get());
    }

    void Assign(const Data& data) override;

    bool HasValue() const override { return buffer != nullptr; }

    std::unique_ptr<PjRtBuffer> buffer;
  };

  // PJRT-specific computation wrapper that holds a compiled PjRtLoadedExecutable
  struct PjRtComputation : public Computation {
    PjRtComputation(XlaComputation computation, ProgramShape program_shape,
                    std::vector<std::string> devices,
                    std::unique_ptr<PjRtLoadedExecutable> executable)
        : Computation(std::move(computation), std::move(program_shape),
                      std::move(devices)),
          executable(std::move(executable)) {}

    std::unique_ptr<PjRtLoadedExecutable> executable;
  };

  struct Options {
    std::string default_device;
    std::set<std::string> devices;
    // Platform to use: "cpu", "gpu", "tpu"
    std::string platform = "cpu";
  };

  explicit PjRtComputationClient(Options options);
  ~PjRtComputationClient() override;

  // TransferManager implementation
  std::vector<Literal> TransferFromServerImpl(
      absl::Span<const DataPtr> handles) override;

  // Compile computations for the given devices
  std::vector<ComputationPtr> Compile(const std::string& device,
                                      const std::vector<std::string>& devices,
                                      std::vector<CompileInstance> instances);

  // Execute a computation with the given arguments
  std::vector<DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const std::string& device, const ExecuteComputationOptions& options);

  // Execute replicated computations
  std::vector<std::vector<DataPtr>> ExecuteReplicated(
      const Computation& computation,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteReplicatedOptions& options);

  // Execute parallel computations
  std::vector<std::vector<DataPtr>> ExecuteParallel(
      absl::Span<const Computation* const> computations,
      const std::vector<std::vector<DataPtr>>& arguments,
      absl::Span<const std::string> devices,
      const ExecuteParallelOptions& options);

  // Execute chained operations
  std::vector<DataPtr> ExecuteChained(absl::Span<const ExecuteChainedOp> ops,
                                      const std::string& device);

  std::string GetDefaultDevice() const override;

  swift_xla::Device GetDefaultDeviceStruct() const override;

  size_t GetNumDevices() const;

  std::vector<std::string> GetLocalDevices() const;

  void SetRngSeed(size_t seed) override;

  std::map<std::string, Metric> GetMetrics() const override;

  // Get the underlying PJRT client
  PjRtClient* GetPjRtClient() const { return pjrt_client_.get(); }

  // Get a PJRT device by name
  PjRtDevice* GetPjRtDevice(const std::string& device_name) const;

 private:
  class PjRtDevice;  // Forward declaration

  // Initialize PJRT client and devices
  void InitializeClient();
  void InitializeDevices();

  // Transfer data to PJRT device
  std::vector<DataPtr> TransferToServerInternal(
      PjRtDevice* device, absl::Span<const TensorSource> tensors);

  // Create a PJRT buffer from a literal
  std::unique_ptr<PjRtBuffer> CreateBufferFromLiteral(
      const Literal& literal, PjRtDevice* device);

  Options options_;
  std::unique_ptr<PjRtClient> pjrt_client_;
  std::mutex lock_;
  std::atomic<size_t> rng_seed_;

  // Cache for compiled executables
  struct CompilationCacheKey {
    struct Hash {
      size_t operator()(const CompilationCacheKey& entry) const;
    };

    CompilationCacheKey(std::string device, std::string serialized_computation)
        : device(std::move(device)),
          serialized_computation(std::move(serialized_computation)) {}
    CompilationCacheKey() = default;
    CompilationCacheKey(CompilationCacheKey&&) = default;
    CompilationCacheKey& operator=(CompilationCacheKey&&) = default;
    bool operator==(const CompilationCacheKey& rhs) const {
      return device == rhs.device &&
             serialized_computation == rhs.serialized_computation;
    }

    std::string device;
    std::string serialized_computation;
  };

  util::Cache<CompilationCacheKey, Computation, CompilationCacheKey::Hash>
      compilation_cache_;
};

}  // namespace xla

#endif  // X10_XLA_CLIENT_PJRT_COMPUTATION_CLIENT_H_
