// Copyright 2020 TensorFlow Authors
// Copyright 2024 OpenXLA Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/xla_client/pjrt_computation_client.h"

#include <cstdlib>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xla/xla_client/env_vars.h"
#include "xla/xla_client/sys_util.h"
#include "xla/xla_client/thread_pool.h"
#include "xla/xla_client/util.h"
#include "xla/xla_client/xla_util.h"
#include "xla/xla_client/local_device.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/client/xla_computation.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/util.h"

// Platform-specific PJRT client headers
#ifdef XLA_CPU_BACKEND
#include "xla/pjrt/cpu/cpu_client.h"
#endif

#ifdef XLA_GPU_BACKEND
#include "xla/pjrt/gpu/gpu_client.h"
#endif

#ifdef XLA_TPU_BACKEND
#include "xla/pjrt/tpu/tpu_client.h"
#endif

namespace xla {

using DataPtr = ComputationClient::DataPtr;
using ComputationPtr = ComputationClient::ComputationPtr;
using TensorSource = ComputationClient::TensorSource;

// PJRT Device implementation
class PjRtComputationClient::PjRtDevice : public ComputationClient::Device {
 public:
  PjRtDevice(std::string name, PjRtComputationClient* client,
             xla::PjRtDevice* pjrt_device)
      : Device(name), client_(client), pjrt_device_(pjrt_device) {}

  DataPtr CreateDataPlaceholder(Shape shape) override {
    return std::make_shared<PjRtData>(this, std::move(shape));
  }

  std::vector<DataPtr> TransferToServer(
      absl::Span<const TensorSource> tensors) override {
    return client_->TransferToServerInternal(this, tensors);
  }

  std::vector<ComputationClient::ComputationPtr> Compile(
      const std::vector<std::string>& devices,
      std::vector<CompileInstance> instances) override {
    return client_->Compile(name(), devices, std::move(instances));
  }

  std::string ResourceDomain() const override {
    // For PJRT, the resource domain is the platform name
    return client_->GetPjRtClient()->platform_name();
  }

  std::vector<ComputationClient::DataPtr> ExecuteChained(
      absl::Span<const ComputationClient::ExecuteChainedOp> ops) override {
    return client_->ExecuteChained(ops, name());
  }

  std::vector<ComputationClient::DataPtr> ExecuteComputation(
      const Computation& computation, absl::Span<const DataPtr> arguments,
      const ExecuteComputationOptions& options) override {
    return client_->ExecuteComputation(computation, arguments, name(), options);
  }

  ComputationClient::TransferManager* GetTransferManager() const override {
    return client_;
  }

  xla::PjRtDevice* pjrt_device() const { return pjrt_device_; }

  bool IsLocal() override { return pjrt_device_->IsAddressable(); }

 private:
  PjRtComputationClient* client_;
  xla::PjRtDevice* pjrt_device_;
};

void PjRtComputationClient::PjRtData::Assign(const Data& data) {
  const PjRtData& pjrt_data = dynamic_cast<const PjRtData&>(data);
  if (pjrt_data.HasValue()) {
    // Clone the buffer to the same device
    auto status_or_buffer = pjrt_data.buffer->CopyToDevice(
        static_cast<PjRtDevice*>(device())->pjrt_device());
    XLA_CHECK(status_or_buffer.ok()) << "Failed to copy buffer: "
                                     << status_or_buffer.status();
    buffer = std::move(status_or_buffer.value());
  }
}

size_t PjRtComputationClient::CompilationCacheKey::Hash::operator()(
    const CompilationCacheKey& entry) const {
  util::PartialHasher<std::string, 4096> hasher;
  hash_t h = util::DataHash(entry.device.data(), entry.device.size());
  return util::HashReduce(
      util::HashCombine(h, hasher(entry.serialized_computation)));
}

PjRtComputationClient::PjRtComputationClient(Options options)
    : options_(std::move(options)),
      rng_seed_(0x5a2d296e9) {
  InitializeClient();
  InitializeDevices();
}

PjRtComputationClient::~PjRtComputationClient() = default;

void PjRtComputationClient::InitializeClient() {
  std::string platform = options_.platform;
  if (platform.empty()) {
    platform = sys_util::GetEnvString("XLA_PLATFORM", "cpu");
  }

  LOG(INFO) << "Initializing PJRT client for platform: " << platform;

  if (platform == "cpu") {
#ifdef XLA_CPU_BACKEND
    CpuClientOptions cpu_options;
    cpu_options.cpu_device_count = sys_util::GetEnvInt("XLA_CPU_DEVICE_COUNT", 1);
    auto status_or_client = GetTfrtCpuClient(cpu_options);
    XLA_CHECK(status_or_client.ok())
        << "Failed to create CPU PJRT client: " << status_or_client.status();
    pjrt_client_ = std::move(status_or_client.value());
#else
    XLA_CHECK(false) << "CPU backend not compiled in";
#endif
  } else if (platform == "gpu" || platform == "cuda") {
#ifdef XLA_GPU_BACKEND
    GpuClientOptions gpu_options;
    auto status_or_client = GetStreamExecutorGpuClient(gpu_options);
    XLA_CHECK(status_or_client.ok())
        << "Failed to create GPU PJRT client: " << status_or_client.status();
    pjrt_client_ = std::move(status_or_client.value());
#else
    XLA_CHECK(false) << "GPU backend not compiled in";
#endif
  } else if (platform == "tpu") {
#ifdef XLA_TPU_BACKEND
    auto status_or_client = GetTpuClient(/*max_inflight_computations=*/32);
    XLA_CHECK(status_or_client.ok())
        << "Failed to create TPU PJRT client: " << status_or_client.status();
    pjrt_client_ = std::move(status_or_client.value());
#else
    XLA_CHECK(false) << "TPU backend not compiled in";
#endif
  } else {
    XLA_CHECK(false) << "Unknown platform: " << platform;
  }

  LOG(INFO) << "PJRT client created for platform: "
            << pjrt_client_->platform_name();
}

void PjRtComputationClient::InitializeDevices() {
  absl::Span<xla::PjRtDevice* const> pjrt_devices = pjrt_client_->devices();

  LOG(INFO) << "Initializing " << pjrt_devices.size() << " PJRT devices";

  std::string default_device_name;
  for (xla::PjRtDevice* pjrt_device : pjrt_devices) {
    std::string device_kind = pjrt_device->device_kind();
    // Convert device kind to upper case for consistency
    std::transform(device_kind.begin(), device_kind.end(), device_kind.begin(),
                   ::toupper);

    std::string device_name = absl::StrCat(device_kind, ":", pjrt_device->id());

    auto device = std::make_unique<PjRtDevice>(device_name, this, pjrt_device);

    if (default_device_name.empty() && pjrt_device->IsAddressable()) {
      default_device_name = device_name;
    }

    LOG(INFO) << "Added PJRT device: " << device_name
              << " (addressable: " << pjrt_device->IsAddressable() << ")";

    AddDevice(std::move(device));
  }

  if (options_.default_device.empty()) {
    options_.default_device = default_device_name;
  }

  LOG(INFO) << "Default device: " << options_.default_device;
}

std::vector<Literal> PjRtComputationClient::TransferFromServerImpl(
    absl::Span<const DataPtr> handles) {
  metrics::TimedSection timed(TransferFromServerMetric());

  std::vector<Literal> results;
  results.reserve(handles.size());

  for (const DataPtr& handle : handles) {
    const PjRtData* pjrt_data = dynamic_cast<const PjRtData*>(handle.get());
    XLA_CHECK(pjrt_data != nullptr) << "Invalid data handle type";
    XLA_CHECK(pjrt_data->HasValue()) << "Data handle has no value";

    // Transfer buffer to host
    auto status_or_literal = pjrt_data->buffer->ToLiteralSync();
    XLA_CHECK(status_or_literal.ok())
        << "Failed to transfer data from device: " << status_or_literal.status();

    results.push_back(std::move(status_or_literal.value()));

    OutboundDataMetric()->AddSample(
        ShapeUtil::ByteSizeOf(results.back().shape()));
  }

  return results;
}

std::vector<DataPtr> PjRtComputationClient::TransferToServerInternal(
    PjRtDevice* device, absl::Span<const TensorSource> tensors) {
  metrics::TimedSection timed(TransferToServerMetric());

  std::vector<DataPtr> results;
  results.reserve(tensors.size());

  for (const TensorSource& tensor : tensors) {
    // Create a literal from the tensor source
    Literal literal(tensor.shape);
    tensor.populate_fn(tensor, literal.untyped_data(),
                       literal.size_bytes());

    // Transfer to device
    auto buffer = CreateBufferFromLiteral(literal, device);

    results.push_back(std::make_shared<PjRtData>(
        device, tensor.shape, std::move(buffer)));

    InboundDataMetric()->AddSample(ShapeUtil::ByteSizeOf(tensor.shape));
    CreateDataHandlesCounter()->AddValue(1);
  }

  return results;
}

std::unique_ptr<PjRtBuffer> PjRtComputationClient::CreateBufferFromLiteral(
    const Literal& literal, PjRtDevice* device) {
  auto status_or_buffer = pjrt_client_->BufferFromHostLiteral(
      literal, device->pjrt_device());
  XLA_CHECK(status_or_buffer.ok())
      << "Failed to create buffer from literal: " << status_or_buffer.status();

  auto buffer = std::move(status_or_buffer.value());

  // Wait for the transfer to complete
  auto status = buffer->GetReadyFuture().Await();
  XLA_CHECK(status.ok()) << "Buffer transfer failed: " << status;

  return buffer;
}

std::vector<ComputationPtr> PjRtComputationClient::Compile(
    const std::string& device, const std::vector<std::string>& devices,
    std::vector<CompileInstance> instances) {
  metrics::TimedSection timed(CompileMetric());

  std::vector<ComputationPtr> results;
  results.reserve(instances.size());

  for (auto& instance : instances) {
    // Serialize computation for caching
    std::string serialized = instance.computation.proto().SerializeAsString();
    CompilationCacheKey cache_key(device, serialized);

    // Check cache
    auto cached = compilation_cache_.Get(cache_key);
    if (cached != nullptr) {
      results.push_back(cached);
      continue;
    }

    // Compile the computation
    CompileOptions compile_options;

    if (instance.output_shape != nullptr) {
      // Set up argument layouts if needed
      compile_options.executable_build_options.set_result_layout(
          *instance.output_shape);
    }

    auto status_or_executable = pjrt_client_->Compile(
        instance.computation, compile_options);
    XLA_CHECK(status_or_executable.ok())
        << "Compilation failed: " << status_or_executable.status();

    auto executable = std::move(status_or_executable.value());

    // Get program shape
    auto status_or_program_shape = instance.computation.GetProgramShape();
    XLA_CHECK(status_or_program_shape.ok())
        << "Failed to get program shape: " << status_or_program_shape.status();

    auto computation = std::make_shared<PjRtComputation>(
        std::move(instance.computation),
        status_or_program_shape.value(),
        devices,
        std::move(executable));

    // Cache the result
    compilation_cache_.Add(std::move(cache_key), computation);

    results.push_back(computation);
    CreateCompileHandlesCounter()->AddValue(1);
  }

  return results;
}

std::vector<DataPtr> PjRtComputationClient::ExecuteComputation(
    const Computation& computation, absl::Span<const DataPtr> arguments,
    const std::string& device, const ExecuteComputationOptions& options) {
  metrics::TimedSection timed(ExecuteMetric());

  const PjRtComputation* pjrt_computation =
      dynamic_cast<const PjRtComputation*>(&computation);
  XLA_CHECK(pjrt_computation != nullptr) << "Invalid computation type";
  XLA_CHECK(pjrt_computation->executable != nullptr)
      << "Computation has no executable";

  // Prepare input buffers
  std::vector<PjRtBuffer*> input_buffers;
  input_buffers.reserve(arguments.size());

  for (const DataPtr& arg : arguments) {
    const PjRtData* pjrt_data = dynamic_cast<const PjRtData*>(arg.get());
    XLA_CHECK(pjrt_data != nullptr) << "Invalid argument type";
    XLA_CHECK(pjrt_data->HasValue()) << "Argument has no value";
    input_buffers.push_back(pjrt_data->buffer.get());
  }

  // Get the target device
  PjRtDevice* pjrt_device = static_cast<PjRtDevice*>(GetDevice(device));

  // Execute the computation
  ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;

  auto status_or_results = pjrt_computation->executable->ExecuteSharded(
      input_buffers, pjrt_device->pjrt_device(), execute_options);
  XLA_CHECK(status_or_results.ok())
      << "Execution failed: " << status_or_results.status();

  auto result_buffers = std::move(status_or_results.value());

  // Wrap results in DataPtr
  std::vector<DataPtr> results;
  results.reserve(result_buffers.size());

  const auto& result_shapes = computation.program_shape().result();

  for (size_t i = 0; i < result_buffers.size(); ++i) {
    Shape result_shape;
    if (result_shapes.IsTuple() && options.explode_tuple) {
      result_shape = ShapeUtil::GetTupleElementShape(result_shapes, i);
    } else if (i == 0) {
      result_shape = result_shapes;
    } else {
      XLA_CHECK(false) << "Unexpected result index: " << i;
    }

    results.push_back(std::make_shared<PjRtData>(
        pjrt_device, result_shape, std::move(result_buffers[i])));
    CreateDataHandlesCounter()->AddValue(1);
  }

  return results;
}

std::vector<std::vector<DataPtr>> PjRtComputationClient::ExecuteReplicated(
    const Computation& computation,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions& options) {
  metrics::TimedSection timed(ExecuteReplicatedMetric());

  const PjRtComputation* pjrt_computation =
      dynamic_cast<const PjRtComputation*>(&computation);
  XLA_CHECK(pjrt_computation != nullptr) << "Invalid computation type";

  // Prepare input buffers for all replicas
  std::vector<std::vector<PjRtBuffer*>> all_input_buffers;
  all_input_buffers.reserve(arguments.size());

  for (const auto& replica_args : arguments) {
    std::vector<PjRtBuffer*> replica_buffers;
    replica_buffers.reserve(replica_args.size());
    for (const DataPtr& arg : replica_args) {
      const PjRtData* pjrt_data = dynamic_cast<const PjRtData*>(arg.get());
      XLA_CHECK(pjrt_data != nullptr && pjrt_data->HasValue());
      replica_buffers.push_back(pjrt_data->buffer.get());
    }
    all_input_buffers.push_back(std::move(replica_buffers));
  }

  // Execute on all replicas
  ExecuteOptions execute_options;
  execute_options.untuple_result = options.explode_tuple;

  auto status_or_results = pjrt_computation->executable->Execute(
      all_input_buffers, execute_options);
  XLA_CHECK(status_or_results.ok())
      << "Replicated execution failed: " << status_or_results.status();

  auto all_result_buffers = std::move(status_or_results.value());

  // Wrap results
  std::vector<std::vector<DataPtr>> results;
  results.reserve(all_result_buffers.size());

  for (size_t replica = 0; replica < all_result_buffers.size(); ++replica) {
    std::vector<DataPtr> replica_results;
    replica_results.reserve(all_result_buffers[replica].size());

    PjRtDevice* device = static_cast<PjRtDevice*>(
        GetDevice(std::string(devices[replica])));

    for (auto& buffer : all_result_buffers[replica]) {
      Shape shape(buffer->on_device_shape());
      replica_results.push_back(std::make_shared<PjRtData>(
          device, shape, std::move(buffer)));
    }
    results.push_back(std::move(replica_results));
  }

  return results;
}

std::vector<std::vector<DataPtr>> PjRtComputationClient::ExecuteParallel(
    absl::Span<const Computation* const> computations,
    const std::vector<std::vector<DataPtr>>& arguments,
    absl::Span<const std::string> devices,
    const ExecuteParallelOptions& options) {
  metrics::TimedSection timed(ExecuteParallelMetric());

  std::vector<std::vector<DataPtr>> results;
  results.reserve(computations.size());

  // Execute each computation in parallel using thread pool
  std::vector<util::ExceptionCleanup> cleanups;
  util::MultiWait mwait(computations.size());

  for (size_t i = 0; i < computations.size(); ++i) {
    results.emplace_back();

    env::ScheduleClosure([this, &computations, &arguments, &devices, &options,
                          &results, &mwait, i]() {
      try {
        ExecuteComputationOptions exec_options;
        exec_options.explode_tuple = options.explode_tuple;
        results[i] = ExecuteComputation(*computations[i], arguments[i],
                                        std::string(devices[i]), exec_options);
        mwait.Done();
      } catch (...) {
        mwait.Done();
        throw;
      }
    });
  }

  mwait.Wait();

  return results;
}

std::vector<DataPtr> PjRtComputationClient::ExecuteChained(
    absl::Span<const ExecuteChainedOp> ops, const std::string& device) {
  metrics::TimedSection timed(ExecuteChainedMetric());

  // For PJRT, we implement chained execution by executing each operation
  // sequentially, using the results of previous operations as inputs.
  std::vector<DataPtr> intermediates;
  std::vector<DataPtr> final_results;

  for (const auto& op : ops) {
    if (op.device_data != nullptr) {
      // This is just input data
      intermediates.push_back(op.device_data);
    } else {
      // This is a computation to execute
      XLA_CHECK(op.computation != nullptr);

      // Gather inputs from previous operations
      std::vector<DataPtr> inputs;
      for (const auto& input : op.inputs) {
        XLA_CHECK_LT(input.op_index, intermediates.size());
        if (input.output_index.has_value()) {
          // The intermediate is a tuple, extract the element
          // For now, we assume the intermediate is not a tuple
          inputs.push_back(intermediates[input.op_index]);
        } else {
          inputs.push_back(intermediates[input.op_index]);
        }
      }

      // Execute the computation
      ExecuteComputationOptions exec_options;
      exec_options.explode_tuple = true;
      auto results = ExecuteComputation(*op.computation, inputs, device,
                                        exec_options);

      // Store intermediate results
      if (results.size() == 1) {
        intermediates.push_back(results[0]);
      } else {
        // Multiple outputs - we need to handle tuple results
        intermediates.push_back(results[0]);
      }

      // Add to final results if specified
      for (const auto& output : op.outputs) {
        if (output.output_index.has_value()) {
          XLA_CHECK_LT(*output.output_index, results.size());
          final_results.resize(
              std::max(final_results.size(), output.result_index + 1));
          final_results[output.result_index] = results[*output.output_index];
        } else {
          final_results.resize(
              std::max(final_results.size(), output.result_index + 1));
          final_results[output.result_index] = intermediates.back();
        }
      }
    }
  }

  return final_results;
}

std::string PjRtComputationClient::GetDefaultDevice() const {
  return options_.default_device;
}

swift_xla::Device PjRtComputationClient::GetDefaultDeviceStruct() const {
  return swift_xla::Device(options_.default_device);
}

size_t PjRtComputationClient::GetNumDevices() const {
  return GetAllDevicePointers().size();
}

std::vector<std::string> PjRtComputationClient::GetLocalDevices() const {
  std::vector<std::string> local_devices;
  for (const auto* device : GetAllDevicePointers()) {
    const PjRtDevice* pjrt_device = static_cast<const PjRtDevice*>(device);
    if (pjrt_device->pjrt_device()->IsAddressable()) {
      local_devices.push_back(device->name());
    }
  }
  return local_devices;
}

void PjRtComputationClient::SetRngSeed(size_t seed) {
  rng_seed_.store(seed);
}

std::map<std::string, Metric> PjRtComputationClient::GetMetrics() const {
  return metrics::CreateMetricReport();
}

PjRtDevice* PjRtComputationClient::GetPjRtDevice(
    const std::string& device_name) const {
  return static_cast<PjRtDevice*>(GetDevice(device_name))->pjrt_device();
}

// Factory function to create the computation client
std::unique_ptr<ComputationClient> ComputationClient::Create() {
  PjRtComputationClient::Options options;

  // Parse environment variables for configuration
  options.platform = sys_util::GetEnvString("XLA_PLATFORM", "cpu");
  options.default_device = sys_util::GetEnvString("XLA_DEFAULT_DEVICE", "");

  LOG(INFO) << "Creating PJRT computation client with platform: "
            << options.platform;

  return std::make_unique<PjRtComputationClient>(std::move(options));
}

}  // namespace xla
