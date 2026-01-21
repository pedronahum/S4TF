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

// PJRT C API for loading pre-built plugins
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"

#include <dlfcn.h>
#include <filesystem>
#include <regex>

namespace xla {

using DataPtr = ComputationClient::DataPtr;
using ComputationPtr = ComputationClient::ComputationPtr;
using TensorSource = ComputationClient::TensorSource;

namespace {

// ============================================================================
// Pre-built PJRT Plugin Discovery
// ============================================================================
// These functions help locate PJRT plugins from Python packages like JAX or
// TensorFlow, allowing users to skip building C++ from source.

// Known PJRT plugin library names for different backends
struct PluginInfo {
  std::string env_var;           // Environment variable override
  std::vector<std::string> lib_names;  // Library file names to search for
  std::string package_hint;      // Python package that contains this plugin
};

const std::map<std::string, PluginInfo>& GetPluginRegistry() {
  static const std::map<std::string, PluginInfo> registry = {
      {"cpu", {
          "PJRT_CPU_LIBRARY_PATH",
          {"pjrt_c_api_cpu_plugin.so", "libpjrt_c_api_cpu.so",
           "pjrt_plugin_xla_cpu.so"},
          "jax[cpu] or tensorflow"
      }},
      {"cuda", {
          "PJRT_CUDA_LIBRARY_PATH",
          {"pjrt_c_api_gpu_plugin.so", "libpjrt_c_api_gpu.so",
           "pjrt_plugin_xla_cuda.so", "libxla_cuda.so"},
          "jax[cuda] or tensorflow"
      }},
      {"gpu", {
          "PJRT_GPU_LIBRARY_PATH",
          {"pjrt_c_api_gpu_plugin.so", "libpjrt_c_api_gpu.so",
           "pjrt_plugin_xla_cuda.so", "libxla_cuda.so"},
          "jax[cuda] or tensorflow"
      }},
      {"tpu", {
          "PJRT_TPU_LIBRARY_PATH",
          {"libtpu.so", "pjrt_c_api_tpu_plugin.so"},
          "libtpu or jax[tpu]"
      }},
      {"rocm", {
          "PJRT_ROCM_LIBRARY_PATH",
          {"pjrt_c_api_rocm_plugin.so", "libpjrt_c_api_rocm.so"},
          "jax[rocm]"
      }},
  };
  return registry;
}

// Get Python site-packages directories
std::vector<std::string> GetPythonSitePackages() {
  std::vector<std::string> paths;

  // Try to run Python to get site-packages
  FILE* pipe = popen("python3 -c 'import site; print(\"\\n\".join(site.getsitepackages()))' 2>/dev/null", "r");
  if (pipe) {
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      std::string path(buffer);
      // Remove trailing newline
      if (!path.empty() && path.back() == '\n') {
        path.pop_back();
      }
      if (!path.empty()) {
        paths.push_back(path);
      }
    }
    pclose(pipe);
  }

  // Also check common paths
  const char* home = getenv("HOME");
  if (home) {
    paths.push_back(std::string(home) + "/.local/lib/python3.10/site-packages");
    paths.push_back(std::string(home) + "/.local/lib/python3.11/site-packages");
    paths.push_back(std::string(home) + "/.local/lib/python3.12/site-packages");
  }

  // Check for conda environment
  const char* conda_prefix = getenv("CONDA_PREFIX");
  if (conda_prefix) {
    paths.push_back(std::string(conda_prefix) + "/lib/python3.10/site-packages");
    paths.push_back(std::string(conda_prefix) + "/lib/python3.11/site-packages");
    paths.push_back(std::string(conda_prefix) + "/lib/python3.12/site-packages");
  }

  return paths;
}

// Search for a PJRT plugin library
std::string FindPjrtPlugin(const std::string& platform) {
  auto& registry = GetPluginRegistry();
  auto it = registry.find(platform);
  if (it == registry.end()) {
    return "";  // Unknown platform
  }

  const PluginInfo& info = it->second;

  // First, check environment variable override
  const char* env_path = getenv(info.env_var.c_str());
  if (env_path && std::filesystem::exists(env_path)) {
    LOG(INFO) << "Using PJRT plugin from " << info.env_var << ": " << env_path;
    return env_path;
  }

  // Check PJRT_PLUGIN_LIBRARY_PATH (generic override)
  const char* generic_path = getenv("PJRT_PLUGIN_LIBRARY_PATH");
  if (generic_path && std::filesystem::exists(generic_path)) {
    LOG(INFO) << "Using PJRT plugin from PJRT_PLUGIN_LIBRARY_PATH: " << generic_path;
    return generic_path;
  }

  // Search in Python site-packages
  auto site_packages = GetPythonSitePackages();

  // Directories to search within site-packages
  std::vector<std::string> search_subdirs = {
      "jaxlib",
      "jaxlib/xla_extension",
      "jax_plugins",
      "jax_plugins/xla_cpu",
      "jax_plugins/xla_cuda",
      "tensorflow",
      "tensorflow/compiler/tf2xla/python",
      "tensorflow/python/_pywrap_tfe",
      "xla",
      "xla/pjrt",
  };

  for (const auto& site_pkg : site_packages) {
    // Search directly in site-packages
    for (const auto& lib_name : info.lib_names) {
      std::string path = site_pkg + "/" + lib_name;
      if (std::filesystem::exists(path)) {
        LOG(INFO) << "Found PJRT plugin: " << path;
        return path;
      }
    }

    // Search in subdirectories
    for (const auto& subdir : search_subdirs) {
      for (const auto& lib_name : info.lib_names) {
        std::string path = site_pkg + "/" + subdir + "/" + lib_name;
        if (std::filesystem::exists(path)) {
          LOG(INFO) << "Found PJRT plugin: " << path;
          return path;
        }
      }
    }
  }

  // Search in LD_LIBRARY_PATH
  const char* ld_path = getenv("LD_LIBRARY_PATH");
  if (ld_path) {
    std::vector<std::string> ld_paths = absl::StrSplit(ld_path, ':');
    for (const auto& dir : ld_paths) {
      for (const auto& lib_name : info.lib_names) {
        std::string path = std::string(dir) + "/" + lib_name;
        if (std::filesystem::exists(path)) {
          LOG(INFO) << "Found PJRT plugin in LD_LIBRARY_PATH: " << path;
          return path;
        }
      }
    }
  }

  return "";  // Not found
}

// Load a PJRT plugin and create a client
std::unique_ptr<PjRtClient> LoadPjrtPlugin(const std::string& library_path) {
  LOG(INFO) << "Loading PJRT plugin from: " << library_path;

  // Load the shared library
  void* handle = dlopen(library_path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    LOG(ERROR) << "Failed to load PJRT plugin: " << dlerror();
    return nullptr;
  }

  // Look for the GetPjrtApi function
  using GetPjrtApiFn = const PJRT_Api* (*)();
  auto get_pjrt_api = reinterpret_cast<GetPjrtApiFn>(
      dlsym(handle, "GetPjrtApi"));

  if (!get_pjrt_api) {
    LOG(ERROR) << "PJRT plugin does not export GetPjrtApi: " << dlerror();
    dlclose(handle);
    return nullptr;
  }

  // Get the PJRT API
  const PJRT_Api* api = get_pjrt_api();
  if (!api) {
    LOG(ERROR) << "GetPjrtApi returned nullptr";
    dlclose(handle);
    return nullptr;
  }

  LOG(INFO) << "Loaded PJRT plugin with version: "
            << api->pjrt_api_version.major_version << "."
            << api->pjrt_api_version.minor_version;

  // Create a PJRT C API client
  auto status_or_client = GetCApiClient(api);
  if (!status_or_client.ok()) {
    LOG(ERROR) << "Failed to create PJRT C API client: "
               << status_or_client.status();
    // Note: We don't dlclose here because the API may be in use
    return nullptr;
  }

  return std::move(status_or_client.value());
}

}  // namespace

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

  // Check if we should try to use pre-built plugins
  bool use_prebuilt = sys_util::GetEnvBool("XLA_USE_PREBUILT_PJRT", true);

  if (use_prebuilt) {
    // Try to find and load a pre-built PJRT plugin from Python packages
    std::string plugin_path = FindPjrtPlugin(platform);
    if (!plugin_path.empty()) {
      pjrt_client_ = LoadPjrtPlugin(plugin_path);
      if (pjrt_client_) {
        LOG(INFO) << "Successfully loaded pre-built PJRT plugin for platform: "
                  << platform;
        LOG(INFO) << "PJRT client created from plugin: "
                  << pjrt_client_->platform_name();
        return;
      }
      LOG(WARNING) << "Failed to load PJRT plugin, falling back to compiled-in backend";
    } else {
      LOG(INFO) << "No pre-built PJRT plugin found for platform: " << platform
                << ". Install jax or tensorflow, or set PJRT_*_LIBRARY_PATH";
    }
  }

  // Fall back to compiled-in backends
  LOG(INFO) << "Using compiled-in PJRT backend for platform: " << platform;

  if (platform == "cpu") {
#ifdef XLA_CPU_BACKEND
    CpuClientOptions cpu_options;
    cpu_options.cpu_device_count = sys_util::GetEnvInt("XLA_CPU_DEVICE_COUNT", 1);
    auto status_or_client = GetTfrtCpuClient(cpu_options);
    XLA_CHECK(status_or_client.ok())
        << "Failed to create CPU PJRT client: " << status_or_client.status();
    pjrt_client_ = std::move(status_or_client.value());
#else
    auto& registry = GetPluginRegistry();
    auto it = registry.find("cpu");
    XLA_CHECK(false) << "CPU backend not compiled in. "
                     << "Install a pre-built PJRT plugin ("
                     << (it != registry.end() ? it->second.package_hint : "jax or tensorflow")
                     << ") or build with XLA_CPU_BACKEND=1";
#endif
  } else if (platform == "gpu" || platform == "cuda") {
#ifdef XLA_GPU_BACKEND
    GpuClientOptions gpu_options;
    auto status_or_client = GetStreamExecutorGpuClient(gpu_options);
    XLA_CHECK(status_or_client.ok())
        << "Failed to create GPU PJRT client: " << status_or_client.status();
    pjrt_client_ = std::move(status_or_client.value());
#else
    auto& registry = GetPluginRegistry();
    auto it = registry.find("gpu");
    XLA_CHECK(false) << "GPU backend not compiled in. "
                     << "Install a pre-built PJRT plugin ("
                     << (it != registry.end() ? it->second.package_hint : "jax[cuda] or tensorflow")
                     << ") or build with XLA_GPU_BACKEND=1";
#endif
  } else if (platform == "tpu") {
#ifdef XLA_TPU_BACKEND
    auto status_or_client = GetTpuClient(/*max_inflight_computations=*/32);
    XLA_CHECK(status_or_client.ok())
        << "Failed to create TPU PJRT client: " << status_or_client.status();
    pjrt_client_ = std::move(status_or_client.value());
#else
    auto& registry = GetPluginRegistry();
    auto it = registry.find("tpu");
    XLA_CHECK(false) << "TPU backend not compiled in. "
                     << "Install a pre-built PJRT plugin ("
                     << (it != registry.end() ? it->second.package_hint : "libtpu or jax[tpu]")
                     << ") or build with XLA_TPU_BACKEND=1";
#endif
  } else if (platform == "rocm") {
    // ROCm/AMD GPU - only available via plugin
    auto& registry = GetPluginRegistry();
    auto it = registry.find("rocm");
    XLA_CHECK(false) << "ROCm backend requires a pre-built PJRT plugin. "
                     << "Install " << (it != registry.end() ? it->second.package_hint : "jax[rocm]")
                     << " and set PJRT_ROCM_LIBRARY_PATH";
  } else {
    XLA_CHECK(false) << "Unknown platform: " << platform
                     << ". Supported platforms: cpu, gpu, cuda, tpu, rocm";
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
