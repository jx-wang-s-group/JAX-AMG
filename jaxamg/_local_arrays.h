// Rank-local arrays as runtime operands of rank-identical programs.
//
// An SPMD program (shard_map) must be the same program on every rank: rank-local
// data captured as constants makes each rank compile a different program, whose
// collectives XLA need not order alike. Instead each rank registers its own
// array under an id agreed across ranks (same shape and dtype everywhere), and
// the program loads it with a no-input FFI call naming only that id.
//
// The data is a device array owned by the caller (a JAX array, complete before
// registration and never written afterwards); the registry holds its address,
// and a load is one device-to-device copy on XLA's stream. The caller keeps
// the array alive while the id is registered and until the executions that
// may read it have finished (see SynchronizeDevices).
#pragma once

#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace jaxamg_local
{
  namespace ffi = xla::ffi;

  struct Entry
  {
    const void *data = nullptr;
    size_t bytes = 0;
  };

  inline std::mutex &Mutex()
  {
    static std::mutex m;
    return m;
  }

  inline std::unordered_map<int64_t, Entry> &Arrays()
  {
    static std::unordered_map<int64_t, Entry> arrays;
    return arrays;
  }

  inline void Register(int64_t id, uintptr_t data, size_t bytes)
  {
    std::lock_guard<std::mutex> lock(Mutex());
    Arrays()[id] = Entry{reinterpret_cast<const void *>(data), bytes};
  }

  inline void Release(int64_t id)
  {
    std::lock_guard<std::mutex> lock(Mutex());
    Arrays().erase(id);
  }

  inline size_t Count()
  {
    std::lock_guard<std::mutex> lock(Mutex());
    return Arrays().size();
  }

  // Waits for all work queued on the given devices (the ones JAX uses, so no
  // context is created; the caller releases the GIL): data released
  // afterwards is no longer read by any execution. Throws on any CUDA error,
  // so a failed wait is never taken for success.
  inline void SynchronizeDevices(const std::vector<int> &devices)
  {
    int previous = 0;
    cudaError_t err = cudaGetDevice(&previous);
    for (size_t i = 0; err == cudaSuccess && i < devices.size(); ++i)
    {
      err = cudaSetDevice(devices[i]);
      if (err == cudaSuccess)
        err = cudaDeviceSynchronize();
    }
    const cudaError_t restore = cudaSetDevice(previous);
    if (err == cudaSuccess)
      err = restore;
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("device synchronization failed: ") +
                               cudaGetErrorString(err));
  }

  inline ffi::Error Load(cudaStream_t stream, ffi::Result<ffi::AnyBuffer> out, int64_t id)
  {
    Entry e;
    {
      std::lock_guard<std::mutex> lock(Mutex());
      auto it = Arrays().find(id);
      if (it == Arrays().end())
        return ffi::Error::InvalidArgument("local array: unknown id " + std::to_string(id));
      e = it->second;
    }
    const size_t bytes = out->size_bytes();
    if (bytes != e.bytes)
      return ffi::Error::InvalidArgument(
          "local array " + std::to_string(id) + ": " + std::to_string(e.bytes) +
          " bytes registered, " + std::to_string(bytes) + " requested");
    if (bytes == 0)
      return ffi::Error::Success();
    // cudaMemcpyDefault: a program on another device than the array's reads
    // it through unified addressing.
    if (cudaMemcpyAsync(out->untyped_data(), e.data, bytes, cudaMemcpyDefault, stream) !=
        cudaSuccess)
      return ffi::Error::Internal("local array: device copy failed");
    return ffi::Error::Success();
  }

  XLA_FFI_DEFINE_HANDLER(
      LoadLocalArray,
      Load,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>()
          .Ret<ffi::AnyBuffer>()
          .Attr<int64_t>("array_id"));
} // namespace jaxamg_local
