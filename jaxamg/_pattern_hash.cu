// Device-side fingerprint of a CSR sparsity pattern (the solver cache key).
//
// The cache decides from the pattern whether a cached AmgX matrix can take new
// coefficients (AMGX_matrix_replace_coefficients). Hashing on the device reads
// the pattern at device bandwidth and copies back 16 bytes, instead of copying the
// whole pattern to the host on every solve.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <string>

namespace
{
  // SplitMix64 finalizer: a bijective, well-mixed 64-bit map.
  __device__ __forceinline__ uint64_t Mix64(uint64_t z)
  {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  // acc += sum_i Mix64(Mix64(i ^ salt) ^ data[i]): position-dependent, so a
  // permutation of the entries changes the sum; addition makes the reduction
  // order-independent (deterministic).
  template <typename I>
  __global__ void HashKernel(const I *data, int64_t n, uint64_t salt, unsigned long long *acc)
  {
    uint64_t sum = 0;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
      sum += Mix64(Mix64(static_cast<uint64_t>(i) ^ salt) ^
                   static_cast<uint64_t>(static_cast<int64_t>(data[i])));
    for (int offset = 16; offset > 0; offset >>= 1)
      sum += __shfl_down_sync(0xffffffffu, sum, offset);
    __shared__ uint64_t warp_sums[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0)
      warp_sums[warp] = sum;
    __syncthreads();
    if (warp == 0)
    {
      sum = lane < static_cast<int>((blockDim.x + 31) / 32) ? warp_sums[lane] : 0;
      for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
      if (lane == 0)
        atomicAdd(acc, static_cast<unsigned long long>(sum));
    }
  }

  // The launch's own status: cudaLaunchKernel reports this launch, where
  // cudaGetLastError would also return an earlier, unrelated call's error.
  template <typename I>
  cudaError_t Launch(cudaStream_t stream, const I *data, int64_t n, uint64_t salt,
                     unsigned long long *acc)
  {
    if (n <= 0)
      return cudaSuccess;
    constexpr int threads = 256;
    const int64_t blocks = std::min<int64_t>((n + threads - 1) / threads, 4096);
    void *args[] = {&data, &n, &salt, &acc};
    return cudaLaunchKernel(reinterpret_cast<const void *>(&HashKernel<I>),
                            dim3(static_cast<unsigned>(blocks)), dim3(threads), args, 0, stream);
  }

  const char *Failure(const char *call, cudaError_t err)
  {
    static thread_local std::string message;
    message = std::string("pattern hash: ") + call + " failed (" + cudaGetErrorName(err) +
              ": " + cudaGetErrorString(err) + ")";
    return message.c_str();
  }

  // Makes the stream's device current for the scratch allocation and the
  // copies (the caller's current device can differ from the device a program
  // runs on), restoring the caller's device on exit.
  struct StreamDevice
  {
    int previous = -1;
    cudaError_t status = cudaSuccess;
    explicit StreamDevice(cudaStream_t stream)
    {
      int device = -1;
      status = cudaGetDevice(&previous);
      if (status == cudaSuccess)
        status = cudaStreamGetDevice(stream, &device);
      if (status == cudaSuccess && device != previous)
        status = cudaSetDevice(device);
      else
        previous = -1;  // nothing to restore
    }
    ~StreamDevice()
    {
      if (previous >= 0)
        cudaSetDevice(previous);
    }
  };
} // namespace

// Fingerprint of (row_ptrs[0:n_row_ptrs], col_indices[0:nnz]) with
// col_index_bytes 4 (int32) or 8 (int64). Synchronizes the stream. Returns
// nullptr on success, else an error message (valid until the thread's next failure).
extern "C" const char *jaxamg_pattern_hash(cudaStream_t stream, const int *row_ptrs,
                                           int64_t n_row_ptrs, const void *col_indices,
                                           int64_t nnz, int col_index_bytes, uint64_t *hash_out)
{
  if (col_index_bytes != 4 && col_index_bytes != 8)
    return "pattern hash: column indices must be 4 or 8 bytes";
  const StreamDevice on_stream_device(stream);
  if (on_stream_device.status != cudaSuccess)
    return Failure("selecting the stream's device", on_stream_device.status);
  unsigned long long *acc = nullptr;
  cudaError_t err = cudaMallocAsync(reinterpret_cast<void **>(&acc), 2 * sizeof(unsigned long long), stream);
  if (err != cudaSuccess)
    return Failure("cudaMallocAsync", err);
  const char *call = "cudaMemsetAsync";
  err = cudaMemsetAsync(acc, 0, 2 * sizeof(unsigned long long), stream);
  if (err == cudaSuccess)
  {
    call = "the row_ptrs launch";
    err = Launch<int>(stream, row_ptrs, n_row_ptrs, 0x1ULL, acc);
  }
  if (err == cudaSuccess)
  {
    call = "the col_indices launch";
    err = col_index_bytes == 8
              ? Launch<int64_t>(stream, static_cast<const int64_t *>(col_indices), nnz, 0x2ULL, acc + 1)
              : Launch<int>(stream, static_cast<const int *>(col_indices), nnz, 0x2ULL, acc + 1);
  }
  unsigned long long sums[2] = {0, 0};
  if (err == cudaSuccess)
  {
    call = "cudaMemcpyAsync";
    err = cudaMemcpyAsync(sums, acc, sizeof(sums), cudaMemcpyDeviceToHost, stream);
  }
  const cudaError_t freed = cudaFreeAsync(acc, stream);
  if (err == cudaSuccess && freed != cudaSuccess)
  {
    call = "cudaFreeAsync";
    err = freed;
  }
  // Always drain submitted work before the stack-backed host result goes out
  // of scope, including when freeing the device scratch buffer failed.
  const cudaError_t synced = cudaStreamSynchronize(stream);
  if (err == cudaSuccess)
  {
    call = "cudaStreamSynchronize";
    err = synced;
  }
  if (err != cudaSuccess)
    return Failure(call, err);
  // Combine with the sizes, so patterns of different lengths differ.
  uint64_t h = sums[0] ^ (sums[1] * 0x9e3779b97f4a7c15ULL);
  h ^= static_cast<uint64_t>(n_row_ptrs) * 0xbf58476d1ce4e5b9ULL;
  h ^= static_cast<uint64_t>(nnz) * 0x94d049bb133111ebULL;
  *hash_out = h;
  return nullptr;
}
