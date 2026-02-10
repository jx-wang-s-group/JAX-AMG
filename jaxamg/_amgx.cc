/*
 *
 * This file implements the XLA Custom Call handler for NVIDIA AmgX.
 * It uses the JAX Typed FFI (foreign function interface) to expose
 * AmgX functionality to JAX programs running on GPU.
 */

#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <xla/ffi/api/ffi.h>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <atomic>
#include <string>
#include <fstream>
#include <vector>
#include <list>
#include <unordered_map>
#include <functional>
#include <utility>
#include <mpi.h>

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace
{

  // LRU Cache for AmgX solvers.
  template <typename Key, typename Value>
  class LRUCache
  {
  public:
    explicit LRUCache(size_t capacity) : capacity_(capacity) {}

    bool get(const Key &key, Value &value)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = cache_map_.find(key);
      if (it == cache_map_.end())
      {
        return false;
      }
      // Move to front
      lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
      value = it->second->second;
      return true;
    }

    void put(const Key &key, const Value &value, std::function<void(Value &)> destructor)
    {
      std::lock_guard<std::mutex> lock(mutex_);

      // If capacity is 0, don't cache at all. Destroy immediately.
      if (capacity_ == 0)
      {
         if (destructor) destructor(const_cast<Value&>(value));
         return;
      }

      auto it = cache_map_.find(key);
      if (it != cache_map_.end())
      {
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        // Clean up old value before overwriting!
        if (destructor)
        {
           destructor(it->second->second);
        }
        it->second->second = value;
      }
      else
      {
        if (cache_map_.size() >= capacity_)
        {
          auto last = lru_list_.end();
          last--;
          if (destructor)
          {
            destructor(last->second);
          }
          cache_map_.erase(last->first);
          lru_list_.pop_back();
        }
        lru_list_.push_front({key, value});
        cache_map_[key] = lru_list_.begin();
      }
    }

    void erase(const Key &key, std::function<void(Value &)> destructor)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = cache_map_.find(key);
      if (it != cache_map_.end())
      {
        if (destructor) destructor(it->second->second);
        lru_list_.erase(it->second);
        cache_map_.erase(it);
      }
    }

    void clear(std::function<void(Value &)> destructor)
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (auto &pair : lru_list_)
      {
         if (destructor) destructor(pair.second);
      }
      lru_list_.clear();
      cache_map_.clear();
    }

  private:
    size_t capacity_;
    std::list<std::pair<Key, Value>> lru_list_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> cache_map_;
    std::mutex mutex_;
  };

  struct CacheKey
  {
    const void *row_ptrs;
    const void *col_indices;
    int n_rows;
    int nnz;
    std::string config; // Config string content acts as part of key

    bool operator==(const CacheKey &other) const
    {
      return row_ptrs == other.row_ptrs &&
             col_indices == other.col_indices &&
             n_rows == other.n_rows &&
             nnz == other.nnz &&
             config == other.config;
    }
  };
} // namespace

namespace std
{
  template <>
  struct hash<CacheKey>
  {
    size_t operator()(const CacheKey &k) const
    {
      // Simple hash combination
      size_t h1 = hash<const void *>()(k.row_ptrs);
      size_t h2 = hash<const void *>()(k.col_indices);
      size_t h3 = hash<int>()(k.n_rows);
      size_t h4 = hash<int>()(k.nnz);
      size_t h5 = hash<string>()(k.config);

      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
    }
  };
} // namespace std

namespace
{
  struct CachedResources
  {
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_solver_handle solver;
    AMGX_vector_handle x_vec;
    AMGX_vector_handle b_vec;
    bool owns_resources; // True if isolated (Cache=0), False if shared (Cache>=1)
  };

  // Global cache instance (heap-allocated to persist until explicit finalization).
  LRUCache<CacheKey, CachedResources>& GetSolverCache()
  {
    static auto* cache = []() {
      // Default to 1 (reuse enabled) for shared mode.
      size_t capacity = 1;
      const char* env_val = std::getenv("JAXAMG_CACHE_SIZE");
      if (env_val) {
        try {
            capacity = std::stoul(env_val);
            // capacity allowed to be 0
        } catch (...) {
            capacity = 1; // Fallback
        }
      }

      return new LRUCache<CacheKey, CachedResources>(capacity);
    }();
    return *cache;
  }

  void DestroyResources(CachedResources &res)
  {
    try
    {
      // Prevent segfaults at program exit.
      if (cudaDeviceSynchronize() != cudaSuccess) {
          return;
      }

      // Destroy in reverse order of creation
      // Always destroy solver to prevent leaks. GlobalResources (Shared Mode) persists memory pools.
      if (res.solver) AMGX_solver_destroy(res.solver);

      if (res.b_vec) AMGX_vector_destroy(res.b_vec);
      if (res.x_vec) AMGX_vector_destroy(res.x_vec);
      if (res.A) AMGX_matrix_destroy(res.A);

      // Only destroy resources handle in Isolated Mode (Mode 0).
      if (res.owns_resources) {
          if (res.rsrc) AMGX_resources_destroy(res.rsrc);
      }

      if (res.cfg) AMGX_config_destroy(res.cfg);
    }
    catch (...)
    {
      // Swallow all exceptions during destruction to prevent 'terminate' on exit
      // invalid_argument from CUDA is common during shutdown if context is gone
    }
  }

// Undefine existing macro from amgx_c.h to allow custom error handling
#ifdef AMGX_SAFE_CALL
#undef AMGX_SAFE_CALL
#endif

// Macro for functions returning ffi::Error (propagates to Python)
#define AMGX_SAFE_CALL(call)                                     \
  do                                                             \
  {                                                              \
    AMGX_RC err = (call);                                        \
    if (err != AMGX_RC_OK)                                       \
    {                                                            \
      char msg[4096];                                            \
      AMGX_get_error_string(err, msg, 4096);                     \
      std::string error_msg = "AMGX Error: " + std::string(msg); \
      return ffi::Error::Internal(error_msg);                    \
    }                                                            \
  } while (0)

// Macro for functions returning void (just log error)
#define AMGX_SAFE_CALL_VOID(call)                                \
  do                                                             \
  {                                                              \
    AMGX_RC err = (call);                                        \
    if (err != AMGX_RC_OK)                                       \
    {                                                            \
      char msg[4096];                                            \
      AMGX_get_error_string(err, msg, 4096);                     \
      fprintf(stderr, "AMGX Error in void function: %s\n", msg); \
    }                                                            \
  } while (0)

  // Singleton for global AmgX resource management.
  class GlobalResources {
  public:
      static GlobalResources& Get() {
          static GlobalResources* instance = new GlobalResources();
          return *instance;
      }

      AMGX_resources_handle GetHandle(AMGX_config_handle cfg) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!handle_) {
              // Create resources with the provided config.
              // Note: We assume the first config providing resources settings
              // is sufficient for the application's lifetime.
              AMGX_SAFE_CALL_VOID(AMGX_resources_create_simple(&handle_, cfg));
          }
          return handle_;
      }

      void Destroy() {
          std::lock_guard<std::mutex> lock(mutex_);
          if (handle_) {
              AMGX_SAFE_CALL_VOID(AMGX_resources_destroy(handle_));
              handle_ = nullptr;
          }
      }

  private:
      GlobalResources() : handle_(nullptr) {}
      ~GlobalResources() { Destroy(); }

      AMGX_resources_handle handle_;
      std::mutex mutex_;
  };


  std::once_flag g_amgx_init_flag;

  std::atomic<bool> g_amgx_finalized{false};

  void AmgxFinalize()
  {
    bool expected = false;
    if (g_amgx_finalized.compare_exchange_strong(expected, true)) {
        // Clear cache and destroy all resources before finalizing AMGX
        GetSolverCache().clear(DestroyResources);

        // Destroy the global resources handle explicitly
        GlobalResources::Get().Destroy();

        // AMGX_SAFE_CALL_VOID(AMGX_finalize()); // Disable to avoid SEGFAULT at exit
    }
  }


  // Custom callback to suppress AmgX library output (banners, version info)
  void PrintCallback(const char *msg, int length)
  {
    // No-op: Output is fully suppressed to keep stdout clean for the user.
    return;
  }

  void EnsureAmgxInitialized()
  {
    std::call_once(g_amgx_init_flag, []()
                   {
                     // Register print callback before initialization
                     AMGX_register_print_callback(PrintCallback);

                     AMGX_SAFE_CALL_VOID(AMGX_initialize());
                     AMGX_SAFE_CALL_VOID(AMGX_install_signal_handler());
                     // Note: Finalization handled by Python atexit (runs before MPI_FINALIZE)
                   });
  }

  /*
   * AmgxSolveInternal: Templated core implementation of the XLA FFI handler.
  /*
   * AmgxSolveInternal: Templated core implementation of the XLA FFI handler.
   * Supports both float (AMGX_mode_dFFI) and double (AMGX_mode_dDDI).
   */
  template <typename T, ffi::DataType DType, AMGX_Mode Mode>
  ffi::Error AmgxSolveInternal(cudaStream_t stream,
                               ffi::Buffer<ffi::DataType::S32> row_ptrs,
                               ffi::Buffer<ffi::DataType::S32> col_indices,
                               ffi::Buffer<DType> values,
                               ffi::Buffer<DType> b,
                               ffi::ResultBuffer<DType> x,
                               ffi::ResultBuffer<DType> stats,
                               std::string_view config)
  {
    EnsureAmgxInitialized();

    // Ensure input buffers are ready.
    cudaStreamSynchronize(stream);

    CachedResources res;
    // 1. Setup execution context
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
      return ffi::Error::Internal("cudaGetDevice failed");
    }

    if (cudaSetDevice(device) != cudaSuccess)
    {
      return ffi::Error::Internal("cudaSetDevice failed");
    }

    // 2. Prepare data pointers
    // Cast to raw pointers to avoid host transfers.
    int *row_ptrs_data = const_cast<int *>(row_ptrs.typed_data());
    int *col_indices_data = const_cast<int *>(col_indices.typed_data());
    T *values_data = const_cast<T *>(values.typed_data());
    T *b_data = const_cast<T *>(b.typed_data());
    T *x_data = x->typed_data();
    T *stats_data = stats->typed_data();

    // Dimensions
    const int n_rows = static_cast<int>(b.dimensions().size() > 0 ? b.dimensions()[0] : 0);
    const int nnz = static_cast<int>(values.element_count());

    // 3. Check Cache
    CacheKey key = {row_ptrs_data, col_indices_data, n_rows, nnz, std::string(config)};
    // CachedResources res; // Removed duplicate declaration
    bool cache_hit = GetSolverCache().get(key, res);

    bool reuse_success = false;

    if (cache_hit)
    {
      // REUSE RESOURCES
      // Update matrix values
      // Note: AMGX_matrix_replace_coefficients signature: (mtx, n_rows, nnz, values, diag)
      AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(res.A, n_rows, (int)values.element_count(), values_data, nullptr));

      // Upload new RHS
      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_rows, 1, b_data));

      // Reset X to zero
      AMGX_SAFE_CALL(AMGX_vector_set_zero(res.x_vec, n_rows, 1));

      reuse_success = true;

    }

    if (!cache_hit)
    {

      // Create new resources
      // Use teplated mode
      const AMGX_Mode mode = Mode;

      // Prepare configuration
      std::string config_str(config);

      // Attempt to open as file to determine if it's a file path
      std::ifstream file_check(config_str);
      bool is_file = file_check.good();
      file_check.close(); // Close the stream before AmgX tries to read

      if (is_file)
      {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&res.cfg, config_str.c_str()));
      }
      else
      {
        AMGX_SAFE_CALL(AMGX_config_create(&res.cfg, config_str.c_str()));
      }

      // Determine allocation strategy based on cache size.
      bool is_isolated = false; // Default to Shared Mode (Mode 1+)
      const char* env_val = std::getenv("JAXAMG_CACHE_SIZE");
      if (env_val) {
        try {
            unsigned long val = std::stoul(env_val);
            if (val == 0) is_isolated = true;
        } catch(...) {
        }
      }

      if (is_isolated) {
         // Mode 0: Isolated Resources
         res.owns_resources = true;
         AMGX_SAFE_CALL(AMGX_resources_create_simple(&res.rsrc, res.cfg));
      } else {
         // Mode 1+: Shared Global Resources
         res.owns_resources = false;
         res.rsrc = GlobalResources::Get().GetHandle(res.cfg);
      }

      // Create Matrix and Vectors
      AMGX_SAFE_CALL(AMGX_matrix_create(&res.A, res.rsrc, mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.x_vec, res.rsrc, mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.b_vec, res.rsrc, mode));
      AMGX_SAFE_CALL(AMGX_solver_create(&res.solver, res.rsrc, mode, res.cfg));

      // Bind Data (No Copies)
      // Upload matrix data (binds device pointers)

      AMGX_SAFE_CALL(AMGX_matrix_upload_all(res.A, n_rows, (int)values.element_count(), 1, 1,
                                            row_ptrs_data, col_indices_data, values_data, nullptr));

      // Bind RHS vector
      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_rows, 1, b_data));

      // Initialize solution vector to zero
      AMGX_SAFE_CALL(AMGX_vector_set_zero(res.x_vec, n_rows, 1));
    }

    // 5. Solve System
    AMGX_SAFE_CALL(AMGX_solver_setup(res.solver, res.A));
    AMGX_SAFE_CALL(AMGX_solver_solve(res.solver, res.b_vec, res.x_vec));

    // 6. Retrieve Results & Statistics
    AMGX_SOLVE_STATUS status;
    AMGX_SAFE_CALL(AMGX_solver_get_status(res.solver, &status));

    if (status == AMGX_SOLVE_FAILED)
    {
      return ffi::Error::Internal("AmgX solve failed");
    }

    int iters = 0;
    double residual = 0.0;

    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(res.solver, &iters));

    // Retrieve residual at the final iteration
    AMGX_SAFE_CALL(AMGX_solver_get_iteration_residual(res.solver, iters, 0, &residual));

    // Transfer stats to output buffer [iters, residual, status]
    T stats_host[3] = {
        static_cast<T>(iters),
        static_cast<T>(residual),
        static_cast<T>(status)};
    cudaMemcpyAsync(stats_data, stats_host, 3 * sizeof(T), cudaMemcpyHostToDevice, stream);

    // Download solution (copies from AmgX internal buffer to JAX output buffer)
    AMGX_SAFE_CALL(AMGX_vector_download(res.x_vec, x_data));

    // 7. Store in Cache (if new)
    if (!cache_hit)
    {
       GetSolverCache().put(key, res, DestroyResources);
    }
    // Do NOT destroy resources here! They are now owned by the cache.
    // Cleanup happens only on eviction via DestroyResources.

    return ffi::Error::Success();
  }


  /*
   * AmgxSolveMPIInternal: MPI-aware templated core implementation.
   * Uses AMGX_resources_create() with MPI communicator and
   * AMGX_matrix_upload_all_global() for distributed matrices.
   */
  template <typename T, ffi::DataType DType, AMGX_Mode Mode>
  ffi::Error AmgxSolveMPIInternal(cudaStream_t stream,
                                  ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                  ffi::Buffer<ffi::DataType::S64> col_indices,
                                  ffi::Buffer<DType> values,
                                  ffi::Buffer<DType> b,
                                  ffi::Buffer<ffi::DataType::S32> nglobal_buf,
                                  ffi::Buffer<ffi::DataType::S32> comm_ptr_buf,
                                  ffi::Buffer<ffi::DataType::S32> lrank_buf,
                                  ffi::ResultBuffer<DType> x,
                                  ffi::ResultBuffer<DType> stats,
                                  std::string_view config)
  {
    EnsureAmgxInitialized();

    // Setup execution context
    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
      return ffi::Error::Internal("cudaGetDevice failed");
    }

    if (cudaSetDevice(device) != cudaSuccess)
    {
      return ffi::Error::Internal("cudaSetDevice failed");
    }

    // Synchronize stream to ensure inputs are ready
    cudaStreamSynchronize(stream);

    // Get MPI parameters
    int nglobal_host;
    int comm_ptr_parts[2]; // [low, high] for 64-bit reconstruction
    int lrank_host;

    cudaMemcpy(&nglobal_host, nglobal_buf.typed_data(), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(comm_ptr_parts, comm_ptr_buf.typed_data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lrank_host, lrank_buf.typed_data(), sizeof(int), cudaMemcpyDeviceToHost);

    // Reconstruct 64-bit pointer from two 32-bit parts
    uint64_t comm_ptr_val = (static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[1])) << 32) |
                            static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[0]));

    // MPI._addressof() returns the address of the MPI_Comm object
    // We need to pass this address to AMGX (which expects MPI_Comm*)
    MPI_Comm *mpi_comm = reinterpret_cast<MPI_Comm *>(comm_ptr_val);

    // 3. Prepare data pointers
    int *row_ptrs_data = const_cast<int *>(row_ptrs.typed_data());
    int64_t *col_indices_data = const_cast<int64_t *>(col_indices.typed_data());
    T *values_data = const_cast<T *>(values.typed_data());
    T *b_data = const_cast<T *>(b.typed_data());
    T *x_data = x->typed_data();
    T *stats_data = stats->typed_data();

    // Dimensions
    const int n_local = static_cast<int>(b.dimensions().size() > 0 ? b.dimensions()[0] : 0);
    const int nnz = static_cast<int>(values.element_count());

    // Initialize AMGX resources with MPI
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle x_vec, b_vec;
    AMGX_solver_handle solver;

    std::string config_str(config);
    std::ifstream file_check(config_str);
    bool is_file = file_check.good();
    file_check.close();

    if (is_file)
    {
      AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, config_str.c_str()));
    }
    else
    {
      AMGX_SAFE_CALL(AMGX_config_create(&cfg, config_str.c_str()));
    }

    AMGX_SAFE_CALL(AMGX_resources_create(&rsrc, cfg, mpi_comm, 1, &lrank_host));
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, Mode));
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, Mode, cfg));

    // Upload distributed matrix with global column indices
    int nrings = 1;
    AMGX_config_get_default_number_of_rings(cfg, &nrings);

    AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(
        A, nglobal_host, n_local, nnz, 1, 1,
        row_ptrs_data, col_indices_data, values_data, nullptr,
        nrings, nrings, nullptr));

    // Create and initialize vectors (AMGX manages halo space)
    AMGX_SAFE_CALL(AMGX_vector_create(&x_vec, rsrc, Mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&b_vec, rsrc, Mode));
    AMGX_SAFE_CALL(AMGX_vector_bind(x_vec, A));
    AMGX_SAFE_CALL(AMGX_vector_bind(b_vec, A));

    std::vector<T> h_x(n_local, 0);
    AMGX_SAFE_CALL(AMGX_vector_upload(x_vec, n_local, 1, h_x.data()));
    AMGX_SAFE_CALL(AMGX_vector_upload(b_vec, n_local, 1, b_data));

    // Solve
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
    AMGX_SAFE_CALL(AMGX_solver_solve(solver, b_vec, x_vec));

    // Retrieve results
    AMGX_SOLVE_STATUS status;
    AMGX_SAFE_CALL(AMGX_solver_get_status(solver, &status));

    if (status == AMGX_SOLVE_FAILED)
    {
      AMGX_solver_destroy(solver);
      AMGX_vector_destroy(b_vec);
      AMGX_vector_destroy(x_vec);
      AMGX_matrix_destroy(A);
      AMGX_resources_destroy(rsrc);
      AMGX_config_destroy(cfg);
      return ffi::Error::Internal("AmgX MPI solve failed");
    }

    int iters = 0;
    double residual = 0.0;
    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(solver, &iters));
    AMGX_SAFE_CALL(AMGX_solver_get_iteration_residual(solver, iters, 0, &residual));

    T stats_host[3] = {static_cast<T>(iters), static_cast<T>(residual), static_cast<T>(status)};
    cudaMemcpyAsync(stats_data, stats_host, 3 * sizeof(T), cudaMemcpyHostToDevice, stream);
    AMGX_SAFE_CALL(AMGX_vector_download(x_vec, x_data));

    // Cleanup
    AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
    AMGX_SAFE_CALL(AMGX_vector_destroy(b_vec));
    AMGX_SAFE_CALL(AMGX_vector_destroy(x_vec));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    return ffi::Error::Success();
  }

  // Float implementation (single-GPU)
  ffi::Error AmgxSolveImpl(cudaStream_t stream,
                           ffi::Buffer<ffi::DataType::S32> row_ptrs,
                           ffi::Buffer<ffi::DataType::S32> col_indices,
                           ffi::Buffer<ffi::DataType::F32> values,
                           ffi::Buffer<ffi::DataType::F32> b,
                           ffi::ResultBuffer<ffi::DataType::F32> x,
                           ffi::ResultBuffer<ffi::DataType::F32> stats,
                           std::string_view config)
  {
    return AmgxSolveInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(stream, row_ptrs, col_indices, values, b, x, stats, config);
  }

  // Double implementation
  ffi::Error AmgxSolveImplDouble(cudaStream_t stream,
                                 ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                 ffi::Buffer<ffi::DataType::S32> col_indices,
                                 ffi::Buffer<ffi::DataType::F64> values,
                                 ffi::Buffer<ffi::DataType::F64> b,
                                 ffi::ResultBuffer<ffi::DataType::F64> x,
                                 ffi::ResultBuffer<ffi::DataType::F64> stats,
                                 std::string_view config)
  {
    return AmgxSolveInternal<double, ffi::DataType::F64, AMGX_mode_dDDI>(stream, row_ptrs, col_indices, values, b, x, stats, config);
  }

  // MPI Float implementation
  ffi::Error AmgxSolveMPIImpl(cudaStream_t stream,
                              ffi::Buffer<ffi::DataType::S32> row_ptrs,
                              ffi::Buffer<ffi::DataType::S64> col_indices,
                              ffi::Buffer<ffi::DataType::F32> values,
                              ffi::Buffer<ffi::DataType::F32> b,
                              ffi::Buffer<ffi::DataType::S32> nglobal,
                              ffi::Buffer<ffi::DataType::S32> comm_ptr,
                              ffi::Buffer<ffi::DataType::S32> lrank,
                              ffi::ResultBuffer<ffi::DataType::F32> x,
                              ffi::ResultBuffer<ffi::DataType::F32> stats,
                              std::string_view config)
  {
    return AmgxSolveMPIInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config);
  }

  // MPI Double implementation
  ffi::Error AmgxSolveMPIImplDouble(cudaStream_t stream,
                                    ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                    ffi::Buffer<ffi::DataType::S64> col_indices,
                                    ffi::Buffer<ffi::DataType::F64> values,
                                    ffi::Buffer<ffi::DataType::F64> b,
                                    ffi::Buffer<ffi::DataType::S32> nglobal,
                                    ffi::Buffer<ffi::DataType::S32> comm_ptr,
                                    ffi::Buffer<ffi::DataType::S32> lrank,
                                    ffi::ResultBuffer<ffi::DataType::F64> x,
                                    ffi::ResultBuffer<ffi::DataType::F64> stats,
                                    std::string_view config)
  {
    return AmgxSolveMPIInternal<double, ffi::DataType::F64, AMGX_mode_dDDI>(stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config);
  }

  // Register XLA FFI Handler (single-GPU)
  XLA_FFI_DEFINE_HANDLER(
      AmgxSolve,
      AmgxSolveImpl,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>() // CUDA stream context
          .Arg<ffi::Buffer<ffi::S32>>()             // row_ptrs
          .Arg<ffi::Buffer<ffi::S32>>()             // col_indices
          .Arg<ffi::Buffer<ffi::F32>>()             // values
          .Arg<ffi::Buffer<ffi::F32>>()             // b
          .Ret<ffi::Buffer<ffi::F32>>()             // x
          .Ret<ffi::Buffer<ffi::F32>>()             // stats
          .Attr<std::string_view>("config")         // config string
  );

  XLA_FFI_DEFINE_HANDLER(
      AmgxSolveDouble,
      AmgxSolveImplDouble,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>() // CUDA stream context
          .Arg<ffi::Buffer<ffi::S32>>()             // row_ptrs
          .Arg<ffi::Buffer<ffi::S32>>()             // col_indices
          .Arg<ffi::Buffer<ffi::F64>>()             // values
          .Arg<ffi::Buffer<ffi::F64>>()             // b
          .Ret<ffi::Buffer<ffi::F64>>()             // x
          .Ret<ffi::Buffer<ffi::F64>>()             // stats
          .Attr<std::string_view>("config")         // config string
  );

  // Register MPI handlers
  XLA_FFI_DEFINE_HANDLER(
      AmgxSolveMPI,
      AmgxSolveMPIImpl,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>() // CUDA stream context
          .Arg<ffi::Buffer<ffi::S32>>()             // row_ptrs
          .Arg<ffi::Buffer<ffi::S64>>()             // col_indices (GLOBAL, int64)
          .Arg<ffi::Buffer<ffi::F32>>()             // values
          .Arg<ffi::Buffer<ffi::F32>>()             // b (local)
          .Arg<ffi::Buffer<ffi::S32>>()             // nglobal
          .Arg<ffi::Buffer<ffi::S32>>()             // comm_ptr (2 x int32)
          .Arg<ffi::Buffer<ffi::S32>>()             // lrank
          .Ret<ffi::Buffer<ffi::F32>>()             // x (local)
          .Ret<ffi::Buffer<ffi::F32>>()             // stats
          .Attr<std::string_view>("config")         // config string
  );

  XLA_FFI_DEFINE_HANDLER(
      AmgxSolveMPIDouble,
      AmgxSolveMPIImplDouble,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>() // CUDA stream context
          .Arg<ffi::Buffer<ffi::S32>>()             // row_ptrs
          .Arg<ffi::Buffer<ffi::S64>>()             // col_indices (GLOBAL, int64)
          .Arg<ffi::Buffer<ffi::F64>>()             // values
          .Arg<ffi::Buffer<ffi::F64>>()             // b (local)
          .Arg<ffi::Buffer<ffi::S32>>()             // nglobal
          .Arg<ffi::Buffer<ffi::S32>>()             // comm_ptr (2 x int32)
          .Arg<ffi::Buffer<ffi::S32>>()             // lrank
          .Ret<ffi::Buffer<ffi::F64>>()             // x (local)
          .Ret<ffi::Buffer<ffi::F64>>()             // stats
          .Attr<std::string_view>("config")         // config string
  );

  // -------------------------------------------------------------------------
  // MPI AllGather Custom Call
  // -------------------------------------------------------------------------

  // Check if CUDA-aware MPI should be used (respects MPI4JAX convention)
  static bool use_cuda_aware_mpi()
  {
    static int cached = -1;
    if (cached == -1)
    {
      const char *env = std::getenv("MPI4JAX_USE_CUDA_MPI");
      if (env != nullptr)
      {
        cached = (std::string(env) == "1" || std::string(env) == "true") ? 1 : 0;
      }
      else
      {
        // Default: use host-staged MPI (safer, works with all MPI implementations)
        cached = 0;
      }
    }
    return cached == 1;
  }

  template <typename T, ffi::DataType DType>
  ffi::Error AmgxAllGatherInternal(cudaStream_t stream,
                                   ffi::Buffer<DType> sendbuf,
                                   ffi::Buffer<ffi::DataType::S32> recvcounts,
                                   ffi::Buffer<ffi::DataType::S32> displs,
                                   ffi::Buffer<ffi::DataType::S32> comm_ptr_buf,
                                   ffi::ResultBuffer<DType> recvbuf)
  {
    // Synchronize stream to ensure inputs are ready
    cudaStreamSynchronize(stream);

    // Get MPI Communicator
    int comm_ptr_parts[2];
    cudaMemcpy(comm_ptr_parts, comm_ptr_buf.typed_data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
    uint64_t comm_ptr_val = (static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[1])) << 32) |
                            static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[0]));
    MPI_Comm *mpi_comm = reinterpret_cast<MPI_Comm *>(comm_ptr_val);

    // Get Counts and Displacements (must be on host for MPI call)
    size_t nranks = recvcounts.element_count();
    std::vector<int> counts_h(nranks);
    std::vector<int> displs_h(nranks);

    cudaMemcpy(counts_h.data(), recvcounts.typed_data(), nranks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(displs_h.data(), displs.typed_data(), nranks * sizeof(int), cudaMemcpyDeviceToHost);

    int send_count = static_cast<int>(sendbuf.element_count());
    int total_recv_count = displs_h.back() + counts_h.back();
    MPI_Datatype mpi_type = (sizeof(T) == 8) ? MPI_DOUBLE : MPI_FLOAT;

    int err;
    if (use_cuda_aware_mpi())
    {
      // CUDA-aware MPI: pass GPU pointers directly
      err = MPI_Allgatherv(
          const_cast<T *>(sendbuf.typed_data()), send_count, mpi_type,
          recvbuf->typed_data(), counts_h.data(), displs_h.data(), mpi_type,
          *mpi_comm);
    }
    else
    {
      // Host-staged MPI (default, compatible with all MPI implementations)
      std::vector<T> send_host(send_count);
      std::vector<T> recv_host(total_recv_count);

      cudaMemcpy(send_host.data(), sendbuf.typed_data(), send_count * sizeof(T), cudaMemcpyDeviceToHost);

      err = MPI_Allgatherv(send_host.data(), send_count, mpi_type,
                           recv_host.data(), counts_h.data(), displs_h.data(), mpi_type,
                           *mpi_comm);

      if (err == MPI_SUCCESS)
      {
        cudaMemcpy(recvbuf->typed_data(), recv_host.data(), total_recv_count * sizeof(T), cudaMemcpyHostToDevice);
      }
    }

    if (err != MPI_SUCCESS)
    {
      return ffi::Error::Internal("MPI_Allgatherv failed");
    }

    return ffi::Error::Success();
  }

  ffi::Error AmgxAllGatherImpl(cudaStream_t stream,
                               ffi::Buffer<ffi::DataType::F32> sendbuf,
                               ffi::Buffer<ffi::DataType::S32> recvcounts,
                               ffi::Buffer<ffi::DataType::S32> displs,
                               ffi::Buffer<ffi::DataType::S32> comm_ptr,
                               ffi::ResultBuffer<ffi::DataType::F32> recvbuf)
  {
    return AmgxAllGatherInternal<float, ffi::DataType::F32>(stream, sendbuf, recvcounts, displs, comm_ptr, recvbuf);
  }

  ffi::Error AmgxAllGatherImplDouble(cudaStream_t stream,
                                     ffi::Buffer<ffi::DataType::F64> sendbuf,
                                     ffi::Buffer<ffi::DataType::S32> recvcounts,
                                     ffi::Buffer<ffi::DataType::S32> displs,
                                     ffi::Buffer<ffi::DataType::S32> comm_ptr,
                                     ffi::ResultBuffer<ffi::DataType::F64> recvbuf)
  {
    return AmgxAllGatherInternal<double, ffi::DataType::F64>(stream, sendbuf, recvcounts, displs, comm_ptr, recvbuf);
  }

  XLA_FFI_DEFINE_HANDLER(
      AmgxAllGather,
      AmgxAllGatherImpl,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>()
          .Arg<ffi::Buffer<ffi::F32>>() // sendbuf
          .Arg<ffi::Buffer<ffi::S32>>() // recvcounts
          .Arg<ffi::Buffer<ffi::S32>>() // displs
          .Arg<ffi::Buffer<ffi::S32>>() // comm_ptr
          .Ret<ffi::Buffer<ffi::F32>>() // recvbuf
  );

  XLA_FFI_DEFINE_HANDLER(
      AmgxAllGatherDouble,
      AmgxAllGatherImplDouble,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>()
          .Arg<ffi::Buffer<ffi::F64>>() // sendbuf
          .Arg<ffi::Buffer<ffi::S32>>() // recvcounts
          .Arg<ffi::Buffer<ffi::S32>>() // displs
          .Arg<ffi::Buffer<ffi::S32>>() // comm_ptr
          .Ret<ffi::Buffer<ffi::F64>>() // recvbuf
  );

} // namespace

PYBIND11_MODULE(_amgx, m)
{
  m.def("get_amgx_solve_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolve)); });
  m.def("get_amgx_solve_double_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolveDouble)); });
  m.def("get_amgx_solve_mpi_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolveMPI)); });
  m.def("get_amgx_solve_mpi_double_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolveMPIDouble)); });
  m.def("get_amgx_allgather_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxAllGather)); });
  m.def("get_amgx_allgather_double_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxAllGatherDouble)); });

  m.def("initialize", &EnsureAmgxInitialized);
  m.def("finalize", &AmgxFinalize);
  m.def("clear_solver_cache", []()
        { GetSolverCache().clear(DestroyResources); });
}
