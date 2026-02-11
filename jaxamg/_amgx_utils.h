/*
 * Internal header for AmgX resources and cache management.
 * Included by _amgx.cc.
 */

#ifndef JAXAMG_AMGX_RESOURCES_H_
#define JAXAMG_AMGX_RESOURCES_H_

#include <cuda_runtime.h>
#include <amgx_c.h>
#include <mpi.h>
#include <xla/ffi/api/ffi.h>
#include <cstdio>
#include <mutex>
#include <atomic>
#include <string>
#include <fstream>
#include <vector>
#include <list>
#include <unordered_map>
#include <functional>
#include <utility>
#include <string_view>

namespace ffi = xla::ffi;

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

  // FNV-1a hash of byte sequences.
  inline size_t fnv1a_hash(const void *data, size_t len)
  {
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    size_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < len; ++i)
    {
      hash ^= bytes[i];
      hash *= 1099511628211ULL;
    }
    return hash;
  }

  struct MPICacheKey
  {
    // No device pointers: JAX eager calls get new addresses each time, causing
    // cache thrashing. Value-based keys + structure_hash (FNV-1a of row_ptrs
    // content) ensure stable hits and correct structural identity for
    // AMGX_matrix_replace_coefficients on the cache-hit path.
    int n_local;
    int n_global;
    int nnz;
    int lrank;
    int mode; // AMGX_Mode (dFFI vs dDDI) to distinguish f32/f64
    uint64_t comm_ptr;
    size_t structure_hash;
    std::string config;

    bool operator==(const MPICacheKey &other) const
    {
      return n_local == other.n_local &&
             n_global == other.n_global &&
             nnz == other.nnz &&
             lrank == other.lrank &&
             mode == other.mode &&
             comm_ptr == other.comm_ptr &&
             structure_hash == other.structure_hash &&
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

  template <>
  struct hash<MPICacheKey>
  {
    size_t operator()(const MPICacheKey &k) const
    {
      size_t h1 = hash<int>()(k.n_local);
      size_t h2 = hash<int>()(k.n_global);
      size_t h3 = hash<int>()(k.nnz);
      size_t h4 = hash<int>()(k.lrank);
      size_t h5 = hash<int>()(k.mode);
      size_t h6 = hash<uint64_t>()(k.comm_ptr);
      size_t h7 = hash<size_t>()(k.structure_hash);
      size_t h8 = hash<string>()(k.config);

      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^
             (h5 << 4) ^ (h6 << 5) ^ (h7 << 6) ^ (h8 << 7);
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

  struct CachedMPIResources
  {
    AMGX_config_handle cfg = nullptr;
    AMGX_resources_handle rsrc = nullptr;
    AMGX_matrix_handle A = nullptr;
    AMGX_solver_handle solver = nullptr;
    AMGX_vector_handle x_vec = nullptr;
    AMGX_vector_handle b_vec = nullptr;
    void *values_buf = nullptr; // Persistent buffer for replace_coefficients
    bool owns_resources = false; // True if isolated (Cache=0), False if shared (Cache>=1)
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

  LRUCache<MPICacheKey, CachedMPIResources>& GetMPISolverCache()
  {
    static auto* cache = []() {
      size_t capacity = 1;
      const char* env_val = std::getenv("JAXAMG_CACHE_SIZE");
      if (env_val) {
        try {
          capacity = std::stoul(env_val);
        } catch (...) {
          capacity = 1;
        }
      }

      return new LRUCache<MPICacheKey, CachedMPIResources>(capacity);
    }();
    return *cache;
  }

  inline AMGX_RC CreateAmgxConfigFromStringOrFile(std::string_view config, AMGX_config_handle *cfg)
  {
    std::string config_str(config);
    std::ifstream file_check(config_str);
    bool is_file = file_check.good();
    file_check.close();

    if (is_file)
    {
      return AMGX_config_create_from_file(cfg, config_str.c_str());
    }
    return AMGX_config_create(cfg, config_str.c_str());
  }

  inline void DestroyResources(CachedResources &res)
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

  inline void DestroyMPIResources(CachedMPIResources &res)
  {
    try
    {
      if (cudaDeviceSynchronize() != cudaSuccess) {
        return;
      }

      if (res.solver) AMGX_solver_destroy(res.solver);
      if (res.b_vec) AMGX_vector_destroy(res.b_vec);
      if (res.x_vec) AMGX_vector_destroy(res.x_vec);
      if (res.A) AMGX_matrix_destroy(res.A);
      // Only destroy resources handle in Isolated Mode (Cache=0).
      if (res.owns_resources) {
        if (res.rsrc) AMGX_resources_destroy(res.rsrc);
      }
      if (res.cfg) AMGX_config_destroy(res.cfg);
      if (res.values_buf) cudaFree(res.values_buf);
    }
    catch (...)
    {
    }
  }

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

  // Singleton for global MPI AmgX resource management. Shares one
  // AMGX_resources_handle across all MPI cache entries so multiple
  // matrix/solver pairs can coexist on the same communicator.
  class GlobalMPIResources {
  public:
      static GlobalMPIResources& Get() {
          static GlobalMPIResources* instance = new GlobalMPIResources();
          return *instance;
      }

      AMGX_resources_handle GetHandle(AMGX_config_handle cfg,
                                       MPI_Comm *comm, int ndevs, int *devs) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!handle_) {
              AMGX_SAFE_CALL_VOID(AMGX_resources_create(&handle_, cfg, comm, ndevs, devs));
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
      GlobalMPIResources() : handle_(nullptr) {}
      ~GlobalMPIResources() { Destroy(); }

      AMGX_resources_handle handle_;
      std::mutex mutex_;
  };

  std::once_flag g_amgx_init_flag;

  // Custom callback to suppress AmgX library output (banners, version info)
  inline void PrintCallback(const char *msg, int length)
  {
    // No-op: Output is fully suppressed to keep stdout clean for the user.
    return;
  }

  inline void EnsureAmgxInitialized()
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

  inline void AmgxFinalize()
  {
    // Clear caches and destroy global resources. Safe to call multiple times:
    // clear() on an empty cache is a no-op, Destroy() checks for null.
    GetSolverCache().clear(DestroyResources);
    GetMPISolverCache().clear(DestroyMPIResources);
    GlobalResources::Get().Destroy();
    GlobalMPIResources::Get().Destroy();
  }

  // Check if CUDA-aware MPI should be used (respects MPI4JAX convention)
  inline bool use_cuda_aware_mpi()
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

} // namespace

#endif // JAXAMG_AMGX_UTILS_H_
