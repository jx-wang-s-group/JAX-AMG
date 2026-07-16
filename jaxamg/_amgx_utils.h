/*
 * Internal header for AmgX resources and cache management.
 * Included by _amgx.cc.
 */

#ifndef JAXAMG_AMGX_RESOURCES_H_
#define JAXAMG_AMGX_RESOURCES_H_

#include <cuda_runtime.h>
#include <amgx_c.h>
#ifdef JAXAMG_WITH_MPI
#include <mpi.h>
#endif
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

// Forward declarations for global stats capture variables (defined in _amgx.cc)
extern std::string g_stats_string;
extern bool g_capture_stats;

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

    size_t size() const
    {
      std::lock_guard<std::mutex> lock(mutex_);
      return cache_map_.size();
    }

    size_t capacity() const
    {
      return capacity_;
    }

    std::vector<Key> snapshot_keys() const
    {
      std::lock_guard<std::mutex> lock(mutex_);
      std::vector<Key> keys;
      keys.reserve(lru_list_.size());
      for (const auto &pair : lru_list_)
      {
        keys.push_back(pair.first);
      }
      return keys;
    }

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

    // Evict LRU entries so there is room for `incoming` new entries.
    // Returns true if any eviction happened.
    bool evict_lru_if_needed(size_t incoming, std::function<void(Value &)> destructor)
    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (capacity_ == 0)
      {
        return false;
      }

      bool evicted = false;
      while (cache_map_.size() + incoming > capacity_ && !lru_list_.empty())
      {
        auto last = lru_list_.end();
        --last;
        if (destructor)
        {
          destructor(last->second);
        }
        cache_map_.erase(last->first);
        lru_list_.pop_back();
        evicted = true;
      }

      return evicted;
    }

  private:
    size_t capacity_;
    std::list<std::pair<Key, Value>> lru_list_;
    std::unordered_map<Key, typename std::list<std::pair<Key, Value>>::iterator> cache_map_;
    mutable std::mutex mutex_;
  };

  struct CacheKey
  {
    int n_rows;
    int nnz;
    int mode; // AMGX_Mode (dFFI vs dDDI)
    bool transpose_solve;
    size_t structure_hash; // FNV-1a of row_ptrs + col_indices content
    std::string config;

    bool operator==(const CacheKey &other) const
    {
      return n_rows == other.n_rows &&
             nnz == other.nnz &&
             mode == other.mode &&
             transpose_solve == other.transpose_solve &&
             structure_hash == other.structure_hash &&
             config == other.config;
    }
  };

  // Word-wise FNV-1a-style hash: four lanes break the multiply dependency chain
  // so hashing the structure (row_ptrs + col_indices, hundreds of MB at n~1e7)
  // isn't a bottleneck on repeated solves. Not canonical FNV-1a but a
  // deterministic, content-sensitive digest, which is all the cache key needs.
  // Pass `seed` to chain buffers into one digest; inputs are >=4-byte aligned.
  inline size_t fnv1a_hash(const void *data, size_t len,
                           size_t seed = 14695981039346656037ULL)
  {
    constexpr size_t PRIME = 1099511628211ULL;
    const uint8_t *bytes = static_cast<const uint8_t *>(data);
    const uint32_t *words = static_cast<const uint32_t *>(data);
    const size_t nwords = len / 4;
    size_t h0 = seed, h1 = seed ^ 0x9e3779b97f4a7c15ULL,
           h2 = seed + 1ULL, h3 = seed ^ 0xff51afd7ed558ccdULL;
    size_t i = 0;
    for (; i + 4 <= nwords; i += 4)
    {
      h0 = (h0 ^ words[i]) * PRIME;
      h1 = (h1 ^ words[i + 1]) * PRIME;
      h2 = (h2 ^ words[i + 2]) * PRIME;
      h3 = (h3 ^ words[i + 3]) * PRIME;
    }
    size_t hash = (h0 * PRIME) ^ (h1 * PRIME) ^ (h2 * PRIME) ^ (h3 * PRIME);
    for (; i < nwords; ++i)
    {
      hash = (hash ^ words[i]) * PRIME;
    }
    for (size_t b = nwords * 4; b < len; ++b)
    {
      hash = (hash ^ bytes[b]) * PRIME;
    }
    return hash;
  }

  struct MPICacheKey
  {
    // No device pointers: JAX eager calls get new addresses each time, causing
    // cache thrashing. Value-based keys + structure_hash (FNV-1a of row_ptrs +
    // col_indices content) ensure stable hits and correct structural identity
    // for AMGX_matrix_replace_coefficients on the cache-hit path.
    int n_local;
    int n_global;
    int nnz;
    int lrank;
    int mode; // AMGX_Mode (dFFI vs dDDI)
    bool transpose_solve;
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
             transpose_solve == other.transpose_solve &&
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
      size_t h1 = hash<int>()(k.n_rows);
      size_t h2 = hash<int>()(k.nnz);
      size_t h3 = hash<int>()(k.mode);
      size_t h4 = hash<bool>()(k.transpose_solve);
      size_t h5 = hash<size_t>()(k.structure_hash);
      size_t h6 = hash<string>()(k.config);

      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4) ^ (h6 << 5);
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
      size_t h6 = hash<bool>()(k.transpose_solve);
      size_t h7 = hash<uint64_t>()(k.comm_ptr);
      size_t h8 = hash<size_t>()(k.structure_hash);
      size_t h9 = hash<string>()(k.config);

      return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^
             (h5 << 4) ^ (h6 << 5) ^ (h7 << 6) ^ (h8 << 7) ^ (h9 << 8);
    }
  };
} // namespace std

namespace
{
  struct CachedResources
  {
    AMGX_config_handle cfg = nullptr;
    AMGX_resources_handle rsrc = nullptr;
    AMGX_matrix_handle A = nullptr;
    AMGX_solver_handle solver = nullptr;
    AMGX_vector_handle x_vec = nullptr;
    AMGX_vector_handle b_vec = nullptr;
    void *values_buf = nullptr;            // MPI replace_coefficients buffer (null for non-MPI)
    void *row_ptrs_buf = nullptr;          // MPI: cache-owned copy of local row_ptrs (see cold path)
    void *col_indices_buf = nullptr;       // MPI: cache-owned int32 copy of local global col_indices
    void *transpose_row_ptrs = nullptr;    // transpose_solve mode only
    void *transpose_col_indices = nullptr; // transpose_solve mode only
    void *transpose_values = nullptr;      // transpose_solve mode only
    bool owns_resources = false;           // true if isolated (JAXAMG_CACHE_SIZE=0)
  };

  // Parse JAXAMG_CACHE_SIZE env var (default: 1).
  inline size_t GetCacheCapacity()
  {
    const char *env_val = std::getenv("JAXAMG_CACHE_SIZE");
    if (env_val)
    {
      try { return std::stoul(env_val); }
      catch (...) {}
    }
    return 1;
  }

  inline bool IsIsolatedMode()
  {
    return GetCacheCapacity() == 0;
  }

  // Global cache instances (heap-allocated to persist until explicit finalization).
  LRUCache<CacheKey, CachedResources>& GetSolverCache()
  {
    static auto* cache = new LRUCache<CacheKey, CachedResources>(GetCacheCapacity());
    return *cache;
  }

  LRUCache<MPICacheKey, CachedResources>& GetMPISolverCache()
  {
    static auto* cache = new LRUCache<MPICacheKey, CachedResources>(GetCacheCapacity());
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

      // Destroy in reverse order of creation.
      if (res.solver) AMGX_solver_destroy(res.solver);
      if (res.b_vec) AMGX_vector_destroy(res.b_vec);
      if (res.x_vec) AMGX_vector_destroy(res.x_vec);
      if (res.A) AMGX_matrix_destroy(res.A);

      if (res.owns_resources) {
          if (res.rsrc) AMGX_resources_destroy(res.rsrc);
          if (res.cfg) AMGX_config_destroy(res.cfg);
      }
      // In shared mode (owns_resources == false), the founding config is
      // owned by Global{MPI}Resources and destroyed in its Destroy().
      // Non-founding configs are tiny and cleaned up at process exit.
      if (res.values_buf) cudaFree(res.values_buf);
      if (res.row_ptrs_buf) cudaFree(res.row_ptrs_buf);
      if (res.col_indices_buf) cudaFree(res.col_indices_buf);
      if (res.transpose_row_ptrs) cudaFree(res.transpose_row_ptrs);
      if (res.transpose_col_indices) cudaFree(res.transpose_col_indices);
      if (res.transpose_values) cudaFree(res.transpose_values);
    }
    catch (...)
    {
      // Swallow all exceptions during destruction to prevent 'terminate' on exit.
    }
  }

  // Cleans up freshly created (not-yet-cached) resources on any early return
  // -- AMGX_SAFE_CALL failures, allocation failures, or a failed solve. Arm it
  // when entering the fresh-creation path and disarm just before the resources
  // are handed to the cache; cache-owned entries are never destroyed here.
  struct FreshResourceGuard
  {
    CachedResources *res;
    bool armed = false;
    ~FreshResourceGuard()
    {
      if (armed && res)
      {
        DestroyResources(*res);
      }
    }
  };

  // Singleton for global AmgX resource management.
  // Owns the founding AMGX_config_handle that the resources were created with,
  // because the resources handle internally references it.
  class GlobalResources {
  public:
      static GlobalResources& Get() {
          static GlobalResources* instance = new GlobalResources();
          return *instance;
      }

      AMGX_resources_handle GetHandle(AMGX_config_handle cfg) {
          std::lock_guard<std::mutex> lock(mutex_);
          if (!handle_) {
              AMGX_SAFE_CALL_VOID(AMGX_resources_create_simple(&handle_, cfg));
              cfg_ = cfg;
          }
          return handle_;
      }

      void Destroy() {
          std::lock_guard<std::mutex> lock(mutex_);
          if (handle_) {
              AMGX_SAFE_CALL_VOID(AMGX_resources_destroy(handle_));
              handle_ = nullptr;
          }
          if (cfg_) {
              AMGX_SAFE_CALL_VOID(AMGX_config_destroy(cfg_));
              cfg_ = nullptr;
          }
      }

  private:
      GlobalResources() : handle_(nullptr), cfg_(nullptr) {}
      ~GlobalResources() { Destroy(); }

      AMGX_resources_handle handle_;
      AMGX_config_handle cfg_;
      std::mutex mutex_;
  };

#ifdef JAXAMG_WITH_MPI
  // Singleton for global MPI AmgX resource management. Keeps one
  // AMGX_resources_handle per communicator: solves on the same communicator
  // share it, distinct communicators each get their own (one handle reused
  // across communicators would run AmgX's collectives on the wrong one). Owns
  // the founding config handle per communicator (as GlobalResources does).
  class GlobalMPIResources {
  public:
      static GlobalMPIResources& Get() {
          static GlobalMPIResources* instance = new GlobalMPIResources();
          return *instance;
      }

      AMGX_resources_handle GetHandle(AMGX_config_handle cfg,
                                       MPI_Comm *comm, int ndevs, int *devs) {
          std::lock_guard<std::mutex> lock(mutex_);
          // Key on the communicator's identity (the same pointer Python passes as
          // comm_ptr), so each communicator gets its own resources handle.
          uint64_t key = reinterpret_cast<uint64_t>(comm);
          auto it = handles_.find(key);
          if (it != handles_.end()) {
              return it->second.handle;
          }
          AMGX_resources_handle handle = nullptr;
          AMGX_SAFE_CALL_VOID(AMGX_resources_create(&handle, cfg, comm, ndevs, devs));
          handles_[key] = {handle, cfg};
          return handle;
      }

      void Destroy() {
          std::lock_guard<std::mutex> lock(mutex_);
          for (auto &kv : handles_) {
              if (kv.second.handle) AMGX_SAFE_CALL_VOID(AMGX_resources_destroy(kv.second.handle));
              if (kv.second.cfg) AMGX_SAFE_CALL_VOID(AMGX_config_destroy(kv.second.cfg));
          }
          handles_.clear();
      }

  private:
      struct Entry {
          AMGX_resources_handle handle = nullptr;
          AMGX_config_handle cfg = nullptr;
      };
      GlobalMPIResources() {}
      ~GlobalMPIResources() { Destroy(); }

      std::unordered_map<uint64_t, Entry> handles_;
      std::mutex mutex_;
  };
#endif // JAXAMG_WITH_MPI

  std::once_flag g_amgx_init_flag;

  // Custom callback to suppress AmgX library output (banners, version info)
  inline void PrintCallback(const char *msg, int length)
  {
    if (::g_capture_stats) {
        ::g_stats_string.append(msg, length);
    }
  }

  struct StatsCaptureGuard {
      bool old_val;
      StatsCaptureGuard(bool capture) {
          old_val = ::g_capture_stats;
          ::g_capture_stats = capture;
          if (capture) {
              ::g_stats_string.clear();
          }
      }
      ~StatsCaptureGuard() {
          ::g_capture_stats = old_val;
      }
  };

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
    GetMPISolverCache().clear(DestroyResources);
    GlobalResources::Get().Destroy();
#ifdef JAXAMG_WITH_MPI
    GlobalMPIResources::Get().Destroy();
#endif
  }

} // namespace

#endif // JAXAMG_AMGX_UTILS_H_
