/*
 * Internal header for AmgX solver implementations.
 * Included by _amgx.cc.
 */

#ifndef JAXAMG_AMGX_SOLVERS_H_
#define JAXAMG_AMGX_SOLVERS_H_

#include <cuda_runtime.h>
#include <cusparse.h>
#include <amgx_c.h>
#include <xla/ffi/api/ffi.h>
#ifdef JAXAMG_WITH_MPI
#include <mpi.h>
#endif
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <limits>
#include <type_traits>

#include "_amgx_utils.h"

namespace ffi = xla::ffi;

namespace
{

  // Reused cuSPARSE handle for the single-GPU transpose: created once and reused
  // (stream rebound per call), avoiding a create/destroy on every backward pass.
  // Left to be reclaimed at process exit (a single handle, independent of AmgX).
  inline cusparseHandle_t GetCusparseHandle()
  {
    static cusparseHandle_t handle = []()
    {
      cusparseHandle_t h = nullptr;
      if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS)
      {
        h = nullptr;
      }
      return h;
    }();
    return handle;
  }

  // The scratch workspace lives in the solver-cache entry (workspace_slot /
  // workspace_size_slot) and is only (re)allocated when the required size
  // grows: the transpose reruns on every backward solve whose values changed,
  // and a per-call cudaMalloc/cudaFree pair is comparatively expensive
  // (cudaFree also implicitly synchronizes the device).
  template <typename T>
  inline const char *CsrTransposeDevice(cudaStream_t stream,
                                        int n_rows,
                                        int nnz,
                                        const int *row_ptrs,
                                        const int *col_indices,
                                        const T *values,
                                        int *row_ptrs_t,
                                        int *col_indices_t,
                                        T *values_t,
                                        void **workspace_slot,
                                        size_t *workspace_size_slot)
  {
    cusparseHandle_t handle = GetCusparseHandle();
    if (handle == nullptr)
    {
      return "cusparseCreate failed";
    }
    if (cusparseSetStream(handle, stream) != CUSPARSE_STATUS_SUCCESS)
    {
      return "cusparseSetStream failed";
    }

    cudaDataType dtype = std::is_same<T, double>::value ? CUDA_R_64F : CUDA_R_32F;
    size_t buffer_size = 0;
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle,
        n_rows,
        n_rows,
        nnz,
        values,
        row_ptrs,
        col_indices,
        values_t,
        row_ptrs_t,
        col_indices_t,
        dtype,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        &buffer_size);
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
      return "cusparseCsr2cscEx2_bufferSize failed";
    }

    if (buffer_size > *workspace_size_slot)
    {
      if (*workspace_slot)
      {
        cudaFree(*workspace_slot);
        *workspace_slot = nullptr;
        *workspace_size_slot = 0;
      }
      if (cudaMalloc(workspace_slot, buffer_size) != cudaSuccess)
      {
        return "cudaMalloc failed for cusparse transpose workspace";
      }
      *workspace_size_slot = buffer_size;
    }

    status = cusparseCsr2cscEx2(
        handle,
        n_rows,
        n_rows,
        nnz,
        values,
        row_ptrs,
        col_indices,
        values_t,
        row_ptrs_t,
        col_indices_t,
        dtype,
        CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO,
        CUSPARSE_CSR2CSC_ALG1,
        *workspace_slot);

    if (status != CUSPARSE_STATUS_SUCCESS)
    {
      return "cusparseCsr2cscEx2 failed";
    }

    if (cudaStreamSynchronize(stream) != cudaSuccess)
    {
      return "cudaStreamSynchronize failed after csr transpose";
    }

    return nullptr;
  }

  // Assemble the host-side stats buffer: [iterations, final_residual, status,
  // residual_history...]. The history capacity is whatever the caller allocated
  // beyond the first three entries (max_iters + 1 slots, or zero for callers
  // that predate residual history); slots past iteration `iters` stay NaN so
  // Python can tell padding from recorded residuals.
  template <typename T>
  inline std::vector<T> CollectSolveStats(AMGX_solver_handle solver,
                                          size_t stats_len,
                                          int iters,
                                          double residual,
                                          AMGX_SOLVE_STATUS status)
  {
    std::vector<T> stats_host(stats_len, std::numeric_limits<T>::quiet_NaN());
    if (stats_len >= 3)
    {
      stats_host[0] = static_cast<T>(iters);
      stats_host[1] = static_cast<T>(residual);
      stats_host[2] = static_cast<T>(status);
      const size_t n_hist = std::min(stats_len - 3, static_cast<size_t>(iters) + 1);
      for (size_t i = 0; i < n_hist; ++i)
      {
        double r = 0.0;
        if (AMGX_solver_get_iteration_residual(solver, static_cast<int>(i), 0, &r) == AMGX_RC_OK)
        {
          stats_host[3 + i] = static_cast<T>(r);
        }
      }
    }
    return stats_host;
  }

  /*
   * AmgxSolveInternal: Templated core implementation of the XLA FFI handler.
   * Supports both float (AMGX_mode_dFFI) and double (AMGX_mode_dDDI).
   */
  template <typename T, ffi::DataType DType, AMGX_Mode Mode>
  inline ffi::Error AmgxSolveInternal(cudaStream_t stream,
                                      ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                      ffi::Buffer<ffi::DataType::S32> col_indices,
                                      ffi::Buffer<DType> values,
                                      ffi::Buffer<DType> b,
                                      ffi::ResultBuffer<DType> x,
                                      ffi::ResultBuffer<DType> stats,
                                      std::string_view config,
                                      int32_t transpose_solve,
                                      int32_t return_stats,
                                      int32_t reuse_setup)
  {
    EnsureAmgxInitialized();

    // Ensure input buffers are ready.
    cudaStreamSynchronize(stream);

    CachedResources res;
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

    // Cast to raw pointers to avoid host transfers.
    int *row_ptrs_data = const_cast<int *>(row_ptrs.typed_data());
    int *col_indices_data = const_cast<int *>(col_indices.typed_data());
    T *values_data = const_cast<T *>(values.typed_data());
    T *b_data = const_cast<T *>(b.typed_data());
    T *x_data = x->typed_data();
    T *stats_data = stats->typed_data();

    const int n_rows = static_cast<int>(b.dimensions().size() > 0 ? b.dimensions()[0] : 0);
    const int nnz = static_cast<int>(values.element_count());

    // The cache-hit path only replaces coefficient values, so the key must
    // capture the full sparsity pattern (row_ptrs + col_indices) for a changed
    // pattern to miss and trigger a fresh setup. Reuse host buffers across calls
    // to avoid reallocating these large arrays on every solve.
    static thread_local std::vector<int> h_row_ptrs, h_col_indices;
    if (static_cast<int>(h_row_ptrs.size()) < n_rows + 1)
      h_row_ptrs.resize(n_rows + 1);
    if (static_cast<int>(h_col_indices.size()) < nnz)
      h_col_indices.resize(nnz);
    cudaMemcpyAsync(h_row_ptrs.data(), row_ptrs_data,
                    (n_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_col_indices.data(), col_indices_data,
                    nnz * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    size_t structure_hash = fnv1a_hash(h_row_ptrs.data(), (n_rows + 1) * sizeof(int));
    structure_hash = fnv1a_hash(h_col_indices.data(), nnz * sizeof(int), structure_hash);

    CacheKey key = {n_rows, nnz, static_cast<int>(Mode), transpose_solve != 0, structure_hash, std::string(config)};
    bool cache_hit = GetSolverCache().get(key, res);

    // Destroys freshly created resources on any early return below (armed only
    // on the cache-miss path; disarmed once the cache takes ownership).
    FreshResourceGuard fresh_guard{&res};

    bool reuse_success = false;

    if (cache_hit)
    {
      if (transpose_solve != 0)
      {
        if (!res.transpose_row_ptrs || !res.transpose_col_indices || !res.transpose_values)
        {
          return ffi::Error::Internal("transpose_solve cache entry missing transpose buffers");
        }
        const char *transpose_err = CsrTransposeDevice<T>(
            stream,
            n_rows,
            nnz,
            row_ptrs_data,
            col_indices_data,
            values_data,
            static_cast<int *>(res.transpose_row_ptrs),
            static_cast<int *>(res.transpose_col_indices),
            static_cast<T *>(res.transpose_values),
            &res.transpose_workspace,
            &res.transpose_workspace_size);
        if (transpose_err != nullptr)
        {
          return ffi::Error::Internal(transpose_err);
        }
        AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(
            res.A, n_rows, (int)values.element_count(), static_cast<T *>(res.transpose_values), nullptr));
      }
      else
      {
        AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(
            res.A, n_rows, (int)values.element_count(), values_data, nullptr));
      }

      // reuse_setup keeps the existing hierarchy while still refreshing the fine matrix.
      if (!reuse_setup)
        AMGX_SAFE_CALL(AMGX_solver_resetup(res.solver, res.A));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_rows, 1, b_data));
      AMGX_SAFE_CALL(AMGX_vector_set_zero(res.x_vec, n_rows, 1));

      reuse_success = true;
    }

    StatsCaptureGuard capture_guard(return_stats != 0);

    if (!cache_hit)
    {
      fresh_guard.armed = true;

      AMGX_SAFE_CALL(CreateAmgxConfigFromStringOrFile(config, &res.cfg));

      if (IsIsolatedMode())
      {
        res.owns_resources = true;
        AMGX_SAFE_CALL(AMGX_resources_create_simple(&res.rsrc, res.cfg));
      }
      else
      {
        res.owns_resources = false;
        res.rsrc = GlobalResources::Get().GetHandle(res.cfg);
      }

      AMGX_SAFE_CALL(AMGX_matrix_create(&res.A, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.x_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.b_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_solver_create(&res.solver, res.rsrc, Mode, res.cfg));

      if (transpose_solve != 0)
      {
        if (cudaMalloc(&res.transpose_row_ptrs, (n_rows + 1) * sizeof(int)) != cudaSuccess)
        {
          return ffi::Error::Internal("cudaMalloc failed for transpose row_ptrs");
        }
        if (cudaMalloc(&res.transpose_col_indices, nnz * sizeof(int)) != cudaSuccess)
        {
          return ffi::Error::Internal("cudaMalloc failed for transpose col_indices");
        }
        if (cudaMalloc(&res.transpose_values, nnz * sizeof(T)) != cudaSuccess)
        {
          return ffi::Error::Internal("cudaMalloc failed for transpose values");
        }

        const char *transpose_err = CsrTransposeDevice<T>(
            stream,
            n_rows,
            nnz,
            row_ptrs_data,
            col_indices_data,
            values_data,
            static_cast<int *>(res.transpose_row_ptrs),
            static_cast<int *>(res.transpose_col_indices),
            static_cast<T *>(res.transpose_values),
            &res.transpose_workspace,
            &res.transpose_workspace_size);
        if (transpose_err != nullptr)
        {
          return ffi::Error::Internal(transpose_err);
        }

        AMGX_SAFE_CALL(AMGX_matrix_upload_all(
            res.A,
            n_rows,
            (int)values.element_count(),
            1,
            1,
            static_cast<int *>(res.transpose_row_ptrs),
            static_cast<int *>(res.transpose_col_indices),
            static_cast<T *>(res.transpose_values),
            nullptr));
      }
      else
      {
        AMGX_SAFE_CALL(AMGX_matrix_upload_all(
            res.A,
            n_rows,
            (int)values.element_count(),
            1,
            1,
            row_ptrs_data,
            col_indices_data,
            values_data,
            nullptr));
      }

      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_rows, 1, b_data));
      AMGX_SAFE_CALL(AMGX_vector_set_zero(res.x_vec, n_rows, 1));
      AMGX_SAFE_CALL(AMGX_solver_setup(res.solver, res.A));
    }

    // Solve System
    AMGX_SAFE_CALL(AMGX_solver_solve(res.solver, res.b_vec, res.x_vec));

    // Retrieve Results & Statistics
    AMGX_SOLVE_STATUS status;
    AMGX_SAFE_CALL(AMGX_solver_get_status(res.solver, &status));

    if (status == AMGX_SOLVE_FAILED)
    {
      // fresh_guard destroys not-yet-cached resources; cached entries stay.
      return ffi::Error::Internal("AmgX solve failed");
    }

    int iters = 0;
    double residual = 0.0;

    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(res.solver, &iters));

    AMGX_RC res_rc = AMGX_solver_get_iteration_residual(res.solver, iters, 0, &residual);
    if (res_rc != AMGX_RC_OK)
    {
      residual = -1.0;
    }

    std::vector<T> stats_host = CollectSolveStats<T>(
        res.solver, stats->element_count(), iters, residual, status);
    cudaMemcpyAsync(stats_data, stats_host.data(), stats_host.size() * sizeof(T),
                    cudaMemcpyHostToDevice, stream);

    AMGX_SAFE_CALL(AMGX_vector_download(res.x_vec, x_data));

    // 7. Store in Cache (if new); the cache takes ownership from the guard.
    if (!cache_hit)
    {
      fresh_guard.armed = false;
      GetSolverCache().put(key, res, DestroyResources);
    }

    // AMGX_vector_download copies x with a plain cudaMemcpy on the legacy
    // default stream, which is NOT ordered against XLA's non-blocking stream.
    // Without this sync XLA can consume the output buffer before the copy
    // lands (stale zeros/garbage under GPU contention). Mirrors the MPI path.
    cudaDeviceSynchronize();

    return ffi::Error::Success();
  }

#ifdef JAXAMG_WITH_MPI
  /*
   * AmgxSolveMPIInternal: MPI-aware templated core implementation.
   * Uses AMGX_resources_create() with MPI communicator and
   * AMGX_matrix_upload_all_global() for distributed matrices.
   */
  template <typename T, ffi::DataType DType, AMGX_Mode Mode>
  inline ffi::Error AmgxSolveMPIInternal(cudaStream_t stream,
                                         ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                         ffi::Buffer<ffi::DataType::S64> col_indices,
                                         ffi::Buffer<DType> values,
                                         ffi::Buffer<DType> b,
                                         ffi::Buffer<ffi::DataType::S32> nglobal_buf,
                                         ffi::Buffer<ffi::DataType::S32> comm_ptr_buf,
                                         ffi::Buffer<ffi::DataType::S32> lrank_buf,
                                         ffi::ResultBuffer<DType> x,
                                         ffi::ResultBuffer<DType> stats,
                                         std::string_view config,
                                         int32_t transpose_solve,
                                         int32_t return_stats,
                                         int32_t reuse_setup)
  {
    if (transpose_solve != 0)
    {
      return ffi::Error::Internal(
          "transpose_solve is not supported in the MPI FFI path");
    }

    EnsureAmgxInitialized();

    int device;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
      return ffi::Error::Internal("cudaGetDevice failed");
    }

    if (cudaSetDevice(device) != cudaSuccess)
    {
      return ffi::Error::Internal("cudaSetDevice failed");
    }

    cudaStreamSynchronize(stream);
    int nglobal_host;
    int comm_ptr_parts[2];
    int lrank_host;

    cudaMemcpy(&nglobal_host, nglobal_buf.typed_data(), sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(comm_ptr_parts, comm_ptr_buf.typed_data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lrank_host, lrank_buf.typed_data(), sizeof(int), cudaMemcpyDeviceToHost);

    // Reconstruct 64-bit MPI_Comm* from two 32-bit halves (passed via JAX S32 buffers)
    uint64_t comm_ptr_val = (static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[1])) << 32) |
                            static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[0]));
    MPI_Comm *mpi_comm = reinterpret_cast<MPI_Comm *>(comm_ptr_val);

    int *row_ptrs_data = const_cast<int *>(row_ptrs.typed_data());
    int64_t *col_indices_data = const_cast<int64_t *>(col_indices.typed_data());
    T *values_data = const_cast<T *>(values.typed_data());
    T *b_data = const_cast<T *>(b.typed_data());
    T *x_data = x->typed_data();
    T *stats_data = stats->typed_data();

    const int n_local = static_cast<int>(b.dimensions().size() > 0 ? b.dimensions()[0] : 0);
    const int nnz = static_cast<int>(values.element_count());

    CachedResources res;

    // Hash row_ptrs and the (global, int64) col_indices; see the single-GPU
    // path for the rationale. Reuse host buffers across calls to avoid
    // reallocating these large arrays on every solve.
    static thread_local std::vector<int> h_row_ptrs;
    static thread_local std::vector<int64_t> h_col_indices;
    if (static_cast<int>(h_row_ptrs.size()) < n_local + 1)
      h_row_ptrs.resize(n_local + 1);
    if (static_cast<int>(h_col_indices.size()) < nnz)
      h_col_indices.resize(nnz);
    cudaMemcpyAsync(h_row_ptrs.data(), row_ptrs_data,
                    (n_local + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_col_indices.data(), col_indices_data,
                    nnz * sizeof(int64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    size_t structure_hash = fnv1a_hash(h_row_ptrs.data(), (n_local + 1) * sizeof(int));
    structure_hash = fnv1a_hash(h_col_indices.data(), nnz * sizeof(int64_t), structure_hash);

    MPICacheKey key = {
        n_local,
        nglobal_host,
        nnz,
        lrank_host,
        static_cast<int>(Mode),
        false,
        comm_ptr_val,
        structure_hash,
        std::string(config)};
    bool cache_hit = GetMPISolverCache().get(key, res);

    // Destroys freshly created resources on any early return below (armed only
    // on the cache-miss path; disarmed once the cache takes ownership).
    FreshResourceGuard fresh_guard{&res};

    if (cache_hit)
    {
      // Refresh the cached matrix values. The D2D copy is asynchronous;
      // synchronize before AMGX reads values_buf.
      cudaMemcpyAsync(res.values_buf, values_data, nnz * sizeof(T), cudaMemcpyDeviceToDevice, stream);
      cudaStreamSynchronize(stream);
      AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(
          res.A, n_local, nnz, res.values_buf, nullptr));

      // reuse_setup keeps the existing hierarchy while still refreshing the fine matrix.
      if (!reuse_setup)
        AMGX_SAFE_CALL(AMGX_solver_resetup(res.solver, res.A));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_local, 1, b_data));
      std::vector<T> h_x(n_local, static_cast<T>(0));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.x_vec, n_local, 1, h_x.data()));
    }

    StatsCaptureGuard capture_guard(return_stats != 0);

    if (!cache_hit)
    {
      // On cache miss, evict first (if full) so we never create a new MPI
      // resources handle while stale distributed resources are still alive.
      // This avoids communicator setup crashes under small cache capacities.
      GetMPISolverCache().evict_lru_if_needed(1, DestroyResources);

      fresh_guard.armed = true;

      AMGX_SAFE_CALL(CreateAmgxConfigFromStringOrFile(config, &res.cfg));

      if (IsIsolatedMode())
      {
        res.owns_resources = true;
        AMGX_SAFE_CALL(AMGX_resources_create(&res.rsrc, res.cfg, mpi_comm, 1, &lrank_host));
      }
      else
      {
        res.owns_resources = false;
        res.rsrc = GlobalMPIResources::Get().GetHandle(res.cfg, mpi_comm, 1, &lrank_host);
      }
      AMGX_SAFE_CALL(AMGX_matrix_create(&res.A, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_solver_create(&res.solver, res.rsrc, Mode, res.cfg));

      // Copy the matrix into cache-owned buffers. The cached distributed matrix
      // and the warm-path AMGX_matrix_replace_coefficients reference these across
      // solves, whereas the XLA inputs are released after this call. col_indices
      // is narrowed to int32 (the low 4 bytes of each int64; valid since global
      // indices are < nglobal_host, an int) for the 32-bit upload path below.
      if (cudaMalloc(&res.values_buf, nnz * sizeof(T)) != cudaSuccess ||
          cudaMalloc(&res.row_ptrs_buf, (n_local + 1) * sizeof(int)) != cudaSuccess ||
          cudaMalloc(&res.col_indices_buf, nnz * sizeof(int)) != cudaSuccess)
      {
        return ffi::Error::Internal("cudaMalloc for MPI matrix buffers failed");
      }
      cudaMemcpyAsync(res.values_buf, values_data, nnz * sizeof(T), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(res.row_ptrs_buf, row_ptrs_data, (n_local + 1) * sizeof(int), cudaMemcpyDeviceToDevice, stream);
      cudaMemcpy2DAsync(res.col_indices_buf, sizeof(int), col_indices_data, sizeof(int64_t),
                        sizeof(int), nnz, cudaMemcpyDeviceToDevice, stream);
      cudaStreamSynchronize(stream);

      // Contiguous row partition (owning rank per global row), required for AMGX
      // to build the multi-ring halo (classical AMG uses 2 rings).
      int nranks_host = 1;
      MPI_Comm_size(*mpi_comm, &nranks_host);
      std::vector<int> counts(nranks_host);
      int n_local_send = n_local;
      if (MPI_Allgather(&n_local_send, 1, MPI_INT, counts.data(), 1, MPI_INT,
                        *mpi_comm) != MPI_SUCCESS)
        return ffi::Error::Internal("MPI_Allgather of local sizes failed");
      std::vector<int> partition_vector(nglobal_host);
      for (int r = 0, off = 0; r < nranks_host; ++r)
        for (int i = 0; i < counts[r] && off < nglobal_host; ++i)
          partition_vector[off++] = r;

      // 32-bit index upload path; the 64-bit AMGX_matrix_upload_all_global
      // produces a diverging AMG hierarchy at >= 4 ranks.
      int nrings = 1;
      AMGX_SAFE_CALL(AMGX_config_get_default_number_of_rings(res.cfg, &nrings));
      AMGX_SAFE_CALL(AMGX_matrix_upload_all_global_32(
          res.A, nglobal_host, n_local, nnz, 1, 1,
          static_cast<int *>(res.row_ptrs_buf),
          static_cast<int *>(res.col_indices_buf),
          static_cast<T *>(res.values_buf), nullptr,
          nrings, nrings, partition_vector.data()));

      AMGX_SAFE_CALL(AMGX_vector_create(&res.x_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.b_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_bind(res.x_vec, res.A));
      AMGX_SAFE_CALL(AMGX_vector_bind(res.b_vec, res.A));

      std::vector<T> h_x(n_local, static_cast<T>(0));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.x_vec, n_local, 1, h_x.data()));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_local, 1, b_data));

      AMGX_SAFE_CALL(AMGX_solver_setup(res.solver, res.A));
    }

    AMGX_SAFE_CALL(AMGX_solver_solve(res.solver, res.b_vec, res.x_vec));
    AMGX_SOLVE_STATUS status;
    AMGX_SAFE_CALL(AMGX_solver_get_status(res.solver, &status));

    if (status == AMGX_SOLVE_FAILED)
    {
      // fresh_guard destroys not-yet-cached resources; cached entries stay.
      return ffi::Error::Internal("AmgX MPI solve failed");
    }

    int iters = 0;
    double residual = 0.0;
    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(res.solver, &iters));

    AMGX_RC res_rc = AMGX_solver_get_iteration_residual(res.solver, iters, 0, &residual);
    if (res_rc != AMGX_RC_OK)
    {
      residual = -1.0;
    }

    std::vector<T> stats_host = CollectSolveStats<T>(
        res.solver, stats->element_count(), iters, residual, status);
    cudaMemcpyAsync(stats_data, stats_host.data(), stats_host.size() * sizeof(T),
                    cudaMemcpyHostToDevice, stream);
    AMGX_SAFE_CALL(AMGX_vector_download(res.x_vec, x_data));

    if (!cache_hit)
    {
      fresh_guard.armed = false;
      GetMPISolverCache().put(key, res, DestroyResources);
    }

    // Required for JIT forward+backward reuse of cached handles.
    cudaDeviceSynchronize();

    return ffi::Error::Success();
  }
#endif // JAXAMG_WITH_MPI

  // Float implementation (single-GPU)
  inline ffi::Error AmgxSolveImpl(cudaStream_t stream,
                                  ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                  ffi::Buffer<ffi::DataType::S32> col_indices,
                                  ffi::Buffer<ffi::DataType::F32> values,
                                  ffi::Buffer<ffi::DataType::F32> b,
                                  ffi::ResultBuffer<ffi::DataType::F32> x,
                                  ffi::ResultBuffer<ffi::DataType::F32> stats,
                                  std::string_view config,
                                  int32_t transpose_solve,
                                  int32_t return_stats,
                                  int32_t reuse_setup)
  {
    return AmgxSolveInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(
        stream, row_ptrs, col_indices, values, b, x, stats, config, transpose_solve, return_stats, reuse_setup);
  }

  // Double implementation
  inline ffi::Error AmgxSolveImplDouble(cudaStream_t stream,
                                        ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                        ffi::Buffer<ffi::DataType::S32> col_indices,
                                        ffi::Buffer<ffi::DataType::F64> values,
                                        ffi::Buffer<ffi::DataType::F64> b,
                                        ffi::ResultBuffer<ffi::DataType::F64> x,
                                        ffi::ResultBuffer<ffi::DataType::F64> stats,
                                        std::string_view config,
                                        int32_t transpose_solve,
                                        int32_t return_stats,
                                        int32_t reuse_setup)
  {
    return AmgxSolveInternal<double, ffi::DataType::F64, AMGX_mode_dDDI>(
        stream, row_ptrs, col_indices, values, b, x, stats, config, transpose_solve, return_stats, reuse_setup);
  }

#ifdef JAXAMG_WITH_MPI
  // MPI Float implementation
  inline ffi::Error AmgxSolveMPIImpl(cudaStream_t stream,
                                     ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                     ffi::Buffer<ffi::DataType::S64> col_indices,
                                     ffi::Buffer<ffi::DataType::F32> values,
                                     ffi::Buffer<ffi::DataType::F32> b,
                                     ffi::Buffer<ffi::DataType::S32> nglobal,
                                     ffi::Buffer<ffi::DataType::S32> comm_ptr,
                                     ffi::Buffer<ffi::DataType::S32> lrank,
                                     ffi::ResultBuffer<ffi::DataType::F32> x,
                                     ffi::ResultBuffer<ffi::DataType::F32> stats,
                                     std::string_view config,
                                     int32_t transpose_solve,
                                     int32_t return_stats,
                                     int32_t reuse_setup)
  {
    return AmgxSolveMPIInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(
        stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config, transpose_solve, return_stats, reuse_setup);
  }

  // MPI Double implementation
  inline ffi::Error AmgxSolveMPIImplDouble(cudaStream_t stream,
                                           ffi::Buffer<ffi::DataType::S32> row_ptrs,
                                           ffi::Buffer<ffi::DataType::S64> col_indices,
                                           ffi::Buffer<ffi::DataType::F64> values,
                                           ffi::Buffer<ffi::DataType::F64> b,
                                           ffi::Buffer<ffi::DataType::S32> nglobal,
                                           ffi::Buffer<ffi::DataType::S32> comm_ptr,
                                           ffi::Buffer<ffi::DataType::S32> lrank,
                                           ffi::ResultBuffer<ffi::DataType::F64> x,
                                           ffi::ResultBuffer<ffi::DataType::F64> stats,
                                           std::string_view config,
                                           int32_t transpose_solve,
                                           int32_t return_stats,
                                           int32_t reuse_setup)
  {
    return AmgxSolveMPIInternal<double, ffi::DataType::F64, AMGX_mode_dDDI>(
        stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config, transpose_solve, return_stats, reuse_setup);
  }

#endif // JAXAMG_WITH_MPI

} // namespace

#endif // JAXAMG_AMGX_SOLVERS_H_
