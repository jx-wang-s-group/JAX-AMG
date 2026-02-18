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
#include <mpi.h>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>
#include <type_traits>

#include "_amgx_utils.h"

namespace ffi = xla::ffi;

namespace
{

  template <typename T>
  inline const char *CsrTransposeDevice(cudaStream_t stream,
                                        int n_rows,
                                        int nnz,
                                        const int *row_ptrs,
                                        const int *col_indices,
                                        const T *values,
                                        int *row_ptrs_t,
                                        int *col_indices_t,
                                        T *values_t)
  {
    cusparseHandle_t handle = nullptr;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS)
    {
      return "cusparseCreate failed";
    }
    if (cusparseSetStream(handle, stream) != CUSPARSE_STATUS_SUCCESS)
    {
      cusparseDestroy(handle);
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
      cusparseDestroy(handle);
      return "cusparseCsr2cscEx2_bufferSize failed";
    }

    void *workspace = nullptr;
    if (buffer_size > 0 && cudaMalloc(&workspace, buffer_size) != cudaSuccess)
    {
      cusparseDestroy(handle);
      return "cudaMalloc failed for cusparse transpose workspace";
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
        workspace);

    cudaFree(workspace);
    cusparseDestroy(handle);

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
                               int32_t transpose_solve)
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

    // Cache the last key to skip the structure hash D2H on repeated calls.
    static CacheKey last_key = {};
    static bool last_key_valid = false;

    size_t structure_hash = 0;
    bool fast_path = false;

    if (last_key_valid &&
        last_key.n_rows == n_rows &&
        last_key.nnz == nnz &&
        last_key.mode == static_cast<int>(Mode) &&
        last_key.transpose_solve == (transpose_solve != 0) &&
        last_key.config == std::string(config))
    {
      structure_hash = last_key.structure_hash;
      fast_path = true;
    }
    else
    {
      std::vector<int> h_row_ptrs(n_rows + 1);
      cudaMemcpy(h_row_ptrs.data(), row_ptrs_data, (n_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost);
      structure_hash = fnv1a_hash(h_row_ptrs.data(), (n_rows + 1) * sizeof(int));
    }

    CacheKey key = {n_rows, nnz, static_cast<int>(Mode), transpose_solve != 0, structure_hash, std::string(config)};
    bool cache_hit = GetSolverCache().get(key, res);

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
            static_cast<T *>(res.transpose_values));
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

      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_rows, 1, b_data));
      AMGX_SAFE_CALL(AMGX_vector_set_zero(res.x_vec, n_rows, 1));

      reuse_success = true;
    }

    last_key = key;
    last_key_valid = true;

    if (!cache_hit)
    {

      AMGX_SAFE_CALL(CreateAmgxConfigFromStringOrFile(config, &res.cfg));

      if (IsIsolatedMode()) {
         res.owns_resources = true;
         AMGX_SAFE_CALL(AMGX_resources_create_simple(&res.rsrc, res.cfg));
      } else {
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
          DestroyResources(res);
          return ffi::Error::Internal("cudaMalloc failed for transpose row_ptrs");
        }
        if (cudaMalloc(&res.transpose_col_indices, nnz * sizeof(int)) != cudaSuccess)
        {
          cudaFree(res.transpose_row_ptrs);
          res.transpose_row_ptrs = nullptr;
          DestroyResources(res);
          return ffi::Error::Internal("cudaMalloc failed for transpose col_indices");
        }
        if (cudaMalloc(&res.transpose_values, nnz * sizeof(T)) != cudaSuccess)
        {
          cudaFree(res.transpose_row_ptrs);
          cudaFree(res.transpose_col_indices);
          res.transpose_row_ptrs = nullptr;
          res.transpose_col_indices = nullptr;
          DestroyResources(res);
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
            static_cast<T *>(res.transpose_values));
        if (transpose_err != nullptr)
        {
          DestroyResources(res);
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
      return ffi::Error::Internal("AmgX solve failed");
    }

    int iters = 0;
    double residual = 0.0;

    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(res.solver, &iters));

    AMGX_RC res_rc = AMGX_solver_get_iteration_residual(res.solver, iters, 0, &residual);
    if (res_rc != AMGX_RC_OK) {
      residual = -1.0;
    }

    T stats_host[3] = {
        static_cast<T>(iters),
        static_cast<T>(residual),
        static_cast<T>(status)};
    cudaMemcpyAsync(stats_data, stats_host, 3 * sizeof(T), cudaMemcpyHostToDevice, stream);

    AMGX_SAFE_CALL(AMGX_vector_download(res.x_vec, x_data));

    // 7. Store in Cache (if new)
    if (!cache_hit)
    {
       GetSolverCache().put(key, res, DestroyResources);
    }

    return ffi::Error::Success();
  }


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
                                  int32_t transpose_solve)
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

    static MPICacheKey mpi_last_key = {};
    static bool mpi_last_key_valid = false;

    size_t structure_hash = 0;

    if (mpi_last_key_valid &&
        mpi_last_key.n_local == n_local &&
        mpi_last_key.n_global == nglobal_host &&
        mpi_last_key.nnz == nnz &&
        mpi_last_key.lrank == lrank_host &&
        mpi_last_key.mode == static_cast<int>(Mode) &&
        mpi_last_key.comm_ptr == comm_ptr_val &&
        mpi_last_key.config == std::string(config))
    {
      structure_hash = mpi_last_key.structure_hash;
    }
    else
    {
      std::vector<int> h_row_ptrs(n_local + 1);
      cudaMemcpy(h_row_ptrs.data(), row_ptrs_data, (n_local + 1) * sizeof(int), cudaMemcpyDeviceToHost);
      structure_hash = fnv1a_hash(h_row_ptrs.data(), (n_local + 1) * sizeof(int));
    }

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

    if (cache_hit)
    {
      cudaMemcpy(res.values_buf, values_data, nnz * sizeof(T), cudaMemcpyDeviceToDevice);
      AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(
          res.A, n_local, nnz, res.values_buf, nullptr));

      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_local, 1, b_data));
      std::vector<T> h_x(n_local, static_cast<T>(0));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.x_vec, n_local, 1, h_x.data()));
    }

    mpi_last_key = key;
    mpi_last_key_valid = true;

    if (!cache_hit)
    {
      // On cache miss, evict first (if full) so we never create a new MPI
      // resources handle while stale distributed resources are still alive.
      // This avoids communicator setup crashes under small cache capacities.
      GetMPISolverCache().evict_lru_if_needed(1, DestroyResources);

      AMGX_SAFE_CALL(CreateAmgxConfigFromStringOrFile(config, &res.cfg));

      if (IsIsolatedMode()) {
        res.owns_resources = true;
        AMGX_SAFE_CALL(AMGX_resources_create(&res.rsrc, res.cfg, mpi_comm, 1, &lrank_host));
      } else {
        res.owns_resources = false;
        res.rsrc = GlobalMPIResources::Get().GetHandle(res.cfg, mpi_comm, 1, &lrank_host);
      }
      AMGX_SAFE_CALL(AMGX_matrix_create(&res.A, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_solver_create(&res.solver, res.rsrc, Mode, res.cfg));

      if (cudaMalloc(&res.values_buf, nnz * sizeof(T)) != cudaSuccess)
      {
        return ffi::Error::Internal("cudaMalloc for values buffer failed");
      }
      cudaMemcpy(res.values_buf, values_data, nnz * sizeof(T), cudaMemcpyDeviceToDevice);

      int nrings = 1;
      AMGX_SAFE_CALL(AMGX_config_get_default_number_of_rings(res.cfg, &nrings));

      AMGX_SAFE_CALL(AMGX_matrix_upload_all_global(
          res.A, nglobal_host, n_local, nnz, 1, 1,
          row_ptrs_data, col_indices_data, static_cast<T *>(res.values_buf), nullptr,
          nrings, nrings, nullptr));

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
      if (!cache_hit)
      {
        DestroyResources(res);
      }
      return ffi::Error::Internal("AmgX MPI solve failed");
    }

    int iters = 0;
    double residual = 0.0;
    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(res.solver, &iters));

    AMGX_RC res_rc = AMGX_solver_get_iteration_residual(res.solver, iters, 0, &residual);
    if (res_rc != AMGX_RC_OK) {
      residual = -1.0;
    }

    T stats_host[3] = {static_cast<T>(iters), static_cast<T>(residual), static_cast<T>(status)};
    cudaMemcpyAsync(stats_data, stats_host, 3 * sizeof(T), cudaMemcpyHostToDevice, stream);
    AMGX_SAFE_CALL(AMGX_vector_download(res.x_vec, x_data));

    if (!cache_hit)
    {
      GetMPISolverCache().put(key, res, DestroyResources);
    }

    // Required for JIT forward+backward reuse of cached handles.
    cudaDeviceSynchronize();

    return ffi::Error::Success();
  }

  // Float implementation (single-GPU)
  inline ffi::Error AmgxSolveImpl(cudaStream_t stream,
                           ffi::Buffer<ffi::DataType::S32> row_ptrs,
                           ffi::Buffer<ffi::DataType::S32> col_indices,
                           ffi::Buffer<ffi::DataType::F32> values,
                           ffi::Buffer<ffi::DataType::F32> b,
                           ffi::ResultBuffer<ffi::DataType::F32> x,
                           ffi::ResultBuffer<ffi::DataType::F32> stats,
                           std::string_view config,
                           int32_t transpose_solve)
  {
    return AmgxSolveInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(
        stream, row_ptrs, col_indices, values, b, x, stats, config, transpose_solve);
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
                                 int32_t transpose_solve)
  {
    return AmgxSolveInternal<double, ffi::DataType::F64, AMGX_mode_dDDI>(
        stream, row_ptrs, col_indices, values, b, x, stats, config, transpose_solve);
  }

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
                              int32_t transpose_solve)
  {
    return AmgxSolveMPIInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(
        stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config, transpose_solve);
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
                                    int32_t transpose_solve)
  {
    return AmgxSolveMPIInternal<double, ffi::DataType::F64, AMGX_mode_dDDI>(
        stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config, transpose_solve);
  }

  // -------------------------------------------------------------------------
  // MPI AllGather Custom Call
  // -------------------------------------------------------------------------

  template <typename T, ffi::DataType DType>
  inline ffi::Error AmgxAllGatherInternal(cudaStream_t stream,
                                   ffi::Buffer<DType> sendbuf,
                                   ffi::Buffer<ffi::DataType::S32> recvcounts,
                                   ffi::Buffer<ffi::DataType::S32> displs,
                                   ffi::Buffer<ffi::DataType::S32> comm_ptr_buf,
                                   ffi::ResultBuffer<DType> recvbuf)
  {
    cudaStreamSynchronize(stream);

    int comm_ptr_parts[2];
    cudaMemcpy(comm_ptr_parts, comm_ptr_buf.typed_data(), 2 * sizeof(int), cudaMemcpyDeviceToHost);
    uint64_t comm_ptr_val = (static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[1])) << 32) |
                            static_cast<uint64_t>(static_cast<uint32_t>(comm_ptr_parts[0]));
    MPI_Comm *mpi_comm = reinterpret_cast<MPI_Comm *>(comm_ptr_val);
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

  inline ffi::Error AmgxAllGatherImpl(cudaStream_t stream,
                               ffi::Buffer<ffi::DataType::F32> sendbuf,
                               ffi::Buffer<ffi::DataType::S32> recvcounts,
                               ffi::Buffer<ffi::DataType::S32> displs,
                               ffi::Buffer<ffi::DataType::S32> comm_ptr,
                               ffi::ResultBuffer<ffi::DataType::F32> recvbuf)
  {
    return AmgxAllGatherInternal<float, ffi::DataType::F32>(stream, sendbuf, recvcounts, displs, comm_ptr, recvbuf);
  }

  inline ffi::Error AmgxAllGatherImplDouble(cudaStream_t stream,
                                     ffi::Buffer<ffi::DataType::F64> sendbuf,
                                     ffi::Buffer<ffi::DataType::S32> recvcounts,
                                     ffi::Buffer<ffi::DataType::S32> displs,
                                     ffi::Buffer<ffi::DataType::S32> comm_ptr,
                                     ffi::ResultBuffer<ffi::DataType::F64> recvbuf)
  {
    return AmgxAllGatherInternal<double, ffi::DataType::F64>(stream, sendbuf, recvcounts, displs, comm_ptr, recvbuf);
  }

} // namespace

#endif // JAXAMG_AMGX_SOLVERS_H_
