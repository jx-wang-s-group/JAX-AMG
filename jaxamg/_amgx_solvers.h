/*
 * Internal header for AmgX solver implementations.
 * Included by _amgx.cc.
 */

#ifndef JAXAMG_AMGX_SOLVERS_H_
#define JAXAMG_AMGX_SOLVERS_H_

#include <cuda_runtime.h>
#include <amgx_c.h>
#include <xla/ffi/api/ffi.h>
#include <mpi.h>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include "_amgx_utils.h"

namespace ffi = xla::ffi;

namespace
{

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
    CacheKey key = {row_ptrs_data, col_indices_data, n_rows, nnz, static_cast<int>(Mode), std::string(config)};
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

      AMGX_SAFE_CALL(CreateAmgxConfigFromStringOrFile(config, &res.cfg));

      if (IsIsolatedMode()) {
         res.owns_resources = true;
         AMGX_SAFE_CALL(AMGX_resources_create_simple(&res.rsrc, res.cfg));
      } else {
         res.owns_resources = false;
         res.rsrc = GlobalResources::Get().GetHandle(res.cfg);
      }

      // Create Matrix and Vectors
      AMGX_SAFE_CALL(AMGX_matrix_create(&res.A, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.x_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.b_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_solver_create(&res.solver, res.rsrc, Mode, res.cfg));

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

    CachedResources res;

    // Hash row_ptrs content to fingerprint sparsity structure (small D2H copy).
    std::vector<int> h_row_ptrs(n_local + 1);
    cudaMemcpy(h_row_ptrs.data(), row_ptrs_data, (n_local + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    size_t structure_hash = fnv1a_hash(h_row_ptrs.data(), (n_local + 1) * sizeof(int));

    MPICacheKey key = {
        n_local,
        nglobal_host,
        nnz,
        lrank_host,
        static_cast<int>(Mode),
        comm_ptr_val,
        structure_hash,
        std::string(config)};
    bool cache_hit = GetMPISolverCache().get(key, res);

    if (cache_hit)
    {
      // Same structure (guaranteed by structure_hash). Replace values only.
      cudaMemcpy(res.values_buf, values_data, nnz * sizeof(T), cudaMemcpyDeviceToDevice);
      AMGX_SAFE_CALL(AMGX_matrix_replace_coefficients(
          res.A, n_local, nnz, res.values_buf, nullptr));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_local, 1, b_data));
      std::vector<T> h_x(n_local, static_cast<T>(0));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.x_vec, n_local, 1, h_x.data()));
    }
    else
    {
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

      // Persistent buffer for values, registered with AMGX's DistributedManager.
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

      // Create and initialize vectors (AMGX manages halo space)
      AMGX_SAFE_CALL(AMGX_vector_create(&res.x_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_create(&res.b_vec, res.rsrc, Mode));
      AMGX_SAFE_CALL(AMGX_vector_bind(res.x_vec, res.A));
      AMGX_SAFE_CALL(AMGX_vector_bind(res.b_vec, res.A));

      std::vector<T> h_x(n_local, static_cast<T>(0));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.x_vec, n_local, 1, h_x.data()));
      AMGX_SAFE_CALL(AMGX_vector_upload(res.b_vec, n_local, 1, b_data));
    }

    // Solve
    AMGX_SAFE_CALL(AMGX_solver_setup(res.solver, res.A));
    AMGX_SAFE_CALL(AMGX_solver_solve(res.solver, res.b_vec, res.x_vec));

    // Retrieve results
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
    AMGX_SAFE_CALL(AMGX_solver_get_iteration_residual(res.solver, iters, 0, &residual));

    T stats_host[3] = {static_cast<T>(iters), static_cast<T>(residual), static_cast<T>(status)};
    cudaMemcpyAsync(stats_data, stats_host, 3 * sizeof(T), cudaMemcpyHostToDevice, stream);
    AMGX_SAFE_CALL(AMGX_vector_download(res.x_vec, x_data));

    if (!cache_hit)
    {
      GetMPISolverCache().put(key, res, DestroyResources);
    }

    // Sync AMGX's internal streams before returning to XLA (required for
    // JIT forward+backward reuse of cached handles).
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
                           std::string_view config)
  {
    return AmgxSolveInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(stream, row_ptrs, col_indices, values, b, x, stats, config);
  }

  // Double implementation
  inline ffi::Error AmgxSolveImplDouble(cudaStream_t stream,
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
                              std::string_view config)
  {
    return AmgxSolveMPIInternal<float, ffi::DataType::F32, AMGX_mode_dFFI>(stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config);
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
                                    std::string_view config)
  {
    return AmgxSolveMPIInternal<double, ffi::DataType::F64, AMGX_mode_dDDI>(stream, row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, x, stats, config);
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
