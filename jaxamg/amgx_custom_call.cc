#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <amgx_c.h>
#include <xla/ffi/api/ffi.h>
#include <cstdint>
#include <mutex>

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace
{

  std::once_flag g_amgx_init_flag;

  void AmgxFinalize()
  {
    AMGX_SAFE_CALL(AMGX_finalize());
  }

  void EnsureAmgxInitialized()
  {
    std::call_once(g_amgx_init_flag, []()
                   {
    AMGX_SAFE_CALL(AMGX_initialize());
    AMGX_SAFE_CALL(AMGX_install_signal_handler());
    std::atexit(AmgxFinalize); });
  }

  // Typed FFI handler using XLA FFI API
  ffi::Error AmgxSolveImpl(cudaStream_t stream,
                           ffi::Buffer<ffi::S32> row_ptrs,
                           ffi::Buffer<ffi::S32> col_indices,
                           ffi::Buffer<ffi::F32> values,
                           ffi::Buffer<ffi::F32> b,
                           ffi::ResultBuffer<ffi::F32> x)
  {
    EnsureAmgxInitialized();

    // Set the device for this stream
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess)
    {
      return ffi::Error::Internal("cudaGetDevice failed");
    }
    // Extract dimensions using dimensions() method
    const int n = static_cast<int>(b.dimensions().size() > 0 ? b.dimensions()[0] : 0);
    const int nnz = static_cast<int>(values.dimensions().size() > 0 ? values.dimensions()[0] : 0);
    const int n_rows_plus_1 = static_cast<int>(row_ptrs.dimensions().size() > 0 ? row_ptrs.dimensions()[0] : 0);

    // Get pointers to buffer data
    int *row_ptrs_data = const_cast<int *>(row_ptrs.typed_data());
    int *col_indices_data = const_cast<int *>(col_indices.typed_data());
    float *values_data = const_cast<float *>(values.typed_data());
    float *b_data = const_cast<float *>(b.typed_data());
    float *x_data = x->typed_data();

    // Synchronize stream before calling AmgX
    cudaStreamSynchronize(stream);

    err = cudaSetDevice(device);
    if (err != cudaSuccess)
    {
      return ffi::Error::Internal("cudaSetDevice failed");
    }

    // Initialize AmgX objects
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle x_vec;
    AMGX_vector_handle b_vec;
    AMGX_solver_handle solver;

    // Use host mode (hFFI) to avoid CUDA context issues
    const AMGX_Mode mode = AMGX_mode_hFFI;
    const char *cfg_str =
        "config_version=2, solver=CG, preconditioner=AMG, max_iters=100, "
        "tolerance=1e-6, norm=L2, print_solve_stats=1, monitor_residual=1, "
        "cycle=V, smoother=JACOBI_L1";

    AMGX_SAFE_CALL(AMGX_config_create(&cfg, cfg_str));
    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

    // Download device data to host for hFFI mode
    int *row_ptrs_host = new int[n + 1];
    int *col_indices_host = new int[nnz];
    float *values_host = new float[nnz];
    float *b_host = new float[n];
    float *x_host = new float[n];

    cudaMemcpyAsync(row_ptrs_host, row_ptrs_data, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(col_indices_host, col_indices_data, nnz * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(values_host, values_data, nnz * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(b_host, b_data, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&x_vec, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&b_vec, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));

    // Upload matrix and vectors
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, n, nnz, 1, 1,
                                          row_ptrs_host, col_indices_host,
                                          values_host, nullptr));
    AMGX_SAFE_CALL(AMGX_vector_upload(b_vec, n, 1, b_host));
    AMGX_SAFE_CALL(AMGX_vector_set_zero(x_vec, n, 1));

    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
    AMGX_SAFE_CALL(AMGX_solver_solve(solver, b_vec, x_vec));

    // Download result and upload to device
    AMGX_SAFE_CALL(AMGX_vector_download(x_vec, x_host));
    cudaMemcpyAsync(x_data, x_host, static_cast<size_t>(n) * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);

    // Cleanup
    AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
    AMGX_SAFE_CALL(AMGX_vector_destroy(x_vec));
    AMGX_SAFE_CALL(AMGX_vector_destroy(b_vec));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    delete[] row_ptrs_host;
    delete[] col_indices_host;
    delete[] values_host;
    delete[] b_host;
    delete[] x_host;

    return ffi::Error::Success();
  }

  // Define handler symbol with typed FFI binding
  XLA_FFI_DEFINE_HANDLER_SYMBOL(
      AmgxSolve, AmgxSolveImpl,
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>() // CUDA stream context
          .Arg<ffi::Buffer<ffi::S32>>()             // row_ptrs
          .Arg<ffi::Buffer<ffi::S32>>()             // col_indices
          .Arg<ffi::Buffer<ffi::F32>>()             // values
          .Arg<ffi::Buffer<ffi::F32>>()             // b
          .Ret<ffi::Buffer<ffi::F32>>()             // x
  );

} // namespace

PYBIND11_MODULE(_amgx_ext, m)
{
  m.def("get_amgx_solve_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolve)); });
}
