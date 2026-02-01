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
#include <mutex>
#include <string>
#include <fstream>

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace
{

// Undefine existing macro from amgx_c.h to allow custom error handling
#ifdef AMGX_SAFE_CALL
  #undef AMGX_SAFE_CALL
#endif

// Macro for functions returning ffi::Error (propagates to Python)
#define AMGX_SAFE_CALL(call)                                                \
  do                                                                        \
  {                                                                         \
    AMGX_RC err = (call);                                                   \
    if (err != AMGX_RC_OK)                                                  \
    {                                                                       \
      char msg[4096];                                                       \
      AMGX_get_error_string(err, msg, 4096);                                \
      std::string error_msg = "AMGX Error: " + std::string(msg);            \
      return ffi::Error::Internal(error_msg);                               \
    }                                                                       \
  } while (0)

// Macro for functions returning void (just log error)
#define AMGX_SAFE_CALL_VOID(call)                                           \
  do                                                                        \
  {                                                                         \
    AMGX_RC err = (call);                                                   \
    if (err != AMGX_RC_OK)                                                  \
    {                                                                       \
      char msg[4096];                                                       \
      AMGX_get_error_string(err, msg, 4096);                                \
      fprintf(stderr, "AMGX Error in void function: %s\n", msg);            \
    }                                                                       \
  } while (0)

  std::once_flag g_amgx_init_flag;

  void AmgxFinalize()
  {
    AMGX_SAFE_CALL_VOID(AMGX_finalize());
  }

  // Custom callback to suppress AmgX library output (banners, version info)
  void PrintCallback(const char *msg, int length)
  {
      // No-op: Output is fully suppressed to keep stdout clean for the user.
      return;
  }

  void EnsureAmgxInitialized()
  {
    std::call_once(g_amgx_init_flag, []() {
        // Register print callback before initialization
        AMGX_register_print_callback(PrintCallback);

        AMGX_SAFE_CALL_VOID(AMGX_initialize());
        AMGX_SAFE_CALL_VOID(AMGX_install_signal_handler());
        std::atexit(AmgxFinalize);
    });
  }

  /*
   * AmgxSolveImpl: Core implementation of the XLA FFI handler.
   *
   * Arguments:
   *   stream: CUDA stream provided by XLA/JAX.
   *   row_ptrs, col_indices, values: CSR matrix components (on device).
   *   b: RHS vector (on device).
   *   x: Output solution buffer (on device).
   *   stats: Output statistics buffer [iterations, residual, status].
   *   config: AmgX configuration string.
   */
  ffi::Error AmgxSolveImpl(cudaStream_t stream,
                           ffi::Buffer<ffi::S32> row_ptrs,
                           ffi::Buffer<ffi::S32> col_indices,
                           ffi::Buffer<ffi::F32> values,
                           ffi::Buffer<ffi::F32> b,
                           ffi::ResultBuffer<ffi::F32> x,
                           ffi::ResultBuffer<ffi::F32> stats,
                           std::string_view config)
  {
    EnsureAmgxInitialized();

    // 1. Setup execution context
    int device;
    if (cudaGetDevice(&device) != cudaSuccess) {
      return ffi::Error::Internal("cudaGetDevice failed");
    }

    if (cudaSetDevice(device) != cudaSuccess) {
      return ffi::Error::Internal("cudaSetDevice failed");
    }

    // Synchronize stream to ensure inputs are ready
    cudaStreamSynchronize(stream);

    // 2. Prepare data pointers (Device Mode / dFFI)
    // We cast to raw pointers to pass directly to AmgX, avoiding host transfers.
    int *row_ptrs_data = const_cast<int *>(row_ptrs.typed_data());
    int *col_indices_data = const_cast<int *>(col_indices.typed_data());
    float *values_data = const_cast<float *>(values.typed_data());
    float *b_data = const_cast<float *>(b.typed_data());
    float *x_data = x->typed_data();
    float *stats_data = stats->typed_data();

    // Dimensions
    const int n_rows = static_cast<int>(b.dimensions().size() > 0 ? b.dimensions()[0] : 0);

    // 3. Initialize AmgX Resources
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle x_vec;
    AMGX_vector_handle b_vec;
    AMGX_solver_handle solver;

    // Use AMGX_mode_dFFI to indicate data is already on the device
    const AMGX_Mode mode = AMGX_mode_dFFI;

    // Prepare configuration
    // Prepare configuration
    std::string config_str(config);

    // Attempt to open as file to determine if it's a file path
    std::ifstream file_check(config_str);
    bool is_file = file_check.good();
    file_check.close();  // Close the stream before AmgX tries to read

    if (is_file) {
        AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, config_str.c_str()));
    } else {
        AMGX_SAFE_CALL(AMGX_config_create(&cfg, config_str.c_str()));
    }

    AMGX_SAFE_CALL(AMGX_resources_create_simple(&rsrc, cfg));

    // Create Matrix and Vectors
    AMGX_SAFE_CALL(AMGX_matrix_create(&A, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&x_vec, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_vector_create(&b_vec, rsrc, mode));
    AMGX_SAFE_CALL(AMGX_solver_create(&solver, rsrc, mode, cfg));

    // 4. Bind Data (No Copies)
    // Upload matrix data (binds device pointers)
    AMGX_SAFE_CALL(AMGX_matrix_upload_all(A, n_rows, (int)values.element_count(), 1, 1,
                                          row_ptrs_data, col_indices_data, values_data, nullptr));

    // Bind RHS vector
    AMGX_SAFE_CALL(AMGX_vector_upload(b_vec, n_rows, 1, b_data));

    // Initialize solution vector to zero
    AMGX_SAFE_CALL(AMGX_vector_set_zero(x_vec, n_rows, 1));

    // 5. Solve System
    AMGX_SAFE_CALL(AMGX_solver_setup(solver, A));
    AMGX_SAFE_CALL(AMGX_solver_solve(solver, b_vec, x_vec));

    // 6. Retrieve Results & Statistics
    AMGX_SOLVE_STATUS status;
    AMGX_SAFE_CALL(AMGX_solver_get_status(solver, &status));

    if (status == AMGX_SOLVE_FAILED) {
      // Clean up before returning error
      AMGX_solver_destroy(solver);
      AMGX_vector_destroy(b_vec);
      AMGX_vector_destroy(x_vec);
      AMGX_matrix_destroy(A);
      AMGX_resources_destroy(rsrc);
      AMGX_config_destroy(cfg);
      return ffi::Error::Internal("AmgX solve failed");
    }

    int iters = 0;
    double residual = 0.0;

    AMGX_SAFE_CALL(AMGX_solver_get_iterations_number(solver, &iters));

    // Retrieve residual at the final iteration
    AMGX_SAFE_CALL(AMGX_solver_get_iteration_residual(solver, iters, 0, &residual));

    // Transfer stats to output buffer [iters, residual, status]
    float stats_host[3] = {
        static_cast<float>(iters),
        static_cast<float>(residual),
        static_cast<float>(status)
    };
    cudaMemcpyAsync(stats_data, stats_host, 3 * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Download solution (copies from AmgX internal buffer to JAX output buffer)
    AMGX_SAFE_CALL(AMGX_vector_download(x_vec, x_data));

    // 7. Cleanup
    AMGX_SAFE_CALL(AMGX_solver_destroy(solver));
    AMGX_SAFE_CALL(AMGX_vector_destroy(b_vec));
    AMGX_SAFE_CALL(AMGX_vector_destroy(x_vec));
    AMGX_SAFE_CALL(AMGX_matrix_destroy(A));
    AMGX_SAFE_CALL(AMGX_resources_destroy(rsrc));
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));

    return ffi::Error::Success();
  }

  // Register XLA FFI Handler
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

} // namespace

PYBIND11_MODULE(_amgx_ext, m)
{
  m.def("get_amgx_solve_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolve)); });
}
