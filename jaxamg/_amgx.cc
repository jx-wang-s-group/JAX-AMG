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
#ifdef JAXAMG_WITH_MPI
#include <mpi.h>
#endif
#include "_amgx_utils.h"
#include "_amgx_solvers.h"
#include "_local_arrays.h"
#include <pybind11/stl.h>
#ifdef JAXAMG_WITH_MPI
#include "_neighbour_exchange.h"
#endif

namespace py = pybind11;
namespace ffi = xla::ffi;

// Global variables for capturing solver output
std::string g_stats_string = "";
bool g_capture_stats = false;

namespace
{
  inline const char *ModeToString(int mode)
  {
    switch (mode)
    {
    case AMGX_mode_dFFI:
      return "float32";
    case AMGX_mode_dDDI:
      return "float64";
    default:
      return "unknown";
    }
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
          .Arg<ffi::Buffer<ffi::F32>>()             // x0 (ignored unless use_x0)
          .Ret<ffi::Buffer<ffi::F32>>()             // x
          .Ret<ffi::Buffer<ffi::F32>>()             // stats
          .Attr<std::string_view>("config")         // config string
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
          .Attr<int32_t>("reuse_setup")             // skip warm resetup
          .Attr<int32_t>("use_x0")                  // honor x0 initial guess
          .Attr<int32_t>("block_dim")               // BSR block size (1 = scalar CSR)
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
          .Arg<ffi::Buffer<ffi::F64>>()             // x0 (ignored unless use_x0)
          .Ret<ffi::Buffer<ffi::F64>>()             // x
          .Ret<ffi::Buffer<ffi::F64>>()             // stats
          .Attr<std::string_view>("config")         // config string
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
          .Attr<int32_t>("reuse_setup")             // skip warm resetup
          .Attr<int32_t>("use_x0")                  // honor x0 initial guess
          .Attr<int32_t>("block_dim")               // BSR block size (1 = scalar CSR)
  );

#ifdef JAXAMG_WITH_MPI
  // The MPI solve handlers. The ordering token joins every rank's
  // communicating calls into one program order (mpi4jax's chain in per-rank
  // programs; a fresh token in SPMD programs, whose common program orders
  // them). ``sizes`` = (n_local, nnz) marks padded buffers, whose prefixes are
  // solved and whose solution tail is zeroed; (-1, -1) means exact buffers.
  template <ffi::DataType DT>
  using MPISolve = ffi::Error (*)(cudaStream_t, ffi::Buffer<ffi::S32>, ffi::Buffer<ffi::S64>,
                                  ffi::Buffer<DT>, ffi::Buffer<DT>, ffi::Buffer<DT>,
                                  ffi::Buffer<ffi::S32>, ffi::Buffer<ffi::S32>,
                                  ffi::Buffer<ffi::S32>, ffi::ResultBuffer<DT>,
                                  ffi::ResultBuffer<DT>, std::string_view, int32_t, int32_t,
                                  int32_t, int32_t, int32_t, int32_t, int, int);

  template <ffi::DataType DT, MPISolve<DT> Solve>
  ffi::Error AmgxSolveMPIPadded(
      cudaStream_t stream, ffi::Buffer<ffi::S32> row_ptrs, ffi::Buffer<ffi::S64> col_indices,
      ffi::Buffer<DT> values, ffi::Buffer<DT> b, ffi::Buffer<DT> x0,
      ffi::Buffer<ffi::S32> nglobal, ffi::Buffer<ffi::S32> comm_ptr, ffi::Buffer<ffi::S32> lrank,
      ffi::Buffer<ffi::S32> sizes, ffi::Token, ffi::ResultBuffer<DT> x,
      ffi::ResultBuffer<DT> stats, ffi::Result<ffi::Token>, std::string_view config,
      int32_t transpose_solve, int32_t return_stats, int32_t reuse_setup, int32_t use_x0,
      int32_t block_dim, int32_t device_mpi)
  {
    int host_sizes[2] = {-1, -1};
    if (sizes.element_count() != 2)
      return ffi::Error::InvalidArgument("local sizes must contain two integers");
    if (cudaStreamSynchronize(stream) != cudaSuccess)
      return ffi::Error::Internal("waiting for the solve operands failed");
    if (cudaMemcpy(host_sizes, sizes.typed_data(), 2 * sizeof(int), cudaMemcpyDeviceToHost) !=
        cudaSuccess)
      return ffi::Error::Internal("reading the local sizes failed");
    const int64_t n_buffer = static_cast<int64_t>(x->element_count());
    if ((host_sizes[0] < 0 || host_sizes[1] < 0) &&
        (host_sizes[0] != -1 || host_sizes[1] != -1))
      return ffi::Error::InvalidArgument("local sizes must be nonnegative or (-1, -1)");
    const int64_t n = host_sizes[0] < 0 ? n_buffer : host_sizes[0];
    const int64_t nnz = host_sizes[1] < 0 ? values.element_count() : host_sizes[1];
    if (n > n_buffer || n > static_cast<int64_t>(b.element_count()) ||
        n > static_cast<int64_t>(x0.element_count()) ||
        n + 1 > static_cast<int64_t>(row_ptrs.element_count()) ||
        nnz > static_cast<int64_t>(values.element_count()) ||
        nnz > static_cast<int64_t>(col_indices.element_count()))
      return ffi::Error::InvalidArgument("local sizes exceed the padded buffers");
    ffi::Error err = Solve(stream, row_ptrs, col_indices, values, b, x0, nglobal, comm_ptr, lrank,
                           x, stats, config, transpose_solve, return_stats, reuse_setup, use_x0,
                           block_dim, device_mpi, host_sizes[0], host_sizes[1]);
    if (err.success() && n < n_buffer &&
        cudaMemsetAsync(x->typed_data() + n, 0,
                        (n_buffer - n) * sizeof(ffi::NativeType<DT>), stream) != cudaSuccess)
      return ffi::Error::Internal("zeroing the solution padding failed");
    return err;
  }

  XLA_FFI_DEFINE_HANDLER(
      AmgxSolveMPI,
      (AmgxSolveMPIPadded<ffi::F32, AmgxSolveMPIImpl>),
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>() // CUDA stream context
          .Arg<ffi::Buffer<ffi::S32>>()             // row_ptrs
          .Arg<ffi::Buffer<ffi::S64>>()             // col_indices (GLOBAL, int64)
          .Arg<ffi::Buffer<ffi::F32>>()             // values
          .Arg<ffi::Buffer<ffi::F32>>()             // b (local)
          .Arg<ffi::Buffer<ffi::F32>>()             // x0 (local; ignored unless use_x0)
          .Arg<ffi::Buffer<ffi::S32>>()             // nglobal
          .Arg<ffi::Buffer<ffi::S32>>()             // comm_ptr (2 x int32)
          .Arg<ffi::Buffer<ffi::S32>>()             // lrank
          .Arg<ffi::Buffer<ffi::S32>>()             // sizes: (n_local, nnz) or (-1, -1)
          .Arg<ffi::Token>()                        // ordering token
          .Ret<ffi::Buffer<ffi::F32>>()             // x (local)
          .Ret<ffi::Buffer<ffi::F32>>()             // stats
          .Ret<ffi::Token>()                        // ordering token out
          .Attr<std::string_view>("config")         // config string
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
          .Attr<int32_t>("reuse_setup")             // skip warm resetup
          .Attr<int32_t>("use_x0")                  // honor x0 initial guess
          .Attr<int32_t>("block_dim")               // BSR block size (1 = scalar CSR)
          .Attr<int32_t>("device_mpi")              // resource transport identity
  );

  XLA_FFI_DEFINE_HANDLER(
      AmgxSolveMPIDouble,
      (AmgxSolveMPIPadded<ffi::F64, AmgxSolveMPIImplDouble>),
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>() // CUDA stream context
          .Arg<ffi::Buffer<ffi::S32>>()             // row_ptrs
          .Arg<ffi::Buffer<ffi::S64>>()             // col_indices (GLOBAL, int64)
          .Arg<ffi::Buffer<ffi::F64>>()             // values
          .Arg<ffi::Buffer<ffi::F64>>()             // b (local)
          .Arg<ffi::Buffer<ffi::F64>>()             // x0 (local; ignored unless use_x0)
          .Arg<ffi::Buffer<ffi::S32>>()             // nglobal
          .Arg<ffi::Buffer<ffi::S32>>()             // comm_ptr (2 x int32)
          .Arg<ffi::Buffer<ffi::S32>>()             // lrank
          .Arg<ffi::Buffer<ffi::S32>>()             // sizes: (n_local, nnz) or (-1, -1)
          .Arg<ffi::Token>()                        // ordering token
          .Ret<ffi::Buffer<ffi::F64>>()             // x (local)
          .Ret<ffi::Buffer<ffi::F64>>()             // stats
          .Ret<ffi::Token>()                        // ordering token out
          .Attr<std::string_view>("config")         // config string
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
          .Attr<int32_t>("reuse_setup")             // skip warm resetup
          .Attr<int32_t>("use_x0")                  // honor x0 initial guess
          .Attr<int32_t>("block_dim")               // BSR block size (1 = scalar CSR)
          .Attr<int32_t>("device_mpi")              // resource transport identity
  );

#endif // JAXAMG_WITH_MPI

} // namespace

PYBIND11_MODULE(_amgx, m)
{
  m.def("get_amgx_solve_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolve)); });
  m.def("get_amgx_solve_double_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolveDouble)); });
#ifdef JAXAMG_WITH_MPI
  m.def("get_amgx_solve_mpi_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolveMPI)); });
  m.def("get_amgx_solve_mpi_double_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxSolveMPIDouble)); });
  m.def("get_neighbour_exchange_handler", []()
        { return py::capsule(reinterpret_cast<void *>(jaxamg_exchange::NeighbourExchange)); });
  m.def("register_exchange_plan", &jaxamg_exchange::RegisterPlan,
        py::arg("plan_id"), py::arg("comm_handle"), py::arg("send_peers"),
        py::arg("send_counts"), py::arg("recv_peers"), py::arg("recv_counts"));
  m.def("release_exchange_plan", &jaxamg_exchange::ReleasePlan);
  m.def("exchange_plan_count", &jaxamg_exchange::PlanCount);
  m.attr("mpi_enabled") = py::bool_(true);
#else
  m.attr("mpi_enabled") = py::bool_(false);
#endif

  m.def("get_local_array_handler", []()
        { return py::capsule(reinterpret_cast<void *>(jaxamg_local::LoadLocalArray)); });
  m.def("register_local_array", &jaxamg_local::Register, py::arg("array_id"),
        py::arg("data_pointer"), py::arg("nbytes"));
  m.def("release_local_array", &jaxamg_local::Release);
  m.def("local_array_count", &jaxamg_local::Count);
  m.def("synchronize_devices", &jaxamg_local::SynchronizeDevices,
        py::call_guard<py::gil_scoped_release>());

  m.def("initialize", &EnsureAmgxInitialized);
  m.def("finalize", &AmgxFinalize);
  m.def("get_stats_string", []() -> std::string { return g_stats_string; });
  m.def("clear_solver_cache", []()
        {
          GetSolverCache().clear(DestroyResources);
          GetMPISolverCache().clear(DestroyResources);
        });
  m.def("get_solver_cache_info", []()
        {
          auto single_keys = GetSolverCache().snapshot_keys();
          auto mpi_keys = GetMPISolverCache().snapshot_keys();

          py::list single_entries;
          for (const auto &k : single_keys) {
            py::dict entry;
            entry["n_rows"] = py::int_(k.n_rows);
            entry["nnz"] = py::int_(k.nnz);
            entry["mode"] = py::str(ModeToString(k.mode));
            entry["transpose_solve"] = py::bool_(k.transpose_solve);
            entry["block_dim"] = py::int_(k.block_dim);
            entry["structure_hash"] = py::int_(k.structure_hash);
            entry["config"] = py::str(k.config);
            single_entries.append(entry);
          }

          py::list mpi_entries;
          for (const auto &k : mpi_keys) {
            py::dict entry;
            entry["n_local"] = py::int_(k.n_local);
            entry["n_global"] = py::int_(k.n_global);
            entry["nnz"] = py::int_(k.nnz);
            entry["lrank"] = py::int_(k.lrank);
            entry["mode"] = py::str(ModeToString(k.mode));
            entry["transpose_solve"] = py::bool_(k.transpose_solve);
            entry["block_dim"] = py::int_(k.block_dim);
            entry["structure_hash"] = py::int_(k.structure_hash);
            entry["config"] = py::str(k.config);
            mpi_entries.append(entry);
          }

          py::dict single_gpu;
          single_gpu["size"] = py::int_(GetSolverCache().size());
          single_gpu["capacity"] = py::int_(GetSolverCache().capacity());
          single_gpu["entries"] = single_entries;

          py::dict mpi;
          mpi["size"] = py::int_(GetMPISolverCache().size());
          mpi["capacity"] = py::int_(GetMPISolverCache().capacity());
          mpi["entries"] = mpi_entries;

          py::dict info;
          info["single_gpu"] = single_gpu;
          info["mpi"] = mpi;
          info["isolated_mode"] = py::bool_(IsIsolatedMode());
          return info;
        });
}
