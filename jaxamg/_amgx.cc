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
          .Ret<ffi::Buffer<ffi::F32>>()             // x
          .Ret<ffi::Buffer<ffi::F32>>()             // stats
          .Attr<std::string_view>("config")         // config string
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
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
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
  );

#ifdef JAXAMG_WITH_MPI
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
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
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
          .Attr<int32_t>("transpose_solve")         // transpose flag
          .Attr<int32_t>("return_stats")            // return stats flag
  );

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
  m.def("get_amgx_allgather_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxAllGather)); });
  m.def("get_amgx_allgather_double_handler", []()
        { return py::capsule(reinterpret_cast<void *>(AmgxAllGatherDouble)); });
  m.attr("mpi_enabled") = py::bool_(true);
#else
  m.attr("mpi_enabled") = py::bool_(false);
#endif

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
