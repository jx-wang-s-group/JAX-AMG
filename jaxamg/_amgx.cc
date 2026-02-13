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
#include "_amgx_utils.h"
#include "_amgx_solvers.h"

namespace py = pybind11;
namespace ffi = xla::ffi;

namespace
{
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
          .Attr<int32_t>("transpose_solve")         // transpose flag
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
        {
          GetSolverCache().clear(DestroyResources);
          GetMPISolverCache().clear(DestroyResources);
        });
}
