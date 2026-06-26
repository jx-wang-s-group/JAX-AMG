import functools
import json
import os
from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.experimental.sparse as jsp
import jax.ffi as ffi
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from . import config as amgx_config
from ._ext import _amgx
from .mpi_utils import (
    _amgx_allgather_impl,
    _mpi4jax_allgatherv,
    _mpi4jax_alltoallv_transpose,
)
from .utils import *

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_AMGX_CALL_NAME = "amgx_solve"
_AMGX_CALL_NAME_DOUBLE = "amgx_solve_double"
_AMGX_CALL_NAME_MPI = "amgx_solve_mpi"
_AMGX_CALL_NAME_MPI_DOUBLE = "amgx_solve_mpi_double"

# Whether the native extension was compiled with MPI support (JAXAMG_WITH_MPI).
# A non-MPI build omits the MPI FFI handlers entirely, so the single-GPU path
# below must be the only thing registered at import time.
HAS_MPI = bool(getattr(_amgx, "mpi_enabled", False))

# Get the handler from C++ and register for CUDA platform
_AMGX_HANDLER = _amgx.get_amgx_solve_handler()
_AMGX_HANDLER_DOUBLE = _amgx.get_amgx_solve_double_handler()

ffi.register_ffi_target(_AMGX_CALL_NAME, _AMGX_HANDLER, platform="CUDA")
ffi.register_ffi_target(_AMGX_CALL_NAME_DOUBLE, _AMGX_HANDLER_DOUBLE, platform="CUDA")

if HAS_MPI:
    _AMGX_HANDLER_MPI = _amgx.get_amgx_solve_mpi_handler()
    _AMGX_HANDLER_MPI_DOUBLE = _amgx.get_amgx_solve_mpi_double_handler()
    ffi.register_ffi_target(_AMGX_CALL_NAME_MPI, _AMGX_HANDLER_MPI, platform="CUDA")
    ffi.register_ffi_target(
        _AMGX_CALL_NAME_MPI_DOUBLE, _AMGX_HANDLER_MPI_DOUBLE, platform="CUDA"
    )


class AMGXStatus(IntEnum):
    """High-level AmgX solve status codes returned in `info["status"]` after calling `jaxamg.solve`.

    These values are mapped from the native backend status for quick checks in
    Python code and in docs.

    Members:
        - `SUCCESS`: Solve converged successfully.
        - `FAILED`: Solver failed due to an internal/runtime error.
        - `DIVERGED`: Iterations diverged.
        - `NOT_CONVERGED`: Reached stopping criteria without convergence.
    """

    SUCCESS = 0
    FAILED = 1
    DIVERGED = 2
    NOT_CONVERGED = 3

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}"


def _amgx_solve_impl(
    row_ptrs: ArrayLike,
    col_indices: ArrayLike,
    values: ArrayLike,
    b: ArrayLike,
    config_str: str = "",
    transpose_solve: bool = False,
    return_stats: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Low-level FFI call to AmgX solver (non-differentiable)."""

    b = jnp.asarray(b)

    out_spec = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((3,), b.dtype),
    )

    call_name = _AMGX_CALL_NAME
    if b.dtype == jnp.float64:
        call_name = _AMGX_CALL_NAME_DOUBLE

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None],
        output_layouts=None,
        vmap_method="sequential",
    )
    results = call(
        row_ptrs,
        col_indices,
        values,
        b,
        config=config_str,
        transpose_solve=np.int32(transpose_solve),
        return_stats=np.int32(return_stats),
    )

    return cast(tuple, results)


def _amgx_solve_mpi_impl(
    row_ptrs: ArrayLike,
    col_indices: ArrayLike,
    values: ArrayLike,
    b: ArrayLike,
    nglobal: ArrayLike,
    comm_ptr: ArrayLike,
    lrank: ArrayLike,
    config_str: str = "",
    transpose_solve: bool = False,
    return_stats: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Low-level FFI call to AmgX MPI solver (non-differentiable)."""

    b = jnp.asarray(b)

    out_spec = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((3,), b.dtype),
    )

    call_name = _AMGX_CALL_NAME_MPI
    if b.dtype == jnp.float64:
        call_name = _AMGX_CALL_NAME_MPI_DOUBLE

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None, None, None, None],
        output_layouts=None,
        vmap_method="sequential",
    )
    results = call(
        row_ptrs,
        col_indices,
        values,
        b,
        nglobal,
        comm_ptr,
        lrank,
        config=config_str,
        transpose_solve=np.int32(transpose_solve),
        return_stats=np.int32(return_stats),
    )

    return cast(tuple, results)


@functools.lru_cache(maxsize=32)
def _get_solver_primitive(
    config_str: str, is_symmetric: bool = False, return_stats: bool = False
) -> Callable:
    """
    Returns a JAX custom_vjp primitive for AmgX solve with a specific configuration.
    Cached to avoid recompilation for identical configurations.
    """

    @jax.custom_vjp
    def solve(A: jsp.BCSR, b: jax.Array) -> tuple[jax.Array, jax.Array]:
        x, info = _amgx_solve_impl(
            A.indptr,
            A.indices,
            A.data,
            b,
            config_str=config_str,
            return_stats=return_stats,
        )
        return x, info

    def fwd(
        A: jsp.BCSR, b: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jsp.BCSR, jax.Array]]:
        x, info = solve(A, b)
        # Returns ((x, info), residuals)
        return (x, info), (A, x)

    def bwd(
        residuals: tuple[jsp.BCSR, jax.Array], g: tuple[jax.Array, jax.Array]
    ) -> tuple[jsp.BCSR, jax.Array]:
        g_x = g[0]
        A, x = residuals

        # Solve A^T λ = g_x
        solver = _get_solver_primitive(config_str, is_symmetric, return_stats=False)

        # Check if matrix is symmetric
        if is_symmetric:
            adj_b, _ = solver(A, g_x)
        else:
            # Use backend transposed solve and keep compatibility fallback.
            try:
                adj_b, _ = _amgx_solve_impl(
                    A.indptr,
                    A.indices,
                    A.data,
                    g_x,
                    config_str=config_str,
                    transpose_solve=True,
                )
            except Exception:
                A_T = jsp.BCSR.from_bcoo(A.to_bcoo().transpose())
                adj_b, _ = solver(A_T, g_x)

        n = A.shape[0]
        row_lengths = A.indptr[1:] - A.indptr[:-1]

        # Safe gradient computation
        row_indices = jnp.repeat(
            jnp.arange(n, dtype=jnp.int32), row_lengths, total_repeat_length=len(A.data)
        )
        grad_values = -adj_b[row_indices] * x[A.indices]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        return grad_A, adj_b

    solve.defvjp(fwd, bwd)
    return solve


@functools.lru_cache(maxsize=32)
def _get_solver_primitive_mpi(
    config_str: str,
    nglobal: int,
    comm_ptr: int,
    lrank: int,
    is_symmetric: bool = False,
    recvcounts_tuple: tuple[int, ...] | None = None,
    max_nnz: int | None = None,
    return_stats: bool = False,
) -> Callable:
    """
    Create cached JAX custom_vjp primitive for MPI AmgX solve.
    Supports automatic differentiation in distributed setting.

    Uses mpi4jax for MPI communication (allgather, transpose).
    GPU vs CPU MPI is controlled by MPI4JAX_USE_CUDA_MPI environment variable:
    - MPI4JAX_USE_CUDA_MPI=1: Use GPU-aware MPI (requires CUDA-aware MPI library)
    - MPI4JAX_USE_CUDA_MPI=0: Use CPU staging (copies GPU<->CPU for MPI)

    Args:
        max_nnz: Maximum nnz across all ranks for transpose buffer sizing.
                 Must be provided when using non-symmetric matrices (required for backward pass).
    """

    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    def allgather(
        sendbuf: ArrayLike,
        recvcounts: ArrayLike,
        displs: ArrayLike,
        comm_ptr_arr: ArrayLike,
    ) -> jax.Array:
        # Use mpi4jax for MPI communication
        if recvcounts_tuple is not None:
            return _mpi4jax_allgatherv(sendbuf, recvcounts_tuple, comm)
        # Fallback to FFI-based allgather if recvcounts_tuple not available
        return _amgx_allgather_impl(
            sendbuf, recvcounts, displs, comm_ptr_arr, nglobal=nglobal
        )

    @jax.custom_vjp
    def solve(
        A: jsp.BCSR, b: jax.Array, recvcounts: jax.Array, displs: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        nglobal_arr = jnp.array([nglobal], dtype=jnp.int32)

        # Split 64-bit comm_ptr into two int32 values for FFI
        comm_ptr_low_unsigned = comm_ptr & 0xFFFFFFFF
        comm_ptr_high_unsigned = (comm_ptr >> 32) & 0xFFFFFFFF
        comm_ptr_low_signed = np.int32(np.uint32(comm_ptr_low_unsigned))
        comm_ptr_high_signed = np.int32(np.uint32(comm_ptr_high_unsigned))

        comm_ptr_arr = jnp.array(
            [comm_ptr_low_signed, comm_ptr_high_signed], dtype=jnp.int32
        )
        lrank_arr = jnp.array([lrank], dtype=jnp.int32)

        x, info = _amgx_solve_mpi_impl(
            A.indptr,
            A.indices,
            A.data,
            b,
            nglobal_arr,
            comm_ptr_arr,
            lrank_arr,
            config_str=config_str,
            return_stats=return_stats,
        )

        return (x, info), comm_ptr_arr

    def fwd(
        A: jsp.BCSR, b: jax.Array, recvcounts: jax.Array, displs: jax.Array
    ) -> tuple[
        tuple[tuple[jax.Array, jax.Array], jax.Array],
        tuple[jsp.BCSR, jax.Array, jax.Array, jax.Array, jax.Array],
    ]:
        out = solve(A, b, recvcounts, displs)
        (x, info), comm_ptr_arr = out
        return out, (
            A,
            x,
            recvcounts,
            displs,
            comm_ptr_arr,
        )

    def bwd(
        residuals: tuple[jsp.BCSR, jax.Array, jax.Array, jax.Array, jax.Array],
        g: tuple[tuple[jax.Array, jax.Array], jax.Array],
    ) -> tuple[jsp.BCSR, jax.Array, None, None]:
        (g_x, _), _ = g
        A, x, recvcounts, displs, comm_ptr_arr = residuals

        # Backward solve: A^T @ adj_b = g_x

        # Check if matrix is symmetric
        if is_symmetric:
            # Skip distributed transpose
            (adj_b, _), _ = solve(A, g_x, recvcounts, displs)
        else:
            # Use mpi4jax for distributed transpose (JIT-compatible, GPU-direct when MPI4JAX_USE_CUDA_MPI=1)
            if recvcounts_tuple is None:
                raise ValueError(
                    "recvcounts_tuple is required for distributed transpose. "
                    "This should be provided automatically when using MPI mode."
                )
            if max_nnz is None:
                raise ValueError(
                    "max_nnz is required for distributed transpose of non-symmetric matrices. "
                    "Ensure max_nnz is computed when creating the solver (via cache_mpi_metadata or dynamic computation)."
                )
            # Order the transpose after the forward solve (it otherwise reads
            # only A); without this XLA may interleave their MPI collectives in
            # a rank-inconsistent order and deadlock.
            a_data, a_indices, a_indptr, _ = jax.lax.optimization_barrier(
                (A.data, A.indices, A.indptr, x)
            )
            at_data, at_indices, at_indptr = _mpi4jax_alltoallv_transpose(
                a_data, a_indices, a_indptr, recvcounts_tuple, comm, max_nnz
            )

            # Reconstruct BCSR for A^T
            A_T = jsp.BCSR((at_data, at_indices, at_indptr), shape=A.shape)

            (adj_b, _), _ = solve(A_T, g_x, recvcounts, displs)

        # Gather x across all ranks for gradient computation; order it after the
        # backward solve (same as the transpose above) so all ranks issue MPI
        # collectives in a consistent order.
        x_bar, _ = jax.lax.optimization_barrier((x, adj_b))
        x_global = allgather(x_bar, recvcounts, displs, comm_ptr_arr)

        # Compute ∂L/∂A: ∂L/∂A_ij = -adj_b[i] * x[j]
        n_local = A.shape[0]
        row_lengths = A.indptr[1:] - A.indptr[:-1]
        row_indices = jnp.repeat(
            jnp.arange(n_local, dtype=jnp.int32),
            row_lengths,
            total_repeat_length=len(A.data),
        )
        grad_values = -adj_b[row_indices] * x_global[A.indices.astype(jnp.int32)]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        return grad_A, adj_b, None, None

    solve.defvjp(fwd, bwd)
    return solve


def _format_and_save_stats(
    stats_str: str,
    save_stats_file: str | os.PathLike,
    comm: "Comm | None" = None,
    mpi_cache: dict | None = None,
) -> None:
    """Resolve MPI rank and save formatted AmgX statistics to a file."""
    rank: int | None = None
    if comm is not None:
        rank = comm.Get_rank()
    elif mpi_cache is not None and "lrank" in mpi_cache:
        rank = mpi_cache["lrank"]
    format_amgx_stats(stats_str, save_stats_file, rank=rank)
    if rank is None or rank == 0:
        print(f"Stats saved to {save_stats_file}")


def solve(
    A: MatrixOrOperator,
    b: ArrayLike,
    config: dict | None = None,
    comm: "Comm | None" = None,
    nglobal: int | None = None,
    partition_info: tuple[int, int] | None = None,
    save_stats_file: str | os.PathLike | None = None,
    **kwargs: Any,
) -> tuple[jax.Array, dict]:
    """Solve `Ax=b` using the AmgX backend. See [Examples](examples.md) for usage.

    Args:
        A: Matrix or callable operator A(x). All matrices/operators are converted to `jax.experimental.sparse.bcsr` sparse matrices internally. In MPI mode this is the local partition.
        b: Right-hand-side vector. In MPI mode this is the local RHS partition.
        config: AmgX configuration dictionary (see [Solver Configuration](config.md) for details). If `None`, JAX-AMG defaults are used.
        comm: MPI communicator (typically `mpi4py.MPI.COMM_WORLD`). If provided, the solve runs in MPI mode. If not provided, MPI mode can still be used if MPI metadata has already been attached via `with_cache(..., mpi=...)`.
        nglobal: Global matrix row count for MPI mode. Required when `comm` is provided and MPI metadata is not pre-attached to `A`.
        partition_info: `(row_start, row_end)` owned by this rank in MPI mode.  Required when `comm` is provided and MPI metadata is not pre-attached to `A`.
        save_stats_file: Optional file path to save detailed AmgX solver statistics.  If None, no file is created.
        **kwargs: Additional AmgX config parameters. These override values in `config` when both are provided.

    Returns:
        x: Solution vector (float32 or float64). In MPI mode, returns local portion.
        info: Dictionary containing `iterations`, `residual`, and `status`.
    """

    b = jnp.asarray(b)

    # Check for GPU backend
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"AMGX requires a GPU backend, but JAX is using '{jax.default_backend()}'. "
            "Please ensure you have a CUDA-enabled GPU and JAX is installed with CUDA support."
        )

    # MPI cache may be pre-attached to A via `with_cache`
    mpi_cache = getattr(A, "_mpi_cache", None)

    # Prepare configuration string/file (skip if using mpi_cache which already has config_str)
    if mpi_cache is not None:
        config_str = mpi_cache["config_str"]
    else:
        config_str = amgx_config.prepare_config(
            config, save_stats=(save_stats_file is not None), **kwargs
        )

    # Detect desired precision
    target_dtype = get_preferred_dtype(A, b)
    if target_dtype == jnp.float64 and b.dtype != jnp.float64:
        b = b.astype(jnp.float64)

    # Check for symmetry attribute on A
    is_symmetric = getattr(A, "_is_symmetric", False)

    # Branch: MPI mode or single-GPU mode
    if mpi_cache is not None or comm is not None:
        # MPI MODE
        if not HAS_MPI:
            raise RuntimeError(
                "jaxamg was built without MPI support, but an MPI solve was "
                "requested (comm was passed or MPI metadata is attached to A). "
                "Rebuild with MPI enabled (JAXAMG_ENABLE_MPI=1, with mpicxx on "
                "PATH) to use distributed solves."
            )
        if mpi_cache is None:
            # Validate parameters for non-cache path
            if nglobal is None:
                raise ValueError("nglobal must be provided when using MPI mode")
            if partition_info is None:
                raise ValueError(
                    "partition_info (row_start, row_end) must be provided when using MPI mode"
                )

        # Convert A to BCSR with int64 indices (required for MPI)
        A_csr = to_bcsr_matrix(A, b=b, use_int64_indices=True)

        if mpi_cache is not None:
            # Use pre-cached MPI metadata
            recvcounts_tuple = mpi_cache["recvcounts_tuple"]
            max_nnz = mpi_cache["max_nnz"]  # Always present in cache
            solver = _get_solver_primitive_mpi(
                mpi_cache["config_str"],
                mpi_cache["nglobal"],
                mpi_cache["comm_ptr"],
                mpi_cache["lrank"],
                is_symmetric=is_symmetric,
                recvcounts_tuple=recvcounts_tuple,
                max_nnz=max_nnz,
                return_stats=1 if save_stats_file else 0,
            )
            recvcounts = jnp.array(recvcounts_tuple, dtype=jnp.int32)
            displs = jnp.array(mpi_cache["displs_tuple"], dtype=jnp.int32)

            (x, info), _ = solver(A_csr, b, recvcounts, displs)
        elif comm is not None:
            # Compute metadata dynamically
            try:
                from mpi4py import MPI
            except ImportError:
                raise ImportError(
                    "mpi4py is required for MPI mode. Install it with: pip install mpi4py"
                )

            # Get MPI rank and compute local GPU assignment
            rank = comm.Get_rank()
            lrank = rank % jax.device_count()
            # Get MPI communicator pointer
            comm_ptr = MPI._addressof(comm)

            # Gather partition sizes from all ranks for gradient allgather operation
            n_local = A_csr.shape[0]
            all_sizes_list = comm.allgather(n_local)
            recvcounts_val = np.array(all_sizes_list, dtype=np.int32)
            displs_val = np.cumsum(np.concatenate(([0], recvcounts_val[:-1]))).astype(
                np.int32
            )

            recvcounts = jnp.array(recvcounts_val)
            displs = jnp.array(displs_val)
            recvcounts_tuple = tuple(recvcounts_val.tolist())

            # Compute max nnz across all ranks for buffer sizing
            max_nnz = max(comm.allgather(len(A_csr.data)))

            solver = _get_solver_primitive_mpi(
                config_str,
                nglobal,
                comm_ptr,
                lrank,
                is_symmetric=is_symmetric,
                recvcounts_tuple=recvcounts_tuple,
                max_nnz=max_nnz,
                return_stats=1 if save_stats_file else 0,
            )
            (x, info), _ = solver(A_csr, b, recvcounts, displs)

    else:
        # Single-GPU mode: use int32 indices
        A_csr = to_bcsr_matrix(A, b)
        # Get cached primitive for this configuration
        solver = _get_solver_primitive(
            config_str,
            is_symmetric=is_symmetric,
            return_stats=1 if save_stats_file else 0,
        )

        x, info = solver(A_csr, b)

    try:
        info_dict = {
            "iterations": int(info[0]),
            "residual": float(info[1]),
            "status": AMGXStatus(int(info[2])),
        }
        if save_stats_file is not None:
            try:
                stats_str = _amgx.get_stats_string()
                _format_and_save_stats(
                    stats_str, save_stats_file, comm=comm, mpi_cache=mpi_cache
                )
            except AttributeError:
                pass
        return x, info_dict
    except Exception:
        # Inside JIT: info elements are tracers; return them as-is.
        return x, {"iterations": info[0], "residual": info[1], "status": info[2]}


def clear_solver_cache() -> None:
    """
    Clear the internal C++ AmgX solver cache.
    This releases all cached AmgX resources (matrices, solvers, vectors).
    """
    _amgx.clear_solver_cache()


def get_solver_cache_info() -> dict[str, Any]:
    """
    Inspect the internal C++ AmgX solver caches.

    Returns:
        A dictionary with cache size/capacity and entry summaries
        for single-GPU and MPI caches, plus whether isolated mode
        (`JAXAMG_CACHE_SIZE=0`) is active.
    """
    solver_info = _amgx.get_solver_cache_info()

    # Convert config strings to JSON
    solver_info["single_gpu"]["entries"] = [
        {**entry, "config": json.loads(entry["config"])}
        for entry in solver_info["single_gpu"]["entries"]
    ]
    solver_info["mpi"]["entries"] = [
        {**entry, "config": json.loads(entry["config"])}
        for entry in solver_info["mpi"]["entries"]
    ]

    return solver_info


def finalize() -> None:
    """
    Manually finalize AmgX resources.
    This clears the cache and calls AMGX_finalize.
    Normally only needed to be called manually in MPI mode to avoid shutdown-time resource warnings.
    """
    clear_solver_cache()
    _amgx.finalize()
