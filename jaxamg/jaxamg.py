import functools
import json
import os
import warnings
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
from .mpi_utils import (
    _mpi4jax_alltoallv_transpose,
    _mpi4jax_halo_gather,
    build_halo_plan,
    local_transpose_nnz,
    register_comm,
    resolve_comm,
)
from .utils import *

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_AMGX_CALL_NAME = "amgx_solve"
_AMGX_CALL_NAME_DOUBLE = "amgx_solve_double"
_AMGX_CALL_NAME_MPI = "amgx_solve_mpi"
_AMGX_CALL_NAME_MPI_DOUBLE = "amgx_solve_mpi_double"

# The native extension is loaded lazily on first use, so importing jaxamg
# (for sparsity tracing, matrix helpers, config handling, or on a CPU-only
# machine such as a CI runner) does not require the AmgX/CUDA libraries.
_amgx: Any = None

# Whether the native extension was compiled with MPI support (JAXAMG_WITH_MPI).
# A non-MPI build omits the MPI FFI handlers entirely. Set by _ensure_backend().
HAS_MPI = False


def _ensure_backend() -> Any:
    """Load the native AmgX extension and register its FFI targets (once)."""
    global _amgx, HAS_MPI
    if _amgx is not None:
        return _amgx
    from ._ext import _amgx as ext

    HAS_MPI = bool(getattr(ext, "mpi_enabled", False))
    ffi.register_ffi_target(
        _AMGX_CALL_NAME, ext.get_amgx_solve_handler(), platform="CUDA"
    )
    ffi.register_ffi_target(
        _AMGX_CALL_NAME_DOUBLE, ext.get_amgx_solve_double_handler(), platform="CUDA"
    )
    if HAS_MPI:
        ffi.register_ffi_target(
            _AMGX_CALL_NAME_MPI, ext.get_amgx_solve_mpi_handler(), platform="CUDA"
        )
        ffi.register_ffi_target(
            _AMGX_CALL_NAME_MPI_DOUBLE,
            ext.get_amgx_solve_mpi_double_handler(),
            platform="CUDA",
        )
    _amgx = ext
    return ext


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
    x0: ArrayLike | None = None,
    config_str: str = "",
    transpose_solve: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Low-level FFI call to AmgX solver (non-differentiable)."""

    _ensure_backend()
    b = jnp.asarray(b)

    out_spec = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((3 + res_history_len,), b.dtype),
    )

    call_name = _AMGX_CALL_NAME
    if b.dtype == jnp.float64:
        call_name = _AMGX_CALL_NAME_DOUBLE

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None, None],
        output_layouts=None,
        vmap_method="sequential",
    )
    # The x0 slot is a required input; pass b as a same-shape dummy when
    # unused (ignored by the C++ side when use_x0 is 0).
    results = call(
        row_ptrs,
        col_indices,
        values,
        b,
        x0 if x0 is not None else b,
        config=config_str,
        transpose_solve=np.int32(transpose_solve),
        return_stats=np.int32(return_stats),
        reuse_setup=np.int32(reuse_setup),
        use_x0=np.int32(use_x0),
    )

    return cast(tuple, results)


def _amgx_solve_mpi_impl(
    row_ptrs: ArrayLike,
    col_indices: ArrayLike,
    values: ArrayLike,
    b: ArrayLike,
    x0: ArrayLike | None,
    nglobal: ArrayLike,
    comm_ptr: ArrayLike,
    lrank: ArrayLike,
    config_str: str = "",
    transpose_solve: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Low-level FFI call to AmgX MPI solver (non-differentiable)."""

    _ensure_backend()
    b = jnp.asarray(b)

    out_spec = (
        jax.ShapeDtypeStruct(b.shape, b.dtype),
        jax.ShapeDtypeStruct((3 + res_history_len,), b.dtype),
    )

    call_name = _AMGX_CALL_NAME_MPI
    if b.dtype == jnp.float64:
        call_name = _AMGX_CALL_NAME_MPI_DOUBLE

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None, None, None, None, None],
        output_layouts=None,
        vmap_method="sequential",
    )
    # The x0 slot is a required input; pass b as a same-shape dummy when
    # unused (ignored by the C++ side when use_x0 is 0).
    results = call(
        row_ptrs,
        col_indices,
        values,
        b,
        x0 if x0 is not None else b,
        nglobal,
        comm_ptr,
        lrank,
        config=config_str,
        transpose_solve=np.int32(transpose_solve),
        return_stats=np.int32(return_stats),
        reuse_setup=np.int32(reuse_setup),
        use_x0=np.int32(use_x0),
    )

    return cast(tuple, results)


@functools.lru_cache(maxsize=32)
def _get_solver_primitive(
    config_str: str,
    is_symmetric: bool = False,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
) -> Callable:
    """
    Returns a JAX custom_vjp primitive for AmgX solve with a specific configuration.
    Cached to avoid recompilation for identical configurations.

    reuse_setup: Skip warm AMGX resetup and keep the cached hierarchy.
    res_history_len: Residual-history slots appended to the stats output.
    use_x0: Honor the x0 operand as the initial guess (otherwise it is a
        same-shape dummy and the solve starts from zero).
    """

    @jax.custom_vjp
    def solve(A: jsp.BCSR, b: jax.Array, x0: jax.Array) -> tuple[jax.Array, jax.Array]:
        x, info = _amgx_solve_impl(
            A.indptr,
            A.indices,
            A.data,
            b,
            x0,
            config_str=config_str,
            return_stats=return_stats,
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=use_x0,
        )
        return x, info

    def fwd(
        A: jsp.BCSR, b: jax.Array, x0: jax.Array
    ) -> tuple[tuple[jax.Array, jax.Array], tuple[jsp.BCSR, jax.Array]]:
        x, info = solve(A, b, x0)
        # Returns ((x, info), residuals)
        return (x, info), (A, x)

    def bwd(
        residuals: tuple[jsp.BCSR, jax.Array], g: tuple[jax.Array, jax.Array]
    ) -> tuple[jsp.BCSR, jax.Array, jax.Array]:
        g_x = g[0]
        A, x = residuals

        # Solve A^T λ = g_x (always from a zero start: x0 shifts only the
        # forward iteration, never the solution, so the adjoint ignores it).
        solver = _get_solver_primitive(
            config_str,
            is_symmetric,
            return_stats=False,
            reuse_setup=reuse_setup,
        )

        # Check if matrix is symmetric
        if is_symmetric:
            adj_b, _ = solver(A, g_x, g_x)
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
                    reuse_setup=reuse_setup,
                )
            except Exception:
                A_T = jsp.BCSR.from_bcoo(A.to_bcoo().transpose())
                adj_b, _ = solver(A_T, g_x, g_x)

        n = A.shape[0]
        row_lengths = A.indptr[1:] - A.indptr[:-1]

        # Safe gradient computation
        row_indices = jnp.repeat(
            jnp.arange(n, dtype=jnp.int32), row_lengths, total_repeat_length=len(A.data)
        )
        grad_values = -adj_b[row_indices] * x[A.indices]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        # The exact solution does not depend on the initial guess.
        return grad_A, adj_b, jnp.zeros_like(adj_b)

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
    nnz_out: int | None = None,
    n_ghost: int = 0,
    return_stats: bool = False,
    reuse_setup: bool = False,
    res_history_len: int = 0,
    use_x0: bool = False,
) -> Callable:
    """
    Create cached JAX custom_vjp primitive for MPI AmgX solve.
    Supports automatic differentiation in distributed setting.

    Uses mpi4jax for MPI communication (transpose, backward halo exchange).
    GPU vs CPU MPI is controlled by MPI4JAX_USE_CUDA_MPI environment variable:
    - MPI4JAX_USE_CUDA_MPI=1: Use GPU-aware MPI (requires CUDA-aware MPI library)
    - MPI4JAX_USE_CUDA_MPI=0: Use CPU staging (copies GPU<->CPU for MPI)

    Args:
        max_nnz: Maximum local nnz of A across ranks, for the transpose send
                 buffers. Required for non-symmetric matrices (backward pass).
        nnz_out: This rank's local nnz of A^T, for the transpose output buffers.
                 Differs from the local nnz of A for structurally nonsymmetric
                 patterns; required for non-symmetric matrices.
    """

    # Backward-pass collectives run on the user's communicator (recovered from
    # comm_ptr), which may be a subcommunicator -- not MPI.COMM_WORLD.
    comm = resolve_comm(comm_ptr)

    # The backward pass's gradient w.r.t. A needs the solution at the columns
    # this rank's rows reference. The halo plan (col_to_combined, send_ids,
    # recv_ghost_slot) is pattern-specific, so it flows as custom_vjp operands
    # rather than being baked into this memoized factory; only the static ghost
    # count n_ghost is captured here.
    @jax.custom_vjp
    def solve(
        A: jsp.BCSR,
        b: jax.Array,
        x0: jax.Array,
        col_to_combined: jax.Array,
        send_ids: jax.Array,
        recv_ghost_slot: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
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
            x0,
            nglobal_arr,
            comm_ptr_arr,
            lrank_arr,
            config_str=config_str,
            return_stats=return_stats,
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=use_x0,
        )

        return x, info

    def fwd(A, b, x0, col_to_combined, send_ids, recv_ghost_slot):
        out = solve(A, b, x0, col_to_combined, send_ids, recv_ghost_slot)
        x, info = out
        return out, (
            A,
            x,
            col_to_combined,
            send_ids,
            recv_ghost_slot,
        )

    def bwd(residuals, g):
        g_x, _ = g
        A, x, col_to_combined, send_ids, recv_ghost_slot = residuals

        # Backward solves always start from zero: x0 shifts only the forward
        # iteration, never the solution, so the adjoint ignores it (and skips
        # the residual-history readback).
        adj_solver = _get_solver_primitive_mpi(
            config_str,
            nglobal,
            comm_ptr,
            lrank,
            is_symmetric=is_symmetric,
            recvcounts_tuple=recvcounts_tuple,
            max_nnz=max_nnz,
            nnz_out=nnz_out,
            n_ghost=n_ghost,
            return_stats=return_stats,
            reuse_setup=reuse_setup,
        )

        # Backward solve: A^T @ adj_b = g_x
        if is_symmetric:
            # Symmetric: skip the distributed transpose.
            adj_b, _ = adj_solver(
                A, g_x, g_x, col_to_combined, send_ids, recv_ghost_slot
            )
        else:
            # Distributed transpose via mpi4jax (JIT-compatible, GPU-direct when
            # MPI4JAX_USE_CUDA_MPI=1).
            if recvcounts_tuple is None or max_nnz is None or nnz_out is None:
                raise ValueError(
                    "recvcounts_tuple, max_nnz, and nnz_out are required for the "
                    "distributed transpose of non-symmetric matrices. Ensure they "
                    "are computed when creating the solver (via cache_mpi_metadata "
                    "or dynamic computation)."
                )
            # Order the transpose after the forward solve (it otherwise reads
            # only A); without this XLA may interleave their MPI collectives in
            # a rank-inconsistent order and deadlock.
            a_data, a_indices, a_indptr, _ = jax.lax.optimization_barrier(
                (A.data, A.indices, A.indptr, x)
            )
            at_data, at_indices, at_indptr = _mpi4jax_alltoallv_transpose(
                a_data, a_indices, a_indptr, recvcounts_tuple, comm, max_nnz, nnz_out
            )

            # Reconstruct BCSR for A^T
            A_T = jsp.BCSR((at_data, at_indices, at_indptr), shape=A.shape)

            adj_b, _ = adj_solver(
                A_T, g_x, g_x, col_to_combined, send_ids, recv_ghost_slot
            )

        # Gradient w.r.t. A: ∂L/∂A_ij = -adj_b[i] * x[j]. Fetch only the solution
        # entries this rank's rows reference via the halo exchange, ordered after
        # the backward solve (same as the transpose above) so all ranks issue MPI
        # collectives in a consistent order.
        x_bar, _ = jax.lax.optimization_barrier((x, adj_b))
        x_combined = _mpi4jax_halo_gather(
            x_bar, send_ids, recv_ghost_slot, n_ghost, comm
        )

        n_local = A.shape[0]
        row_lengths = A.indptr[1:] - A.indptr[:-1]
        row_indices = jnp.repeat(
            jnp.arange(n_local, dtype=jnp.int32),
            row_lengths,
            total_repeat_length=len(A.data),
        )
        grad_values = -adj_b[row_indices] * x_combined[col_to_combined]
        grad_A = jsp.BCSR((grad_values, A.indices, A.indptr), shape=A.shape)

        # The exact solution does not depend on the initial guess.
        return grad_A, adj_b, jnp.zeros_like(adj_b), None, None, None

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
    x0: ArrayLike | None = None,
    config: dict | None = None,
    comm: "Comm | None" = None,
    nglobal: int | None = None,
    partition_info: tuple[int, int] | None = None,
    save_stats_file: str | os.PathLike | None = None,
    reuse_setup: bool = False,
    **kwargs: Any,
) -> tuple[jax.Array, dict]:
    """Solve `Ax=b` using the AmgX backend. See [Examples](examples.md) for usage.

    Args:
        A: Matrix or callable operator A(x). All matrices/operators are converted to `jax.experimental.sparse.bcsr` sparse matrices internally. In MPI mode this is the local partition.
        b: Right-hand-side vector. In MPI mode this is the local RHS partition.
        x0: Optional initial guess (same shape as `b`; local partition in MPI mode). Defaults to zero. A good warm start (e.g. the previous solution in a time-stepping or optimization loop) cuts iterations; it does not change the converged solution or its gradients. Note that with the default `RELATIVE_INI` convergence the tolerance is relative to the *initial* residual, so a very good `x0` tightens the target; consider `convergence="ABSOLUTE"` for warm-started loops.
        config: AmgX configuration dictionary (see [Solver Configuration](config.md) for details). If `None`, JAX-AMG defaults are used.
        comm: MPI communicator (typically `mpi4py.MPI.COMM_WORLD`). If provided, the solve runs in MPI mode. If not provided, MPI mode can still be used if MPI metadata has already been attached via `with_cache(..., mpi=...)`.
        nglobal: Global matrix row count for MPI mode. Required when `comm` is provided and MPI metadata is not pre-attached to `A`.
        partition_info: `(row_start, row_end)` owned by this rank in MPI mode.  Required when `comm` is provided and MPI metadata is not pre-attached to `A`.
        save_stats_file: Optional file path to save detailed AmgX solver statistics.  If None, no file is created.
        reuse_setup: For repeated solves with the same sparsity pattern, skip warm `AMGX_solver_resetup` and keep the cached hierarchy. This is cheaper per solve but may require more iterations if matrix coefficients change significantly.
        **kwargs: Additional AmgX config parameters. These override values in `config` when both are provided.

    Returns:
        x: Solution vector (float32 or float64). In MPI mode, returns local portion.
        info: Dictionary containing `iterations`, `residual`, `status`, and `residual_history` (residual norm per outer iteration, entry 0 being the initial residual; inside `jit` it has fixed length `max_iters + 1` with NaN padding past entry `iterations`).
    """

    b = jnp.asarray(b)

    # Check for GPU backend
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"AMGX requires a GPU backend, but JAX is using '{jax.default_backend()}'. "
            "Please ensure you have a CUDA-enabled GPU and JAX is installed with CUDA support."
        )

    # Load the native extension and register FFI targets (sets HAS_MPI).
    _ensure_backend()

    # MPI cache may be pre-attached to A via `with_cache`
    mpi_cache = getattr(A, "_mpi_cache", None)

    # Prepare configuration string/file (skip if using mpi_cache which already has config_str)
    if mpi_cache is not None:
        if config is not None or kwargs:
            warnings.warn(
                "A carries cached MPI metadata (with_cache(..., mpi=...)); the "
                "config/kwargs passed to solve() are ignored in favor of the "
                "cached config. Pass the config to cache_mpi_metadata() instead.",
                stacklevel=2,
            )
        config_str = mpi_cache["config_str"]
        if save_stats_file is not None and '"print_solve_stats"' not in config_str:
            warnings.warn(
                "save_stats_file was passed, but the cached MPI config was "
                "prepared without stats output; the stats file will be missing "
                "solver statistics. Pass save_stats=True to cache_mpi_metadata().",
                stacklevel=2,
            )
    else:
        config_str = amgx_config.prepare_config(
            config,
            save_stats=(save_stats_file is not None),
            mpi=(comm is not None),
            **kwargs,
        )

    # Residual-history slots appended to the stats output (one per outer
    # iteration, plus the initial residual).
    res_history_len = amgx_config.outer_max_iters(config_str) + 1

    # Detect desired precision (non-float RHS dtypes are promoted to float32)
    target_dtype = get_preferred_dtype(A, b)
    if b.dtype != target_dtype:
        b = b.astype(target_dtype)

    use_x0 = x0 is not None
    if use_x0:
        x0 = jnp.asarray(x0)
        if x0.shape != b.shape:
            raise ValueError(
                f"x0 must have the same shape as b; got {x0.shape} vs {b.shape}"
            )
        if x0.dtype != target_dtype:
            x0 = x0.astype(target_dtype)
    # The primitive always takes an x0 operand; b doubles as a same-shape
    # dummy that the backend ignores when use_x0 is off.
    x0_arg = x0 if use_x0 else b

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
            nnz_out = mpi_cache["nnz_out"]  # None when cached as symmetric
            halo_plan = mpi_cache["halo_plan"]
            solver = _get_solver_primitive_mpi(
                mpi_cache["config_str"],
                mpi_cache["nglobal"],
                mpi_cache["comm_ptr"],
                mpi_cache["lrank"],
                is_symmetric=is_symmetric,
                recvcounts_tuple=recvcounts_tuple,
                max_nnz=max_nnz,
                nnz_out=nnz_out,
                n_ghost=halo_plan.n_ghost,
                return_stats=1 if save_stats_file else 0,
                reuse_setup=reuse_setup,
                res_history_len=res_history_len,
                use_x0=use_x0,
            )

            x, info = solver(
                A_csr,
                b,
                x0_arg,
                jnp.asarray(halo_plan.col_to_combined),
                jnp.asarray(halo_plan.send_ids_2d),
                jnp.asarray(halo_plan.recv_ghost_slot_2d),
            )
        elif comm is not None:
            # Compute metadata dynamically
            import importlib.util

            if importlib.util.find_spec("mpi4py") is None:
                raise ImportError(
                    "mpi4py is required for MPI mode. Install it with: pip install mpi4py"
                )

            # Get MPI rank and compute local GPU assignment
            rank = comm.Get_rank()
            lrank = rank % jax.device_count()
            # Register the communicator and get its address (so the backward pass
            # can recover it for its collectives).
            comm_ptr = register_comm(comm)

            # Gather partition sizes from all ranks (row partition + displacements)
            n_local = A_csr.shape[0]
            all_sizes_list = comm.allgather(n_local)
            recvcounts_val = np.array(all_sizes_list, dtype=np.int32)
            displs_val = np.cumsum(np.concatenate(([0], recvcounts_val[:-1]))).astype(
                np.int32
            )
            recvcounts_tuple = tuple(recvcounts_val.tolist())

            # Validate partition_info against the partition actually implied by
            # the local matrix shapes (which is what AmgX uses). Reduce first so
            # every rank raises together -- a rank-divergent raise would leave
            # the other ranks deadlocked in the collectives below.
            from mpi4py import MPI

            row_start = int(displs_val[rank])
            derived_partition = (row_start, row_start + n_local)
            mismatch = tuple(partition_info) != derived_partition
            if comm.allreduce(mismatch, op=MPI.LOR):
                detail = (
                    f"rank {rank}: partition_info {tuple(partition_info)} != "
                    f"derived {derived_partition}"
                    if mismatch
                    else f"rank {rank} is consistent, but another rank's is not"
                )
                raise ValueError(
                    "partition_info does not match the row partition derived "
                    f"from the local matrix shapes ({detail}). Each rank must "
                    "pass its own (row_start, row_end) matching its local "
                    "partition."
                )

            # max nnz across ranks (transpose send buffers) + this rank's local
            # nnz(A^T) (its output buffers).
            max_nnz = max(comm.allgather(len(A_csr.data)))
            nnz_out = local_transpose_nnz(A_csr.indices, recvcounts_tuple, comm)

            # Halo-exchange plan for the backward pass (fetches only the remote
            # solution entries this rank references, instead of all-gathering).
            halo_plan = build_halo_plan(
                A_csr.indices,
                recvcounts_tuple,
                (row_start, row_start + n_local),
                comm,
            )

            solver = _get_solver_primitive_mpi(
                config_str,
                nglobal,
                comm_ptr,
                lrank,
                is_symmetric=is_symmetric,
                recvcounts_tuple=recvcounts_tuple,
                max_nnz=max_nnz,
                nnz_out=nnz_out,
                n_ghost=halo_plan.n_ghost,
                return_stats=1 if save_stats_file else 0,
                reuse_setup=reuse_setup,
                res_history_len=res_history_len,
                use_x0=use_x0,
            )
            x, info = solver(
                A_csr,
                b,
                x0_arg,
                jnp.asarray(halo_plan.col_to_combined),
                jnp.asarray(halo_plan.send_ids_2d),
                jnp.asarray(halo_plan.recv_ghost_slot_2d),
            )

    else:
        # Single-GPU mode: use int32 indices
        A_csr = to_bcsr_matrix(A, b)
        # Get cached primitive for this configuration
        solver = _get_solver_primitive(
            config_str,
            is_symmetric=is_symmetric,
            return_stats=1 if save_stats_file else 0,
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=use_x0,
        )

        x, info = solver(A_csr, b, x0_arg)

    if isinstance(info, jax.core.Tracer):
        # Inside JIT (or another trace): info elements are tracers; return as-is.
        # The history keeps its fixed trace-time length (outer max_iters + 1),
        # NaN-padded past entry `iterations`.
        return x, {
            "iterations": info[0],
            "residual": info[1],
            "status": info[2],
            "residual_history": info[3:],
        }

    info_dict = {
        "iterations": int(info[0]),
        "residual": float(info[1]),
        "status": AMGXStatus(int(info[2])),
        # Entry i is the residual norm after outer iteration i (entry 0 is the
        # initial residual); trim the NaN padding outside a trace.
        "residual_history": info[3 : 4 + int(info[0])],
    }
    if save_stats_file is not None:
        try:
            stats_str = _ensure_backend().get_stats_string()
        except AttributeError:
            # Older extension without stats capture; nothing to save.
            stats_str = None
        if stats_str is not None:
            _format_and_save_stats(
                stats_str, save_stats_file, comm=comm, mpi_cache=mpi_cache
            )
    return x, info_dict


def clear_solver_cache() -> None:
    """
    Clear the internal C++ AmgX solver cache.
    This releases all cached AmgX resources (matrices, solvers, vectors).
    """
    _ensure_backend().clear_solver_cache()


def get_solver_cache_info() -> dict[str, Any]:
    """
    Inspect the internal C++ AmgX solver caches.

    Returns:
        A dictionary with cache size/capacity and entry summaries
        for single-GPU and MPI caches, plus whether isolated mode
        (`JAXAMG_CACHE_SIZE=0`) is active.
    """
    solver_info = _ensure_backend().get_solver_cache_info()

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
    _ensure_backend().finalize()
