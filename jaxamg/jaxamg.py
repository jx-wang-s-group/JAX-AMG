import jax
import jax.numpy as jnp
import jax.ffi as ffi
import jax.experimental.sparse as jsp
import functools
import numpy as np
from enum import IntEnum

from . import _amgx, config as amgx_config
from .utils import *

from typing import cast, Callable, Any, TYPE_CHECKING
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_AMGX_CALL_NAME = "amgx_solve"
_AMGX_CALL_NAME_DOUBLE = "amgx_solve_double"
_AMGX_CALL_NAME_MPI = "amgx_solve_mpi"
_AMGX_CALL_NAME_MPI_DOUBLE = "amgx_solve_mpi_double"
_AMGX_CALL_NAME_ALLGATHER = "amgx_allgather"
_AMGX_CALL_NAME_ALLGATHER_DOUBLE = "amgx_allgather_double"

# Get the handler from C++ and register for CUDA platform
_AMGX_HANDLER = _amgx.get_amgx_solve_handler()
_AMGX_HANDLER_DOUBLE = _amgx.get_amgx_solve_double_handler()
_AMGX_HANDLER_MPI = _amgx.get_amgx_solve_mpi_handler()
_AMGX_HANDLER_MPI_DOUBLE = _amgx.get_amgx_solve_mpi_double_handler()
_AMGX_HANDLER_ALLGATHER = _amgx.get_amgx_allgather_handler()
_AMGX_HANDLER_ALLGATHER_DOUBLE = _amgx.get_amgx_allgather_double_handler()

ffi.register_ffi_target(_AMGX_CALL_NAME, _AMGX_HANDLER, platform="CUDA")
ffi.register_ffi_target(_AMGX_CALL_NAME_DOUBLE, _AMGX_HANDLER_DOUBLE, platform="CUDA")
ffi.register_ffi_target(_AMGX_CALL_NAME_MPI, _AMGX_HANDLER_MPI, platform="CUDA")
ffi.register_ffi_target(
    _AMGX_CALL_NAME_MPI_DOUBLE, _AMGX_HANDLER_MPI_DOUBLE, platform="CUDA"
)

ffi.register_ffi_target(
    _AMGX_CALL_NAME_ALLGATHER, _AMGX_HANDLER_ALLGATHER, platform="CUDA"
)
ffi.register_ffi_target(
    _AMGX_CALL_NAME_ALLGATHER_DOUBLE,
    _AMGX_HANDLER_ALLGATHER_DOUBLE,
    platform="CUDA",
)


class AMGXStatus(IntEnum):
    SUCCESS = 0
    FAILED = 1
    DIVERGED = 2
    NOT_CONVERGED = 3

    def __repr__(self):
        return f"<{self.__class__.__name__}.{self.name}: {self.value}>"

    def __str__(self):
        return f"{self.__class__.__name__}.{self.name}"


def _amgx_allgather_impl(
    sendbuf: ArrayLike,
    recvcounts: ArrayLike,
    displs: ArrayLike,
    comm_ptr: ArrayLike,
    nglobal: ArrayLike | None = None,
) -> jax.Array:
    """MPI AllGatherv implementation via FFI."""

    sendbuf = jnp.asarray(sendbuf)

    if nglobal is None:
        nglobal = jnp.sum(recvcounts)

    out_spec = jax.ShapeDtypeStruct((nglobal,), sendbuf.dtype)

    call_name = (
        _AMGX_CALL_NAME_ALLGATHER_DOUBLE
        if sendbuf.dtype == jnp.float64
        else _AMGX_CALL_NAME_ALLGATHER
    )

    call = ffi.ffi_call(
        call_name,
        out_spec,
        input_layouts=[None, None, None, None],
        output_layouts=None,
    )
    return call(sendbuf, recvcounts, displs, comm_ptr)


def _mpi4jax_allgatherv(
    sendbuf: jax.Array,
    recvcounts_tuple: tuple[int, ...],
    comm: Any,
) -> jax.Array:
    """
    Allgatherv implementation using mpi4jax (GPU-direct communication).

    Since mpi4jax only has allgather (not allgatherv), we:
    1. Pad local array to max_count
    2. Use allgather
    3. Extract and concatenate the valid portions
    """
    import mpi4jax

    max_count = max(recvcounts_tuple)
    nranks = len(recvcounts_tuple)
    nglobal = sum(recvcounts_tuple)

    # Pad sendbuf to max_count (known at trace time)
    padded = jnp.zeros(max_count, dtype=sendbuf.dtype)
    n_local = recvcounts_tuple[comm.Get_rank()]  # Static value
    padded = padded.at[:n_local].set(sendbuf)

    # Allgather padded arrays (stays on GPU)
    gathered = mpi4jax.allgather(padded, comm=comm)
    # gathered shape: (nranks, max_count)

    # Extract valid portions using static slicing (recvcounts are trace-time constants)
    result_parts = []
    for r in range(nranks):
        count = recvcounts_tuple[r]  # Static
        result_parts.append(gathered[r, :count])

    return jnp.concatenate(result_parts)


def _mpi4jax_alltoallv_transpose(
    data: jax.Array,
    indices: jax.Array,
    indptr: jax.Array,
    recvcounts_tuple: tuple[int, ...],
    comm: Any,
    max_nnz: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Pure JAX implementation of distributed matrix transpose using mpi4jax.

    This version uses only JAX operations and mpi4jax collectives,
    making it fully JIT-compatible. All operations stay on GPU.

    The algorithm:
    1. Convert local CSR to COO format
    2. Determine destination rank for each element (based on column -> row mapping)
    3. Exchange element counts via alltoall
    4. Build padded send buffers using scatter operations
    5. Exchange data via GPU-direct alltoall
    6. Extract valid data and rebuild CSR for A^T

    Args:
        data: CSR data array
        indices: CSR column indices
        indptr: CSR row pointers
        recvcounts_tuple: Partition sizes (rows per rank)
        comm: MPI communicator
        max_nnz: Maximum nnz across all ranks for buffer sizing (must be precalculated)
    """
    import mpi4jax
    from mpi4py import MPI

    rank = comm.Get_rank()
    size = comm.Get_size()
    nnz = data.shape[0]

    # Use static Python values from recvcounts_tuple (known at trace time)
    # This avoids traced array issues with jnp.arange
    n_local = recvcounts_tuple[rank]  # Python int, not traced
    my_row_start = sum(recvcounts_tuple[:rank])  # Python int, not traced

    # Compute partition info as JAX arrays for operations that need them
    r_counts = jnp.array(recvcounts_tuple, dtype=jnp.int32)
    displs = jnp.concatenate(
        [jnp.array([0], dtype=jnp.int64), jnp.cumsum(r_counts[:-1]).astype(jnp.int64)]
    )

    # --- Step 1: Convert CSR to COO format ---
    row_counts = indptr[1:] - indptr[:-1]
    # Use static n_local for jnp.arange
    row_indices_local = jnp.repeat(
        jnp.arange(n_local, dtype=indices.dtype), row_counts, total_repeat_length=nnz
    )
    row_indices_global = row_indices_local + my_row_start
    col_indices_global = indices

    # --- Step 2: Determine destination ranks ---
    dest_ranks = jnp.searchsorted(displs, col_indices_global, side="right") - 1
    dest_ranks = jnp.clip(dest_ranks, 0, size - 1).astype(jnp.int32)

    # --- Step 3: Sort by destination rank ---
    sort_order = jnp.argsort(dest_ranks)
    data_sorted = data[sort_order]
    rows_sorted = row_indices_global[sort_order]
    cols_sorted = col_indices_global[sort_order]
    dest_ranks_sorted = dest_ranks[sort_order]

    # --- Step 4: Count elements per destination ---
    # Use one_hot + sum instead of bincount (not available in JAX)
    one_hot = jax.nn.one_hot(dest_ranks_sorted, size, dtype=jnp.int32)
    send_counts = jnp.sum(one_hot, axis=0)

    # --- Step 5: Exchange counts via mpi4jax ---
    # Reshape for alltoall: each rank sends its count to each other rank
    recv_counts = mpi4jax.alltoall(send_counts.reshape(size, 1), comm=comm).flatten()

    # Compute global max for padding via allreduce
    local_max = jnp.maximum(jnp.max(send_counts), jnp.max(recv_counts))
    local_max = jnp.maximum(local_max, 1)  # At least 1 to avoid empty arrays
    global_max_arr = mpi4jax.allreduce(local_max, op=MPI.MAX, comm=comm)
    # global_max is now a JAX array; use it directly (traced value)

    # --- Step 6: Build padded send buffers using scatter ---
    # Compute position of each element within its destination's buffer
    send_displs = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(send_counts[:-1]).astype(jnp.int32),
        ]
    )

    # For each element i, its position within dest buffer is:
    # (index within sorted array) - (start index for its destination)
    # We compute this using cumsum per destination
    dest_mask = dest_ranks_sorted[:, None] == jnp.arange(size)  # (nnz, size)
    cumsum_per_dest = jnp.cumsum(
        dest_mask.astype(jnp.int32), axis=0
    )  # Running count per dest
    positions = (
        jnp.sum(cumsum_per_dest * dest_mask.astype(jnp.int32), axis=1) - 1
    )  # Position for each element

    # Use precalculated max_nnz for buffer sizing
    # This ensures all ranks use the same buffer size for alltoall
    max_per_rank = max_nnz

    # Initialize send buffers with zeros
    send_data = jnp.zeros((size, max_per_rank), dtype=data.dtype)
    send_rows = jnp.zeros((size, max_per_rank), dtype=indices.dtype)
    send_cols = jnp.zeros((size, max_per_rank), dtype=indices.dtype)

    # Scatter data into send buffers
    # 2D indexing: send_data[dest_ranks_sorted[i], positions[i]] = data_sorted[i]
    send_data = send_data.at[dest_ranks_sorted, positions].set(data_sorted)
    send_rows = send_rows.at[dest_ranks_sorted, positions].set(rows_sorted)
    send_cols = send_cols.at[dest_ranks_sorted, positions].set(cols_sorted)

    # --- Step 7: GPU-direct alltoall via mpi4jax ---
    recv_data = mpi4jax.alltoall(send_data, comm=comm)
    recv_rows = mpi4jax.alltoall(send_rows, comm=comm)
    recv_cols = mpi4jax.alltoall(send_cols, comm=comm)

    # --- Step 8: Extract valid received data ---
    # Flatten and concatenate valid portions from each rank
    recv_displs = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(recv_counts[:-1]).astype(jnp.int32),
        ]
    )
    total_recv = jnp.sum(recv_counts)

    # Create index arrays for gathering valid data
    # For each rank r, we take recv_data[r, 0:recv_counts[r]]
    # We'll use a masked approach that works with JIT

    # Build flat indices: for rank r, positions 0..recv_counts[r]-1 are valid
    # Create a mask for valid positions
    rank_indices = jnp.repeat(jnp.arange(size, dtype=jnp.int32), max_per_rank)
    pos_indices = jnp.tile(jnp.arange(max_per_rank, dtype=jnp.int32), size)
    valid_mask = pos_indices < recv_counts[rank_indices]

    # Flatten recv arrays
    recv_data_flat_all = recv_data.flatten()
    recv_rows_flat_all = recv_rows.flatten()
    recv_cols_flat_all = recv_cols.flatten()

    # Extract valid elements using boolean indexing equivalent
    # Since we can't use dynamic boolean indexing in JIT, use where + scatter
    # Compute destination indices in the output array
    recv_counts_expanded = recv_counts[rank_indices]
    within_rank_pos = pos_indices
    # Cumsum of recv_counts gives starting position for each rank in output
    # Position in output = recv_displs[rank] + within_rank_pos (if valid)

    # Create output position for each element (invalid elements get -1 or beyond)
    flat_output_pos = jnp.where(
        valid_mask,
        recv_displs[rank_indices] + within_rank_pos,
        -1,  # Invalid positions (will be ignored)
    )

    # Initialize output arrays with size = nnz (same as input for shape compatibility)
    recv_data_flat = jnp.zeros(nnz, dtype=data.dtype)
    recv_rows_flat = jnp.zeros(nnz, dtype=indices.dtype)
    recv_cols_flat = jnp.zeros(nnz, dtype=indices.dtype)

    # Scatter valid elements to their positions
    # Use segment_sum pattern: only positions >= 0 are valid
    valid_positions = jnp.maximum(flat_output_pos, 0).astype(jnp.int32)
    scatter_mask = flat_output_pos >= 0

    # Use jnp.where to mask values before scatter (preserves dtype)
    masked_data = jnp.where(
        scatter_mask, recv_data_flat_all, jnp.zeros_like(recv_data_flat_all)
    )
    masked_rows = jnp.where(
        scatter_mask, recv_rows_flat_all, jnp.zeros_like(recv_rows_flat_all)
    )
    masked_cols = jnp.where(
        scatter_mask, recv_cols_flat_all, jnp.zeros_like(recv_cols_flat_all)
    )

    # Use at[].add to scatter (avoids overwrite issues)
    recv_data_flat = recv_data_flat.at[valid_positions].add(masked_data)
    recv_rows_flat = recv_rows_flat.at[valid_positions].add(masked_rows)
    recv_cols_flat = recv_cols_flat.at[valid_positions].add(masked_cols)

    # --- Step 9: Build A^T in CSR format ---
    # For A^T: recv_cols becomes local row (was column in A), recv_rows becomes col (was row in A)
    at_rows_local = recv_cols_flat - my_row_start  # Local row index in A^T
    at_cols = recv_rows_flat  # Column index in A^T (global row in A)

    # Sort by (row, col) for CSR format
    # Create composite key for sorting: row * n_global + col
    n_global = sum(recvcounts_tuple)  # Static Python int
    sort_key = at_rows_local * n_global + at_cols
    sort_idx = jnp.argsort(sort_key)

    r_sorted = at_rows_local[sort_idx]
    c_sorted = at_cols[sort_idx]
    v_sorted = recv_data_flat[sort_idx]

    # Build indptr using segment_sum equivalent
    # Count elements per row
    row_one_hot = jax.nn.one_hot(r_sorted, n_local, dtype=jnp.int32)
    row_counts_at = jnp.sum(row_one_hot, axis=0)

    out_indptr = jnp.zeros(n_local + 1, dtype=indptr.dtype)
    out_indptr = out_indptr.at[1:].set(jnp.cumsum(row_counts_at).astype(jnp.int32))

    # Output data and indices (already sorted)
    out_data = v_sorted
    out_indices = c_sorted

    # Ensure output has same nnz as input for shape compatibility
    # Pad or truncate to match original nnz
    target_nnz = nnz
    actual_nnz = out_data.shape[0]

    # Pad if needed (transpose may have different nnz, but for square matrices it's the same)
    out_data_padded = jnp.zeros(target_nnz, dtype=data.dtype)
    out_indices_padded = jnp.zeros(target_nnz, dtype=indices.dtype)
    out_data_padded = out_data_padded.at[:actual_nnz].set(out_data[:target_nnz])
    out_indices_padded = out_indices_padded.at[:actual_nnz].set(
        out_indices[:target_nnz]
    )
    out_indptr = out_indptr.at[-1].set(target_nnz)

    return out_data_padded, out_indices_padded, out_indptr


def _amgx_solve_impl(
    row_ptrs: ArrayLike,
    col_indices: ArrayLike,
    values: ArrayLike,
    b: ArrayLike,
    config_str: str = "",
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
    )
    results = call(row_ptrs, col_indices, values, b, config=config_str)

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
    )
    results = call(
        row_ptrs, col_indices, values, b, nglobal, comm_ptr, lrank, config=config_str
    )

    return cast(tuple, results)


@functools.lru_cache(maxsize=32)
def _get_solver_primitive(config_str: str, is_symmetric: bool = False) -> Callable:
    """
    Returns a JAX custom_vjp primitive for AmgX solve with a specific configuration.
    Cached to avoid recompilation for identical configurations.
    """

    @jax.custom_vjp
    def solve(A: jsp.BCSR, b: jax.Array) -> tuple[jax.Array, jax.Array]:
        x, info = _amgx_solve_impl(
            A.indptr, A.indices, A.data, b, config_str=config_str
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
        solver = _get_solver_primitive(config_str, is_symmetric)

        # Check if matrix is symmetric
        if is_symmetric:
            adj_b, _ = solver(A, g_x)
        else:
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
    import warnings

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
            at_data, at_indices, at_indptr = _mpi4jax_alltoallv_transpose(
                A.data, A.indices, A.indptr, recvcounts_tuple, comm, max_nnz
            )

            # Reconstruct BCSR for A^T
            A_T = jsp.BCSR((at_data, at_indices, at_indptr), shape=A.shape)

            (adj_b, _), _ = solve(A_T, g_x, recvcounts, displs)

        # Gather x across all ranks for gradient computation
        x_global = allgather(x, recvcounts, displs, comm_ptr_arr)

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


def amg_solve(
    A: MatrixOrOperator,
    b: ArrayLike,
    config: dict | None = None,
    comm: "Comm | None" = None,
    nglobal: int | None = None,
    partition_info: tuple[int, int] | None = None,
    **kwargs: Any,
) -> tuple[jax.Array, dict]:
    """
    Solve Ax=b using AmgX (differentiable).

    Single-GPU mode (default):
        A: Matrix or callable operator A(x).
           Callables are automatically materialized to CSR.
        b: RHS vector (float32 or float64).
        config: Dict of AmgX configuration parameters.
        **kwargs: Additional configuration parameters passed as keyword arguments.
                  These override config if present.

    MPI mode (when comm is provided):
        A: Local portion of matrix with GLOBAL column indices (CSR).
        b: Local portion of RHS vector.
        comm: MPI communicator (from mpi4py.MPI.COMM_WORLD).
        nglobal: Global size of the matrix (total number of rows across all ranks).
        partition_info: Tuple (row_start, row_end) indicating which rows this rank owns.
        config: Dict or string of AmgX configuration parameters.
        **kwargs: Additional configuration parameters.

    If A is attached with MPI cache (via `with_cache`), then comm, nglobal, and partition_info are not needed.

    Returns:
        x: Solution vector (float32 or float64). In MPI mode, returns local portion.
        info: Dictionary containing 'iterations', 'residual', and 'status'.
    """

    b = jnp.asarray(b)

    # Check for GPU backend
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            f"AMGX requires a GPU backend, but JAX is using '{jax.default_backend()}'. "
            "Please ensure you have a CUDA-enabled GPU and JAX is installed with CUDA support."
        )

    # Check if MPI cache is attached to A (via with_cache)
    mpi_cache = getattr(A, "_mpi_cache", None)

    # Prepare configuration string/file (skip if using mpi_cache which already has config_str)
    if mpi_cache is not None:
        config_str = mpi_cache["config_str"]
    else:
        config_str = amgx_config.prepare_config(config, **kwargs)

    # Detect desired precision
    target_dtype = get_preferred_dtype(A, b)
    if target_dtype == jnp.float64 and b.dtype != jnp.float64:
        b = b.astype(jnp.float64)

    # Check for symmetry attribute on A
    is_symmetric = getattr(A, "_is_symmetric", False)

    # Branch: MPI mode or single-GPU mode
    if mpi_cache is not None or comm is not None:
        # MPI MODE
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
            gpu_count = jax.device_count()
            lrank = rank % gpu_count

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
            local_nnz = len(A_csr.data)
            all_nnz = comm.allgather(local_nnz)
            max_nnz = max(all_nnz)

            solver = _get_solver_primitive_mpi(
                config_str,
                nglobal,
                comm_ptr,
                lrank,
                is_symmetric=is_symmetric,
                recvcounts_tuple=recvcounts_tuple,
                max_nnz=max_nnz,
            )
            (x, info), _ = solver(A_csr, b, recvcounts, displs)

    else:
        # Single-GPU mode: use int32 indices
        A_csr = to_bcsr_matrix(A, b)

        # Get cached primitive for this configuration
        solver = _get_solver_primitive(config_str, is_symmetric=is_symmetric)

        x, info = solver(A_csr, b)

    # Convert JAX array stats to python dict (same for both modes)
    try:
        iter_val = int(info[0])
        res_val = float(info[1])
        status_val = AMGXStatus(int(info[2]))
    except Exception:
        # Inside JIT or symbolic execution: return raw arrays/tracers
        iter_val = info[0]
        res_val = info[1]
        status_val = info[2]

    info = {"iterations": iter_val, "residual": res_val, "status": status_val}
    return x, info
