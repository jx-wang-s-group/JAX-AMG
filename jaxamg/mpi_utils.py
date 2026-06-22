"""MPI utilities for distributed AmgX solving."""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import jax
import jax.experimental.sparse as jsp
import jax.ffi as ffi
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.typing import ArrayLike

from . import _amgx
from .utils import temp_enable_x64

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

_AMGX_CALL_NAME_ALLGATHER = "amgx_allgather"
_AMGX_CALL_NAME_ALLGATHER_DOUBLE = "amgx_allgather_double"

_AMGX_HANDLER_ALLGATHER = _amgx.get_amgx_allgather_handler()
_AMGX_HANDLER_ALLGATHER_DOUBLE = _amgx.get_amgx_allgather_double_handler()

ffi.register_ffi_target(
    _AMGX_CALL_NAME_ALLGATHER, _AMGX_HANDLER_ALLGATHER, platform="CUDA"
)
ffi.register_ffi_target(
    _AMGX_CALL_NAME_ALLGATHER_DOUBLE,
    _AMGX_HANDLER_ALLGATHER_DOUBLE,
    platform="CUDA",
)


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
    comm: "Comm",
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
    comm: "Comm",
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
        [jnp.array([0], dtype=jnp.int32), jnp.cumsum(r_counts[:-1]).astype(jnp.int32)]
    )

    # --- Step 1: Convert CSR to COO format ---
    row_counts = indptr[1:] - indptr[:-1]
    # Use static n_local for jnp.arange
    row_indices_local = jnp.repeat(
        jnp.arange(n_local, dtype=jnp.int32), row_counts, total_repeat_length=nnz
    )
    row_indices_global = row_indices_local + my_row_start
    col_indices_global = indices.astype(jnp.int32)

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
    # Use bincount (much lighter than one_hot for large nnz)
    send_counts = jnp.bincount(dest_ranks_sorted, length=size).astype(jnp.int32)

    # --- Step 5: Exchange counts and compute buffer sizes ---
    recv_counts = mpi4jax.alltoall(send_counts, comm=comm)

    # --- Step 6: Build padded send buffers ---
    # Since dest_ranks_sorted is sorted by destination rank, positions can be
    # computed via segment offsets (avoids O(nnz*nranks) one-hot masks).
    send_displs = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(send_counts[:-1]).astype(jnp.int32),
        ]
    )
    positions = jnp.arange(nnz, dtype=jnp.int32) - send_displs[dest_ranks_sorted]

    # Use precalculated max_nnz for buffer sizing
    # This ensures all ranks use the same buffer size for alltoall
    max_per_rank = max_nnz

    # Initialize send buffers with zeros
    send_data = jnp.zeros((size, max_per_rank), dtype=data.dtype)
    send_rows = jnp.zeros((size, max_per_rank), dtype=jnp.int32)
    send_cols = jnp.zeros((size, max_per_rank), dtype=jnp.int32)

    # Scatter data into send buffers
    # 2D indexing: send_data[dest_ranks_sorted[i], positions[i]] = data_sorted[i]
    send_data = send_data.at[dest_ranks_sorted, positions].set(data_sorted)
    send_rows = send_rows.at[dest_ranks_sorted, positions].set(
        rows_sorted.astype(jnp.int32)
    )
    send_cols = send_cols.at[dest_ranks_sorted, positions].set(
        cols_sorted.astype(jnp.int32)
    )

    # --- Step 7: Exchange data via GPU-direct alltoall ---
    recv_data = mpi4jax.alltoall(send_data, comm=comm)
    recv_rows = mpi4jax.alltoall(send_rows, comm=comm)
    recv_cols = mpi4jax.alltoall(send_cols, comm=comm)

    # --- Step 8: Extract valid received data and build A^T ---
    # Flatten and concatenate valid portions from each rank
    recv_displs = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(recv_counts[:-1]).astype(jnp.int32),
        ]
    )

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
    recv_rows_flat = jnp.zeros(nnz, dtype=jnp.int32)
    recv_cols_flat = jnp.zeros(nnz, dtype=jnp.int32)

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

    # Build indptr from row counts
    row_counts_at = jnp.bincount(r_sorted, length=n_local).astype(jnp.int32)

    out_indptr = jnp.zeros(n_local + 1, dtype=indptr.dtype)
    out_indptr = out_indptr.at[1:].set(jnp.cumsum(row_counts_at).astype(jnp.int32))

    # Output data and indices (already sorted). For square matrices with
    # fixed sparsity, transpose preserves nnz, so no padding/truncation needed.
    out_data = v_sorted
    out_indices = c_sorted
    out_indptr = out_indptr.at[-1].set(nnz)

    # Convert output indices to int64
    with temp_enable_x64():
        out_indices_int64 = out_indices.astype(jnp.int64)

    return out_data, out_indices_int64, out_indptr


def partition_csr_matrix(
    A_global: jsp.BCSR | sp.csr_matrix, rank: int, nranks: int
) -> tuple[jsp.BCSR, int, int]:
    """Partition global CSR matrix across MPI ranks (row-based).

    Args:
        A_global: Global CSR matrix (SciPy sparse or JAX BCSR)
        rank: MPI rank (0-indexed)
        nranks: Total number of MPI ranks

    Returns:
        A_local: Local BCSR matrix partition (JAX)
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)

    Note:
        Preserves input dtype (float32/float64). Avoids unnecessary conversions
        by using matrix attributes directly.
    """
    is_scipy = hasattr(A_global, "todense")

    if hasattr(A_global, "indptr"):
        indptr, indices, data = A_global.indptr, A_global.indices, A_global.data
        n = A_global.shape[0]
    else:
        raise ValueError(f"Unsupported matrix type: {type(A_global)}")

    # Row-based partitioning
    row_start, row_end, n_local = get_partition_info(n, rank, nranks)

    # Extract local partition
    nnz_start = indptr[row_start]
    nnz_end = indptr[row_end]

    # Create BCSR: convert to JAX if SciPy, or ensure int32 indices if already JAX
    local_indptr = jnp.asarray(indptr[row_start : row_end + 1] - nnz_start)
    local_indices = jnp.asarray(indices[nnz_start:nnz_end])
    local_data = jnp.asarray(data[nnz_start:nnz_end])

    if not is_scipy:
        local_indices = local_indices.astype(jnp.int32)
        local_indptr = local_indptr.astype(jnp.int32)

    A_local = jsp.BCSR((local_data, local_indices, local_indptr), shape=(n_local, n))
    return A_local, row_start, row_end


def validate_partition(
    A_local: jsp.BCSR, nglobal: int, row_start: int, row_end: int
) -> None:
    """Validate partitioned matrix structure and print diagnostics."""
    n_local = row_end - row_start

    assert (
        A_local.shape[0] == n_local
    ), f"Row count mismatch: {A_local.shape[0]} != {n_local}"
    assert (
        A_local.shape[1] == nglobal
    ), f"Column count mismatch: {A_local.shape[1]} != {nglobal}"
    assert (
        A_local.indptr[0] == 0
    ), f"First row pointer should be 0, got {A_local.indptr[0]}"
    assert A_local.indptr[-1] == len(
        A_local.data
    ), f"Last row pointer mismatch: {A_local.indptr[-1]} != {len(A_local.data)}"

    if len(A_local.indices) > 0:
        max_col = jnp.max(A_local.indices)
        min_col = jnp.min(A_local.indices)
        assert (
            max_col < nglobal
        ), f"Column index {max_col} exceeds global size {nglobal}"
        assert min_col >= 0, f"Column index {min_col} is negative"
        print(f"✓ Partition validated: {n_local} rows, cols [{min_col}, {max_col}]")
    else:
        print(f"✓ Partition validated: {n_local} rows, no non-zeros")


def partition_vector(
    b_global: ArrayLike, rank: int, nranks: int
) -> tuple[ArrayLike, int, int]:
    """Partition global vector across MPI ranks (row-based).

    Args:
        b_global: Global vector
        rank: MPI rank (0-indexed)
        nranks: Total number of MPI ranks

    Returns:
        b_local: Local vector partition
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)
    """
    b_global = jnp.asarray(b_global)
    n = len(b_global)
    row_start, row_end, _ = get_partition_info(n, rank, nranks)
    return b_global[row_start:row_end], row_start, row_end


def gather_vector(x_local: ArrayLike, comm: "Comm", root: int = 0) -> ArrayLike | None:
    """Gather a row-partitioned vector to the root rank using MPI Gatherv.

    Args:
        x_local: This rank's local segment of the distributed vector
        comm: MPI communicator
        root: Root rank to gather to (default: 0)

    Returns:
        JAX array of the assembled global vector (root rank only), None otherwise
    """
    from mpi4py import MPI

    rank = comm.Get_rank()
    x_local_np = np.array(x_local, dtype=np.float64)
    n_local = len(x_local_np)
    all_sizes = comm.gather(n_local, root=root)
    all_sizes = cast(list, all_sizes)

    if rank == root:
        n_global = sum(all_sizes)
        x_global = np.zeros(n_global, dtype=x_local_np.dtype)
        displacements = [0] + list(np.cumsum(all_sizes[:-1]))
        mpi_type = MPI.DOUBLE if x_local_np.dtype == np.float64 else MPI.FLOAT
        comm.Gatherv(
            x_local_np, [x_global, all_sizes, displacements, mpi_type], root=root
        )
        return jnp.array(x_global)
    else:
        comm.Gatherv(x_local_np, None, root=root)
        return None


def make_allgather_vector(
    comm: "Comm",
    partition_info: tuple[int, int],
    nglobal: int,
    *,
    backend: str = "auto",
) -> Callable[[jax.Array], jax.Array]:
    """Build a differentiable MPI all-gather of a row-partitioned vector.

    Returns a callable ``allgather(x_local) -> x_global`` that assembles every
    rank's local segment into the full length-``nglobal`` vector **on every
    rank**, and is differentiable under ``jax.grad`` / ``jax.vjp``.

    Forward:  ``Allgatherv`` (collective).  Backward: each rank receives the
    slice of the incoming global cotangent that corresponds to its own rows
    (``g_global[row_start:row_end]``) -- the exact adjoint of the gather.

    Unlike :func:`gather_vector` (root-only ``Gatherv``, not differentiable),
    this returns the assembled vector on *all* ranks and participates in
    automatic differentiation, which is what makes it usable inside a
    distributed loss.  Use it when the loss is defined on the global solution
    (global normalization, cross-rank coupling, an inner product against a
    dense global vector, ...).  A loss that is separable across the row
    partition (e.g. a plain sum of per-row squared errors) does not need it:
    differentiate the local loss and sum the scalar gradients across ranks.

    Gradient contract:
        The result is replicated across ranks.  When the downstream loss is
        evaluated **identically and redundantly on every rank** from
        ``x_global`` (the usual distributed-optimization pattern), the VJP
        returns each rank's *local contribution* to the gradient of a
        replicated parameter.  To recover the full gradient, sum the per-rank
        parameter gradients yourself (e.g. ``comm.allreduce(g, op=MPI.SUM)``).
        This primitive deliberately performs no such reduction so that it
        remains a pure linear operator.

    Args:
        comm: MPI communicator.
        partition_info: ``(row_start, row_end)`` -- the global rows owned by
            this rank (half-open interval).
        nglobal: Length of the assembled global vector.
        backend: ``"auto"`` (default) uses the GPU-direct mpi4jax all-gather
            when mpi4jax is importable and otherwise falls back to a host
            (``pure_callback``) all-gather; ``"mpi4jax"`` or ``"host"`` force a
            specific backend.

    Returns:
        A differentiable callable ``allgather(x_local) -> x_global``.
    """
    import importlib.util

    row_start, row_end = partition_info
    n_local = row_end - row_start

    # Static communication layout, computed once and captured by the closure.
    all_sizes = comm.allgather(n_local)
    recvcounts_tuple = tuple(int(s) for s in all_sizes)
    recvcounts_np = np.array(recvcounts_tuple, dtype=np.int32)
    displacements = np.insert(np.cumsum(recvcounts_np[:-1]), 0, 0).astype(np.int32)

    total = int(recvcounts_np.sum())
    if total != int(nglobal):
        raise ValueError(
            f"Sum of local sizes ({total}) does not match nglobal ({nglobal})."
        )

    if backend == "auto":
        use_mpi4jax = importlib.util.find_spec("mpi4jax") is not None
    elif backend in ("mpi4jax", "host"):
        use_mpi4jax = backend == "mpi4jax"
    else:
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'auto', 'mpi4jax', or 'host'."
        )

    def _forward_mpi4jax(x_local: jax.Array) -> jax.Array:
        return _mpi4jax_allgatherv(x_local, recvcounts_tuple, comm)

    def _forward_host(x_local: jax.Array) -> jax.Array:
        from mpi4py import MPI

        # Resolve dtype at trace time so both float32 and float64 are supported.
        np_dtype = np.float32 if x_local.dtype == jnp.float32 else np.float64
        mpi_dtype = MPI.FLOAT if np_dtype == np.float32 else MPI.DOUBLE

        def _allgatherv(x_np: np.ndarray) -> np.ndarray:
            # pure_callback may hand back an array with a byte-order prefix
            # (e.g. '=f8'); ascontiguousarray with an explicit dtype strips it
            # so mpi4py can resolve the buffer type.
            x_np = np.ascontiguousarray(x_np, dtype=np_dtype)
            recvbuf = np.empty(nglobal, dtype=np_dtype)
            comm.Allgatherv(x_np, [recvbuf, recvcounts_np, displacements, mpi_dtype])
            return recvbuf

        result_shape = jax.ShapeDtypeStruct((nglobal,), x_local.dtype)
        return jax.pure_callback(_allgatherv, result_shape, x_local)

    _forward = _forward_mpi4jax if use_mpi4jax else _forward_host

    @jax.custom_vjp
    def allgather(x_local: jax.Array) -> jax.Array:
        return _forward(x_local)

    def _allgather_fwd(x_local: jax.Array) -> tuple[jax.Array, None]:
        # The gather is linear, so the backward pass needs no residuals.
        return allgather(x_local), None

    def _allgather_bwd(_: None, g_global: jax.Array) -> tuple[jax.Array]:
        # Adjoint of an all-gather: this rank keeps the segment of the global
        # cotangent that corresponds to its own rows.
        return (g_global[row_start:row_end],)

    allgather.defvjp(_allgather_fwd, _allgather_bwd)
    return allgather


def get_partition_info(n_global: int, rank: int, nranks: int) -> tuple[int, int, int]:
    """Compute partition information for distributed problem."""
    local_size = n_global // nranks
    remainder = n_global % nranks

    if rank < remainder:
        n_local = local_size + 1
        row_start = rank * n_local
    else:
        n_local = local_size
        row_start = rank * local_size + remainder

    row_end = row_start + n_local

    return row_start, row_end, n_local
