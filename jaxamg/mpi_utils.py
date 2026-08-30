"""MPI utilities for distributed AmgX solving."""

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from mpi4py.MPI import Comm


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


# mpi4py communicators are unhashable, so the differentiable MPI primitive is
# keyed on the communicator's integer address. This registry maps that address
# back to the live communicator so the backward pass can run its collectives on
# the user's communicator (possibly a subcommunicator), not MPI.COMM_WORLD.
_COMM_BY_PTR: dict[int, "Comm"] = {}


def register_comm(comm: "Comm") -> int:
    """Record `comm` by its address and return that address (see _COMM_BY_PTR)."""
    from mpi4py import MPI

    ptr = MPI._addressof(comm)
    _COMM_BY_PTR[ptr] = comm
    return ptr


def resolve_comm(comm_ptr: int) -> "Comm":
    """Recover the communicator registered under `comm_ptr` (see register_comm),
    falling back to MPI.COMM_WORLD if it was never registered."""
    from mpi4py import MPI

    return _COMM_BY_PTR.get(comm_ptr, MPI.COMM_WORLD)


class TransposePlan(NamedTuple):
    """Fixed CSR structure and value routing for a distributed transpose."""

    indices: np.ndarray | jax.Array
    indptr: np.ndarray | jax.Array
    local_source_ids: np.ndarray | jax.Array
    local_target_ids: np.ndarray | jax.Array
    send_ids_2d: np.ndarray | jax.Array
    recv_target_ids_2d: np.ndarray | jax.Array
    max_nnz: int

    @property
    def nnz(self) -> int:
        return len(self.indices)


def build_transpose_plan(
    indices: ArrayLike,
    indptr: ArrayLike,
    recvcounts: tuple[int, ...],
    partition_info: tuple[int, int],
    comm: "Comm",
) -> TransposePlan:
    """Precompute the structure and value routing for ``A.T``."""
    from mpi4py import MPI

    indices = np.asarray(indices, dtype=np.int64)
    indptr = np.asarray(indptr, dtype=np.int64)
    row_start, row_end = partition_info
    n_local = row_end - row_start
    n_global = sum(recvcounts)
    nranks = len(recvcounts)

    source_rows = np.repeat(
        np.arange(row_start, row_end, dtype=np.int64), np.diff(indptr)
    )
    invalid_columns = bool(np.any(indices < 0) or np.any(indices >= n_global))
    if comm.allreduce(invalid_columns, op=MPI.LOR):
        raise ValueError("A_local contains a global column index outside the matrix")

    row_bounds = np.cumsum(np.array([0, *recvcounts], dtype=np.int64))
    owners = np.searchsorted(row_bounds, indices, side="right") - 1
    send_order = np.argsort(owners, kind="stable")
    send_counts = np.bincount(owners, minlength=nranks).astype(np.int32)
    recv_counts = np.empty(nranks, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)
    send_displs = np.insert(np.cumsum(send_counts[:-1]), 0, 0).astype(np.int32)
    recv_displs = np.insert(np.cumsum(recv_counts[:-1]), 0, 0).astype(np.int32)

    send_coordinates = np.ascontiguousarray(
        np.column_stack((indices[send_order], source_rows[send_order]))
    )
    send_ids = np.ascontiguousarray(np.arange(len(indices), dtype=np.int64)[send_order])
    recv_nnz = int(recv_counts.sum())
    recv_coordinates = np.empty((recv_nnz, 2), dtype=np.int64)
    comm.Alltoallv(
        [send_coordinates, 2 * send_counts, 2 * send_displs, MPI.INT64_T],
        [recv_coordinates, 2 * recv_counts, 2 * recv_displs, MPI.INT64_T],
    )
    recv_rows = recv_coordinates[:, 0]
    recv_cols = recv_coordinates[:, 1]
    local_rows = recv_rows - row_start
    order = np.lexsort((recv_cols, local_rows))
    local_rows = local_rows[order]
    recv_cols = recv_cols[order]
    row_counts = np.bincount(local_rows, minlength=n_local)
    transpose_indptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int32)

    rank = comm.Get_rank()
    remote_send_counts = send_counts.copy()
    remote_recv_counts = recv_counts.copy()
    remote_send_counts[rank] = 0
    remote_recv_counts[rank] = 0
    local_maxima = np.array(
        [max(remote_send_counts.max(), remote_recv_counts.max()), recv_nnz],
        dtype=np.int64,
    )
    global_maxima = np.empty_like(local_maxima)
    comm.Allreduce(local_maxima, global_maxima, op=MPI.MAX)
    max_per_rank = max(int(global_maxima[0]), 1)
    max_nnz = int(global_maxima[1])

    # Padding uses a one-past-the-end sentinel. Applying the plan pads both the
    # input and output by one zero, so reversing the four routing arrays also
    # gives the value routing for ``A.T.T``.
    send_ids_2d = np.full((nranks, max_per_rank), len(indices), dtype=np.int32)
    recv_target_ids_2d = np.full((nranks, max_per_rank), recv_nnz, dtype=np.int32)
    inverse_order = np.empty(recv_nnz, dtype=np.int32)
    inverse_order[order] = np.arange(recv_nnz, dtype=np.int32)
    for peer in range(nranks):
        if peer == rank:
            continue
        send_count = int(send_counts[peer])
        if send_count:
            start = int(send_displs[peer])
            send_ids_2d[peer, :send_count] = send_ids[start : start + send_count]
        recv_count = int(recv_counts[peer])
        if recv_count:
            start = int(recv_displs[peer])
            recv_target_ids_2d[peer, :recv_count] = inverse_order[
                start : start + recv_count
            ]

    local_send_start = int(send_displs[rank])
    local_recv_start = int(recv_displs[rank])
    local_count = int(send_counts[rank])
    local_source_ids = send_ids[
        local_send_start : local_send_start + local_count
    ].astype(np.int32)
    local_target_ids = inverse_order[local_recv_start : local_recv_start + local_count]

    return TransposePlan(
        np.asarray(recv_cols, dtype=np.int64),
        transpose_indptr,
        local_source_ids,
        local_target_ids,
        send_ids_2d,
        recv_target_ids_2d,
        max_nnz,
    )


class HaloPlan(NamedTuple):
    """Static communication plan for the backward pass halo exchange.

    The gradient ``dL/dA_ij = -adj_b[i] * x[j]`` needs ``x[j]`` only for the
    global columns ``j`` this rank's local rows reference. Those split into
    locally owned columns (already in ``x_local``) and a small set of remote
    "ghost" columns owned by other ranks. This plan, built once from the fixed
    sparsity pattern, fetches only the ghost values instead of all-gathering the
    entire global solution.

    Fields (all static, captured at setup):
        n_local: Rows owned by this rank.
        n_ghost: Distinct remote columns this rank references.
        max_n_ghost: Largest ``n_ghost`` across ranks, used for equal shard
            shapes in the JAX sharding interface.
        max_per_rank: Padded per-rank chunk size for the ``alltoall`` (a global
            max, so every rank uses the same buffer size).
        col_to_combined: For each local nonzero, its index into the combined
            ``[x_local | x_ghost]`` vector (length nnz).
        send_ids_2d: Local ``x`` indices to send to each rank, padded
            ``(nranks, max_per_rank)``.
        recv_ghost_slot_2d: Ghost slot each received value fills, padded
            ``(nranks, max_per_rank)`` with ``n_ghost`` as an ignored sentinel.
    """

    n_local: int
    n_ghost: int
    max_n_ghost: int
    max_per_rank: int
    col_to_combined: np.ndarray | jax.Array
    send_ids_2d: np.ndarray | jax.Array
    recv_ghost_slot_2d: np.ndarray | jax.Array


def build_halo_plan(
    local_col_indices: ArrayLike,
    recvcounts_tuple: tuple[int, ...],
    partition_info: tuple[int, int],
    comm: "Comm",
) -> HaloPlan:
    """Build the backward-pass halo-exchange plan (see :class:`HaloPlan`).

    Determines which remote solution entries this rank needs for its local
    gradient and the reciprocal entries it must supply to other ranks, via two
    small host-side collectives (``Alltoall`` of counts, ``Alltoallv`` of the
    requested global indices). The sparsity pattern is fixed, so this runs once.
    """
    from mpi4py import MPI

    nranks = len(recvcounts_tuple)
    row_start, row_end = partition_info
    n_local = row_end - row_start
    row_bounds = np.cumsum(np.array([0, *recvcounts_tuple], dtype=np.int64))

    cols = np.asarray(local_col_indices).astype(np.int64)
    uniq = np.unique(cols)
    is_remote = (uniq < row_start) | (uniq >= row_end)
    ghost_global_ids = uniq[is_remote]  # sorted (np.unique is sorted)
    n_ghost = int(ghost_global_ids.size)

    # Owner rank of each ghost column, and how many ghosts this rank needs from
    # each owner (recv_counts). The reciprocal send_counts come from an Alltoall.
    ghost_owner = np.clip(
        np.searchsorted(row_bounds, ghost_global_ids, side="right") - 1, 0, nranks - 1
    ).astype(np.int32)
    recv_counts = np.bincount(ghost_owner, minlength=nranks).astype(np.int32)
    send_counts = np.empty(nranks, dtype=np.int32)
    comm.Alltoall(recv_counts, send_counts)

    recv_displs = np.insert(np.cumsum(recv_counts[:-1]), 0, 0).astype(np.int32)
    send_displs = np.insert(np.cumsum(send_counts[:-1]), 0, 0).astype(np.int32)

    # Group this rank's requests by owner (stable keeps ghost order within owner),
    # then tell each owner which global ids we want and learn which ids others
    # want from us.
    order = np.argsort(ghost_owner, kind="stable")
    ghost_ids_by_owner = ghost_global_ids[order].astype(np.int64)
    ghost_slot_by_owner = order.astype(np.int32)

    requested_ids = np.empty(int(send_counts.sum()), dtype=np.int64)
    comm.Alltoallv(
        [ghost_ids_by_owner, recv_counts, recv_displs, MPI.INT64_T],
        [requested_ids, send_counts, send_displs, MPI.INT64_T],
    )
    send_local_ids = (requested_ids - row_start).astype(np.int32)

    # The all-to-all chunk and sharded ghost-vector sizes must agree globally.
    # Reduce both maxima together so sharding does not need another collective.
    local_maxima = np.array(
        [max(send_counts.max(), recv_counts.max()), n_ghost], dtype=np.int64
    )
    global_maxima = np.empty_like(local_maxima)
    comm.Allreduce(local_maxima, global_maxima, op=MPI.MAX)
    max_per_rank = max(int(global_maxima[0]), 1)
    max_n_ghost = int(global_maxima[1])

    send_ids_2d = np.zeros((nranks, max_per_rank), dtype=np.int32)
    recv_ghost_slot_2d = np.full((nranks, max_per_rank), n_ghost, dtype=np.int32)
    for p in range(nranks):
        sc = int(send_counts[p])
        if sc:
            send_ids_2d[p, :sc] = send_local_ids[send_displs[p] : send_displs[p] + sc]
        rc = int(recv_counts[p])
        if rc:
            recv_ghost_slot_2d[p, :rc] = ghost_slot_by_owner[
                recv_displs[p] : recv_displs[p] + rc
            ]

    # Map each local nonzero to its slot in the combined [x_local | x_ghost].
    local_mask = (cols >= row_start) & (cols < row_end)
    ghost_pos = np.clip(np.searchsorted(ghost_global_ids, cols), 0, max(n_ghost - 1, 0))
    col_to_combined = np.where(
        local_mask, cols - row_start, n_local + ghost_pos
    ).astype(np.int32)

    return HaloPlan(
        n_local,
        n_ghost,
        max_n_ghost,
        max_per_rank,
        col_to_combined,
        send_ids_2d,
        recv_ghost_slot_2d,
    )


def _mpi4jax_halo_gather(
    x_local: jax.Array,
    send_ids: jax.Array,
    recv_ghost_slot: jax.Array,
    n_ghost: int,
    comm: "Comm",
) -> jax.Array:
    """Assemble ``[x_local | x_ghost]`` by exchanging only the needed remote
    solution entries (see :class:`HaloPlan` for the plan arrays). ``send_ids``
    and ``recv_ghost_slot`` are the padded ``(nranks, max_per_rank)`` plan
    arrays; ``n_ghost`` is a static ghost count. JIT-compatible, GPU-direct."""
    import mpi4jax

    send_buf = x_local[send_ids]  # (nranks, max_per_rank); padded slots gather x[0]
    recv_buf = mpi4jax.alltoall(send_buf, comm=comm)

    # Scatter valid received values into ghost slots; padded slots hit the
    # sentinel index n_ghost, which is sliced off.
    x_ghost = jnp.zeros(n_ghost + 1, dtype=x_local.dtype)
    x_ghost = x_ghost.at[recv_ghost_slot.reshape(-1)].set(recv_buf.reshape(-1))
    return jnp.concatenate([x_local, x_ghost[:n_ghost]])


def _apply_transpose_plan(
    data: jax.Array,
    local_source_ids: jax.Array,
    local_target_ids: jax.Array,
    send_ids: jax.Array,
    recv_target_ids: jax.Array,
    nnz_out: int,
    exchange: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Apply a fixed transpose plan using the supplied all-to-all operation."""
    data = jnp.pad(data, (0, 1))
    values = jnp.zeros(nnz_out + 1, dtype=data.dtype)
    values = values.at[local_target_ids].set(data[local_source_ids])
    received = exchange(data[send_ids])
    values = values.at[recv_target_ids.reshape(-1)].set(received.reshape(-1))
    return values[:-1]


def _mpi4jax_transpose_values(
    data: jax.Array,
    local_source_ids: jax.Array,
    local_target_ids: jax.Array,
    send_ids: jax.Array,
    recv_target_ids: jax.Array,
    nnz_out: int,
    comm: "Comm",
) -> jax.Array:
    """Exchange only matrix values for a preplanned distributed transpose."""
    import mpi4jax

    return _apply_transpose_plan(
        data,
        local_source_ids,
        local_target_ids,
        send_ids,
        recv_target_ids,
        nnz_out,
        lambda values: mpi4jax.alltoall(values, comm=comm),
    )


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
    is_scipy = sp.issparse(A_global)

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


def partition_operator(
    operator: Callable, nglobal: int, rank: int, nranks: int
) -> tuple[Callable, int, int]:
    """Partition a global matrix-free operator across MPI ranks (row-based).

    Wraps a global operator -- a callable mapping a length-``nglobal`` vector to
    the global result ``A @ x`` -- into this rank's row-local operator, which
    returns only the rows ``[row_start, row_end)`` this rank owns. That is the
    ``(n_local, nglobal)`` form the distributed solve expects, so a user can pass
    a single global operator instead of writing a distributed one by hand.

    No global matrix is formed: each rank materializes only its local block.

    Args:
        operator: Global operator ``A(x)`` mapping a length-``nglobal`` vector to
            a length-``nglobal`` result.
        nglobal: Global problem size (total rows across all ranks).
        rank: MPI rank (0-indexed).
        nranks: Total number of MPI ranks.

    Returns:
        local_operator: Callable mapping the global vector to this rank's rows.
        row_start: Starting row index (global).
        row_end: Ending row index (global, exclusive).
    """
    row_start, row_end, _ = get_partition_info(nglobal, rank, nranks)

    def local_operator(x_global: ArrayLike) -> jax.Array:
        return operator(x_global)[row_start:row_end]

    return local_operator, row_start, row_end


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
    # Preserve float32/float64; promote any other dtype to float64.
    x_local_np = np.ascontiguousarray(x_local)
    if x_local_np.dtype not in (np.float32, np.float64):
        x_local_np = x_local_np.astype(np.float64)
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
