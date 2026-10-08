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

    from .transport import NeighbourPlan


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
    """Fixed CSR structure of this rank's rows of ``Aᵀ`` and the value routing
    from ``A`` (each entry travels to its column's owner).

    Fields:
        indices, indptr: ``Aᵀ``'s local CSR structure (global column ids, int64).
        local_source_ids, local_target_ids: entries of ``A`` whose column this
            rank owns, and their positions in ``Aᵀ``'s values.
        send_ids: entries of ``A`` to send, packed by owner (ascending).
        recv_target_ids: positions in ``Aᵀ``'s values of the received entries,
            packed by source (ascending).
        exchange: the neighbour plan (one value per entry).
        max_nnz: the largest ``nnz(Aᵀ)`` over ranks, for padded (sharded)
            layouts; ``None`` unless the plan was built with ``pad=True``.

    The routing arrays are padded with one-past-the-end sentinels (a zero is
    appended to each values array): to at least one entry, and with
    ``pad=True`` to the global maxima. Reversing the roles gives the routing
    back from ``Aᵀ`` to ``A``.
    """

    indices: np.ndarray | jax.Array
    indptr: np.ndarray | jax.Array
    local_source_ids: np.ndarray | jax.Array
    local_target_ids: np.ndarray | jax.Array
    send_ids: np.ndarray | jax.Array
    recv_target_ids: np.ndarray | jax.Array
    exchange: "NeighbourPlan"
    max_nnz: int | None = None

    @property
    def nnz(self) -> int:
        return len(self.indices)


# Plans memoized by structure, so repeated uncached solves reuse their
# registered exchanges (a collective LAND keeps every rank's decision alike).
_PLAN_MEMO: dict[tuple, tuple] = {}
_PLAN_MEMO_SIZE = 64


def _memoized(kind: str, comm: "Comm", parts: tuple, build: Callable[[], NamedTuple]):
    import hashlib

    from mpi4py import MPI

    digest = hashlib.blake2b(digest_size=16)
    for part in parts:
        if isinstance(part, np.ndarray):
            digest.update(np.ascontiguousarray(part).tobytes())
            digest.update(str((part.dtype, part.shape)).encode())
        else:
            digest.update(repr(part).encode())
    from .transport import transport_comm

    key = (kind, int(MPI._handleof(comm)), digest.hexdigest())
    tcomm = transport_comm(comm)
    # A hit needs the plan's own transport communicator: a communicator handle
    # can be reused by a new communicator after the old one is freed.
    cached = _PLAN_MEMO.get(key)
    hit = cached is not None and cached[0] is tcomm
    if comm.allreduce(hit, op=MPI.LAND):
        assert cached is not None
        return cached[1]
    plan = build()
    if len(_PLAN_MEMO) >= _PLAN_MEMO_SIZE:
        _PLAN_MEMO.pop(next(iter(_PLAN_MEMO)))
    _PLAN_MEMO[key] = (tcomm, plan)
    return plan


def _global_maxima(comm: "Comm", values: list[int]) -> list[int]:
    """One O(1) control reduction (sizes for padded layouts)."""
    from mpi4py import MPI

    local = np.asarray(values, dtype=np.int64)
    out = np.empty_like(local)
    comm.Allreduce(local, out, op=MPI.MAX)
    return [int(v) for v in out]


def _validate_plan_structure(indices, recvcounts, partition_info, comm, indptr=None):
    """Reject bad local structure on all ranks before peer discovery."""
    from mpi4py import MPI

    start, end = partition_info
    total = sum(recvcounts)
    limit = np.iinfo(np.int32).max
    valid = (
        len(recvcounts) == comm.Get_size()
        and all(n >= 0 for n in recvcounts)
        and 0 <= start <= end <= total <= np.iinfo(np.int64).max
        and start == sum(recvcounts[: comm.Get_rank()])
        and end - start == recvcounts[comm.Get_rank()]
        and end - start <= limit
        and indices.ndim == 1
        and (indices.size == 0 or indices.dtype.kind in "iu")
        and indices.size <= limit
        and np.all((indices >= 0) & (indices < total))
    )
    if indptr is not None:
        valid = valid and (
            indptr.ndim == 1
            and indptr.dtype.kind in "iu"
            and indptr.size == end - start + 1
            and indptr[0] == 0
            and indptr[-1] == indices.size
            and np.all(indptr[1:] >= indptr[:-1])
        )
    if comm.allreduce(not valid, op=MPI.LOR):
        raise ValueError(
            "invalid CSR structure or partition in MPI plan; columns must be "
            "in range and local row/nonzero counts must fit signed 32-bit indices"
        )


def build_transpose_plan(
    indices: ArrayLike,
    indptr: ArrayLike,
    recvcounts: tuple[int, ...],
    partition_info: tuple[int, int],
    comm: "Comm",
    *,
    pad: bool = False,
) -> TransposePlan:
    """Precompute the structure and neighbour value routing for ``A.T``.

    Setup exchanges each off-rank entry's coordinates with its column's owner
    only (sparse peer discovery); nothing of size P is sent or allocated beyond
    the partition offsets.
    """
    indices = np.asarray(indices)
    indptr = np.asarray(indptr)
    return _memoized(
        "transpose",
        comm,
        (indices, indptr, tuple(recvcounts), tuple(partition_info), pad),
        lambda: _build_transpose_plan(
            indices, indptr, recvcounts, partition_info, comm, pad
        ),
    )


def _build_transpose_plan(indices, indptr, recvcounts, partition_info, comm, pad):
    from mpi4py import MPI

    from .transport import make_plan, owners_of, sparse_exchange

    _validate_plan_structure(indices, recvcounts, partition_info, comm, indptr)
    indices = indices.astype(np.int64, copy=False)
    indptr = indptr.astype(np.int64, copy=False)
    rank = comm.Get_rank()
    row_start, row_end = partition_info
    n_local = row_end - row_start
    offsets = np.cumsum(np.array([0, *recvcounts], dtype=np.int64))

    source_rows = np.repeat(
        np.arange(row_start, row_end, dtype=np.int64), np.diff(indptr)
    )
    owners = owners_of(indices, offsets)
    order = np.argsort(owners, kind="stable")
    sorted_owners = owners[order]
    peers, starts, counts = np.unique(
        sorted_owners, return_index=True, return_counts=True
    )
    by_owner = {int(p): order[s : s + c] for p, s, c in zip(peers, starts, counts)}
    outgoing = {
        p: np.column_stack((indices[ids], source_rows[ids])).ravel()
        for p, ids in by_owner.items()
        if p != rank
    }
    received = {s: v.reshape(-1, 2) for s, v in sparse_exchange(comm, outgoing).items()}

    # Arrivals in ascending source order (this rank's own entries in its slot);
    # the Aᵀ order is the sort by (local row, column), independent of arrival.
    local_ids = by_owner.get(rank, np.zeros(0, np.int64))
    segments = []
    for source in sorted(set(received) | {rank}):
        if source == rank:
            coords = np.column_stack((indices[local_ids], source_rows[local_ids]))
        else:
            coords = received[source]
        segments.append((source, coords))
    arrivals = (
        np.concatenate([c for _, c in segments])
        if segments
        else np.zeros((0, 2), np.int64)
    )
    local_rows = arrivals[:, 0] - row_start
    columns = arrivals[:, 1]
    if comm.allreduce(len(columns) > np.iinfo(np.int32).max, op=MPI.LOR):
        raise ValueError(
            "transposed local nonzero count exceeds signed 32-bit indexing"
        )
    ordering = np.lexsort((columns, local_rows))
    position = np.empty(len(ordering), dtype=np.int64)
    position[ordering] = np.arange(len(ordering), dtype=np.int64)
    row_counts = np.bincount(local_rows[ordering], minlength=n_local)
    transpose_indptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int32)

    offset = 0
    local_target_ids = np.zeros(0, np.int64)
    recv_targets, recv_counts = [], {}
    for source, coords in segments:
        span = position[offset : offset + len(coords)]
        offset += len(coords)
        if source == rank:
            local_target_ids = span
        else:
            recv_targets.append(span)
            recv_counts[source] = len(coords)
    send_counts = {p: len(ids) for p, ids in by_owner.items() if p != rank}
    send_ids = (
        np.concatenate([by_owner[p] for p in sorted(send_counts)])
        if send_counts
        else np.zeros(0, np.int64)
    )
    recv_target_ids = (
        np.concatenate(recv_targets) if recv_targets else np.zeros(0, np.int64)
    )
    plan = make_plan(comm, send_counts, recv_counts)

    # Routing arrays keep at least one entry (sentinels one past the end), so
    # every rank's exchange depends on its values: see ``transport.exchange``.
    nnz_t = len(ordering)
    max_nnz = None
    send_width, recv_width = max(len(send_ids), 1), max(len(recv_target_ids), 1)
    if pad:
        max_send, max_recv, max_nnz = _global_maxima(
            comm, [len(send_ids), len(recv_target_ids), nnz_t]
        )
        send_width, recv_width = max(max_send, 1), max(max_recv, 1)
    send_ids = np.pad(
        send_ids, (0, send_width - len(send_ids)), constant_values=len(indices)
    )
    recv_target_ids = np.pad(
        recv_target_ids, (0, recv_width - len(recv_target_ids)), constant_values=nnz_t
    )
    return TransposePlan(
        columns[ordering].astype(np.int64),
        transpose_indptr,
        local_ids.astype(np.int32),
        local_target_ids.astype(np.int32),
        send_ids.astype(np.int32),
        recv_target_ids.astype(np.int32),
        plan,
        max_nnz,
    )


class HaloPlan(NamedTuple):
    """Static neighbour plan for the solution halo ``[x_local | x_ghost]``.

    SpMV and the gradient ``dL/dA_ij = -adj_b[i] * x[j]`` need ``x[j]`` for the
    global columns ``j`` this rank's rows reference: locally owned ones and a
    set of remote "ghost" columns, fetched from their owners only.

    Fields:
        n_local: Rows owned by this rank.
        n_ghost: Distinct remote columns this rank references, in ascending
            global order (the ghost slots; also the order they arrive in).
        col_to_combined: For each local nonzero, its index into the combined
            ``[x_local | x_ghost]`` vector (length nnz).
        send_ids: Local ``x`` indices to send, packed by requesting rank.
        exchange: The neighbour plan.
        max_n_ghost: Largest ``n_ghost`` over ranks, for padded (sharded)
            layouts (then ``send_ids`` is padded with index 0 to the largest
            send total); ``None`` unless built with ``pad=True``. ``send_ids``
            always keeps at least one entry.
    """

    n_local: int
    n_ghost: int
    col_to_combined: np.ndarray | jax.Array
    send_ids: np.ndarray | jax.Array
    exchange: "NeighbourPlan"
    max_n_ghost: int | None = None


def build_halo_plan(
    local_col_indices: ArrayLike,
    recvcounts_tuple: tuple[int, ...],
    partition_info: tuple[int, int],
    comm: "Comm",
    *,
    pad: bool = False,
) -> HaloPlan:
    """Build the halo plan (see :class:`HaloPlan`). Each rank tells the owners
    of its ghost columns which ids it needs (sparse peer discovery); the
    sparsity pattern is fixed, so this runs once."""
    cols = np.asarray(local_col_indices)
    return _memoized(
        "halo",
        comm,
        (cols, tuple(recvcounts_tuple), tuple(partition_info), pad),
        lambda: _build_halo_plan(cols, recvcounts_tuple, partition_info, comm, pad),
    )


def _build_halo_plan(cols, recvcounts_tuple, partition_info, comm, pad):
    from mpi4py import MPI

    from .transport import make_plan, owners_of, sparse_exchange

    _validate_plan_structure(cols, recvcounts_tuple, partition_info, comm)
    cols = cols.astype(np.int64, copy=False)
    row_start, row_end = partition_info
    n_local = row_end - row_start
    offsets = np.cumsum(np.array([0, *recvcounts_tuple], dtype=np.int64))

    uniq = np.unique(cols)
    ghost_global_ids = uniq[(uniq < row_start) | (uniq >= row_end)]  # ascending
    n_ghost = int(ghost_global_ids.size)
    if comm.allreduce(n_local + n_ghost > np.iinfo(np.int32).max, op=MPI.LOR):
        raise ValueError("local and ghost row count exceeds signed 32-bit indexing")
    ghost_owner = owners_of(ghost_global_ids, offsets)
    peers, starts, counts = np.unique(
        ghost_owner, return_index=True, return_counts=True
    )
    requests = {
        int(p): ghost_global_ids[s : s + c] for p, s, c in zip(peers, starts, counts)
    }
    asked = sparse_exchange(comm, requests)
    send_counts = {s: len(ids) for s, ids in asked.items()}
    send_ids = (
        np.concatenate([asked[s] - row_start for s in sorted(asked)])
        if asked
        else np.zeros(0, np.int64)
    )
    plan = make_plan(comm, send_counts, {p: len(v) for p, v in requests.items()})

    # Index arrays keep at least one entry (index 0, whose value is never
    # sent): see ``transport.exchange``.
    max_n_ghost = None
    width = max(len(send_ids), 1)
    if pad:
        max_send, max_n_ghost = _global_maxima(comm, [len(send_ids), n_ghost])
        width = max(max_send, 1)
    send_ids = np.pad(send_ids, (0, width - len(send_ids)))

    local_mask = (cols >= row_start) & (cols < row_end)
    ghost_pos = np.clip(np.searchsorted(ghost_global_ids, cols), 0, max(n_ghost - 1, 0))
    col_to_combined = np.where(
        local_mask, cols - row_start, n_local + ghost_pos
    ).astype(np.int32)
    return HaloPlan(
        n_local,
        n_ghost,
        col_to_combined,
        send_ids.astype(np.int32),
        plan,
        max_n_ghost,
    )


def apply_transpose_plan(
    data: jax.Array,
    local_source_ids: jax.Array,
    local_target_ids: jax.Array,
    send_ids: jax.Array,
    recv_target_ids: jax.Array,
    nnz_out: int,
    exchange: Callable[[jax.Array], jax.Array],
) -> jax.Array:
    """Route values along a transpose plan; ``exchange`` moves the packed
    ``data[send_ids]`` (the neighbour exchange, or its reverse). Sentinel
    entries point one past the end, at an appended zero."""
    data = jnp.pad(data, (0, 1))
    values = jnp.zeros(nnz_out + 1, dtype=data.dtype)
    values = values.at[local_target_ids].set(data[local_source_ids])
    values = values.at[recv_target_ids].set(exchange(data[send_ids]))
    return values[:-1]


def transpose_values(
    data: jax.Array,
    plan_arrays: tuple,
    plan: "NeighbourPlan",
    nnz_out: int,
    *,
    reverse: bool = False,
    ordered: bool = True,
) -> jax.Array:
    """``Aᵀ``'s values from ``A``'s (or, ``reverse``, ``A``'s from ``Aᵀ``'s),
    with the plan's arrays ``(local_source, local_target, send, recv_target)``
    passed as runtime operands."""
    from .transport import exchange

    local_source, local_target, send, recv_target = plan_arrays
    if reverse:
        local_source, local_target = local_target, local_source
        send, recv_target = recv_target, send
    return apply_transpose_plan(
        data,
        local_source,
        local_target,
        send,
        recv_target,
        nnz_out,
        lambda packed: exchange(
            packed,
            plan,
            reverse=reverse,
            ordered=ordered,
            out_size=recv_target.shape[0],
        ),
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
