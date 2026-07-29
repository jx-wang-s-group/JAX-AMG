"""JAX sharding integration for distributed AmgX solves.

This module provides an additive interface on top of JAX-AMG's MPI backend.
JAX owns the global arrays and ``shard_map`` execution, while AmgX continues
to use one MPI rank per GPU for the distributed solve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .cache import _build_mpi_cache, with_cache
from .jaxamg import solve
from .mpi_utils import HaloPlan, build_halo_plan
from .utils import (
    MatrixOrOperator,
    get_preferred_dtype,
    temp_enable_x64,
    to_bcsr_matrix,
)

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

ShardedInfo = dict[str, jax.Array]


class _CSRStructure(NamedTuple):
    """Rank-local CSR structure without an otherwise unused values buffer."""

    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nnz: int


class _TransposePlan(NamedTuple):
    """Static structure and sparse value-exchange plan for a transpose."""

    local_source_ids: np.ndarray
    local_target_ids: np.ndarray
    send_ids_2d: np.ndarray
    recv_target_ids_2d: np.ndarray
    max_nnz: int


def _primal_halo_placeholder(n_local: int, nranks: int) -> HaloPlan:
    """Minimal halo operands for an MPI solve whose primal does not use them."""
    return HaloPlan(
        n_local=n_local,
        n_ghost=0,
        max_n_ghost=0,
        max_per_rank=1,
        col_to_combined=np.empty(0, dtype=np.int32),
        send_ids_2d=np.zeros((nranks, 1), dtype=np.int32),
        recv_ghost_slot_2d=np.zeros((nranks, 1), dtype=np.int32),
    )


def _resolve_comm(comm: Comm | None) -> Comm:
    """Use MPI.COMM_WORLD when the caller does not supply a communicator."""
    if comm is not None:
        return comm

    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError(
            "mpi4py is required when comm is omitted from the sharding API"
        ) from exc
    return MPI.COMM_WORLD


def _row_partition_spec(ndim: int, axis_name: str) -> P:
    """Partition the leading row axis and replicate all trailing axes."""
    return P(axis_name, *(None for _ in range(ndim - 1)))


class ShardedMatrix:
    """Distributed CSR matrix with local structure and sharded values."""

    def __init__(
        self,
        local_bcsr: jsp.BCSR,
        data: jax.Array,
        comm: Comm,
        mesh: Mesh,
        axis_name: str,
        partition_info: tuple[int, int],
        row_counts: tuple[int, ...],
        max_local_size: int,
    ) -> None:
        local_device = mesh.local_devices[0]
        local_nnz = int(local_bcsr.data.shape[0])
        local_values = data.addressable_shards[0].data[:local_nnz]
        # Keep only one local matrix-value buffer. In particular, an uneven
        # partition must not retain both the original unpadded values and the
        # padded shard used by ``data``.
        self._local_bcsr = jsp.BCSR(
            (
                local_values,
                jax.device_put(local_bcsr.indices, local_device),
                jax.device_put(local_bcsr.indptr, local_device),
            ),
            shape=local_bcsr.shape,
        )
        self._comm = comm
        self.mesh = mesh
        self.axis_name = axis_name
        self.partition_info = partition_info
        self.row_counts = row_counts
        self.max_local_size = max_local_size
        self.data = data
        self.global_size = int(local_bcsr.shape[1])
        self.local_size = int(local_bcsr.shape[0])
        self.local_nnz = local_nnz
        self.max_local_nnz = int(data.addressable_shards[0].data.shape[0])
        self.shape = (self.global_size, self.global_size)
        self.local_shape = tuple(local_bcsr.shape)

    def local_matrix(self, data: jax.Array | None = None) -> jsp.BCSR:
        """Return this rank's unpadded BCSR matrix for ``data`` or cached values."""
        values = self.data if data is None else data
        _validate_operand(values, self.data, self.mesh, self.axis_name, "matrix data")
        local_values = values.addressable_shards[0].data[: self.local_nnz]
        return jsp.BCSR(
            (local_values, self._local_bcsr.indices, self._local_bcsr.indptr),
            shape=self._local_bcsr.shape,
        )


class ShardedSolve:
    """Callable sharded solver with differentiable packed matrix values."""

    def __init__(
        self,
        solve_fn: Callable[..., tuple[jax.Array, ShardedInfo]],
        local_vector_fn: Callable[[jax.Array], jax.Array],
        global_size: int,
        local_size: int,
    ) -> None:
        self._solve_fn = solve_fn
        self._local_vector_fn = local_vector_fn
        self.global_size = global_size
        self.local_size = local_size

    def __call__(
        self,
        b: jax.Array,
        x0: jax.Array | None = None,
        *,
        A_data: jax.Array | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        """Solve with cached values or an explicit differentiable ``A_data``.

        ``A_data`` may be omitted for a direct call but is required inside a
        JAX transformation so matrix values remain a dynamic operand.
        """
        return self._solve_fn(b, x0, A_data=A_data)

    def local_vector(self, value: jax.Array) -> jax.Array:
        """Return this rank's unpadded rows of a solver vector or RHS matrix."""
        return self._local_vector_fn(value)


def make_sharded_vector(
    local_values: Any,
    *,
    comm: Comm | None = None,
    mesh: Mesh | None = None,
    global_size: int | None = None,
    axis_name: str = "rank",
) -> jax.Array:
    """Create a row-sharded vector or RHS matrix, padding unequal partitions.

    The returned JAX array has equal physical shard sizes, as required by
    ``NamedSharding``. ``make_sharded_solver`` ignores each shard's padding and
    uses the true row counts from ``A_local``. JAX array inputs are padded and
    assembled on device; other array-like inputs use a NumPy host staging path.

    Args:
        local_values: This rank's unpadded values with shape ``(n_local,)`` or
            ``(n_local, nrhs)``.
        comm: MPI communicator whose rank order matches ``mesh``. Defaults to
            ``MPI.COMM_WORLD``.
        mesh: One-dimensional JAX device mesh with one device per MPI rank.
            Defaults to a mesh containing all JAX devices.
        global_size: Optional true global length. When provided, it is checked
            against the sum of local lengths.
        axis_name: Mesh axis used to partition the vector.

    Returns:
        A global JAX array whose leading axis uses ``P(axis_name)`` and whose
        optional RHS-column axis is replicated. Its physical leading-axis
        length is ``comm.size * max(local_sizes)``.
    """
    comm = _resolve_comm(comm)
    if mesh is None:
        mesh = jax.make_mesh((jax.device_count(),), (axis_name,))
    values: jax.Array | np.ndarray
    if isinstance(local_values, jax.Array):
        if not local_values.is_fully_addressable:
            raise ValueError(
                "local_values must be a process-local JAX array, not an already "
                "distributed global array"
            )
        values = local_values
    else:
        values = np.asarray(local_values)
    if values.ndim not in (1, 2):
        raise ValueError(
            f"local_values must be one- or two-dimensional; got shape {values.shape}"
        )
    if values.ndim == 2 and values.shape[1] == 0:
        raise ValueError("a batched RHS must contain at least one column")
    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            f"mesh must have the single axis {axis_name!r}; got {mesh.axis_names!r}"
        )

    comm_size = comm.Get_size()
    if mesh.size != comm_size or mesh.shape[axis_name] != comm_size:
        raise ValueError(
            "the mesh axis must contain exactly one device per MPI rank; got "
            f"mesh size {mesh.size} and communicator size {comm_size}"
        )

    local_shapes = tuple(tuple(shape) for shape in comm.allgather(values.shape))
    trailing_shape = values.shape[1:]
    if any(shape[1:] != trailing_shape for shape in local_shapes):
        raise ValueError(
            f"all MPI ranks must use the same trailing RHS shape; got {local_shapes}"
        )
    local_sizes = tuple(int(shape[0]) for shape in local_shapes)
    inferred_global_size = sum(local_sizes)
    if global_size is not None and int(global_size) != inferred_global_size:
        raise ValueError(
            f"global_size is {global_size}, but local sizes sum to "
            f"{inferred_global_size}"
        )

    max_local_size = max(local_sizes)
    padding = ((0, max_local_size - values.shape[0]),) + ((0, 0),) * (values.ndim - 1)
    if max_local_size == values.shape[0]:
        padded_values = values
    elif isinstance(values, jax.Array):
        padded_values = jnp.pad(values, padding)
    else:
        padded_values = np.pad(values, padding)
    sharding = NamedSharding(mesh, _row_partition_spec(values.ndim, axis_name))
    return jax.make_array_from_process_local_data(
        sharding,
        padded_values,
        global_shape=(comm_size * max_local_size, *trailing_shape),
    )


def _require_shard_map() -> None:
    if not hasattr(jax, "shard_map"):
        raise RuntimeError(
            "JAX-AMG's sharding interface requires a JAX version that exposes "
            "jax.shard_map. Upgrade JAX or use solve(..., comm=...) instead."
        )


def _resolve_mesh(
    b: jax.Array,
    mesh: Mesh | None,
    axis_name: str,
) -> Mesh:
    if mesh is None:
        sharding = getattr(b, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise ValueError(
                "mesh was not provided and b does not have NamedSharding; "
                "pass mesh=... or place b with NamedSharding first"
            )
        inferred_mesh = sharding.mesh
        if not isinstance(inferred_mesh, Mesh):
            raise TypeError("b must use a concrete JAX Mesh")
        mesh = inferred_mesh

    if tuple(mesh.axis_names) != (axis_name,):
        raise ValueError(
            "the initial sharding interface requires a one-dimensional mesh "
            f"with axis name {axis_name!r}; got axes {tuple(mesh.axis_names)!r}"
        )
    return mesh


def _validate_runtime(comm: Comm, mesh: Mesh, axis_name: str) -> None:
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    if comm_size > 1 and not jax.distributed.is_initialized():
        raise RuntimeError(
            "jax.distributed.initialize() must be called before creating a "
            "multi-process sharded solver"
        )
    if jax.process_count() != comm_size:
        raise ValueError(
            "the JAX process count and MPI communicator size must match; got "
            f"{jax.process_count()} and {comm_size}"
        )
    if jax.process_index() != comm_rank:
        raise ValueError(
            "the JAX process index must match the MPI rank; got "
            f"{jax.process_index()} and {comm_rank}"
        )
    if mesh.size != comm_size or mesh.shape[axis_name] != comm_size:
        raise ValueError(
            "the mesh axis must contain exactly one device per MPI rank; got "
            f"mesh size {mesh.size} and communicator size {comm_size}"
        )

    local_devices = tuple(mesh.local_devices)
    if len(local_devices) != 1:
        raise ValueError(
            "the initial sharding interface requires exactly one mesh-local "
            f"device per MPI process; got {len(local_devices)}"
        )
    mesh_position = tuple(mesh.devices.flat).index(local_devices[0])
    if mesh_position != comm_rank:
        raise ValueError(
            "mesh device order must match MPI rank order; this process owns "
            f"MPI rank {comm_rank} but mesh position {mesh_position}"
        )
    if local_devices[0].platform != "gpu":
        raise RuntimeError(
            "AMGX requires GPU mesh devices; the local mesh device uses "
            f"platform {local_devices[0].platform!r}"
        )


def _validate_vector_layout(
    b: jax.Array,
    mesh: Mesh,
    axis_name: str,
    comm_size: int,
    max_local_size: int,
) -> None:
    if b.ndim not in (1, 2):
        raise ValueError(f"b must be one- or two-dimensional; got shape {b.shape}")
    if b.ndim == 2 and b.shape[1] == 0:
        raise ValueError("a batched RHS must contain at least one column")

    sharding = getattr(b, "sharding", None)
    expected_spec = _row_partition_spec(b.ndim, axis_name)
    if not isinstance(sharding, NamedSharding):
        raise ValueError("b must use NamedSharding")
    if sharding.mesh != mesh or sharding.spec != expected_spec:
        raise ValueError(
            f"b must use NamedSharding(mesh, P({axis_name!r})); got {sharding!r}"
        )

    expected_shape = (comm_size * max_local_size, *b.shape[1:])
    if b.shape != expected_shape:
        raise ValueError(
            f"b must have padded sharded shape {expected_shape}; got {b.shape}. "
            "Use jaxamg.make_sharded_vector for uneven row partitions."
        )

    shards = b.addressable_shards
    if len(shards) != 1:
        raise ValueError(
            "each MPI process must own exactly one addressable RHS shard; got "
            f"{len(shards)}"
        )
    index = shards[0].index
    if len(index) != b.ndim or not isinstance(index[0], slice):
        raise ValueError(f"b must have one contiguous local shard; got index {index!r}")

    row_slice = index[0]
    if row_slice.step not in (None, 1):
        raise ValueError(f"b shard must be contiguous; got slice {row_slice!r}")
    physical_start = 0 if row_slice.start is None else int(row_slice.start)
    physical_end = b.shape[0] if row_slice.stop is None else int(row_slice.stop)
    if physical_end - physical_start != max_local_size:
        raise ValueError(
            "each physical RHS shard must have the maximum local row count; "
            f"expected {max_local_size}, got {physical_end - physical_start}"
        )


def _local_partition(
    b: jax.Array,
    mesh: Mesh,
    axis_name: str,
    comm: Comm,
    n_local: int,
    n_global: int,
) -> tuple[tuple[int, int], tuple[int, ...], int]:
    local_sizes = tuple(int(size) for size in comm.allgather(n_local))
    if sum(local_sizes) != n_global:
        raise ValueError(
            "the distributed matrix row counts must sum to its global column "
            f"count; got row counts {local_sizes} and {n_global} columns"
        )
    max_local_size = max(local_sizes)
    _validate_vector_layout(b, mesh, axis_name, comm.Get_size(), max_local_size)

    comm_rank = comm.Get_rank()
    row_start = sum(local_sizes[:comm_rank])
    return (row_start, row_start + n_local), local_sizes, max_local_size


def _validate_matrix(
    A_local: MatrixOrOperator,
    nglobal: int,
    partition_info: tuple[int, int],
    block_dim: int,
) -> None:
    shape = getattr(A_local, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("A_local must have a two-dimensional shape")

    n_local = partition_info[1] - partition_info[0]
    if tuple(shape) != (n_local, nglobal):
        raise ValueError(
            "A_local must contain this process's rows and all global columns; "
            f"expected shape {(n_local, nglobal)}, got {tuple(shape)}"
        )
    if block_dim < 1:
        raise ValueError("block_dim must be a positive integer")
    if n_local % block_dim != 0 or nglobal % block_dim != 0:
        raise ValueError(
            f"local partition ({n_local} rows) and global size ({nglobal}) "
            f"must be divisible by block_dim {block_dim}"
        )


def _normalize_local_matrix(
    A_local: MatrixOrOperator,
    b: jax.Array,
    partition_info: tuple[int, int],
) -> jsp.BCSR:
    """Normalize a validated local matrix partition."""
    n_local = partition_info[1] - partition_info[0]
    nglobal = int(getattr(A_local, "shape")[1])
    _validate_matrix(A_local, nglobal, partition_info, block_dim=1)

    local_rhs = b.addressable_shards[0].data[:n_local]
    matrix_probe_rhs = local_rhs[:, 0] if b.ndim == 2 else local_rhs
    return to_bcsr_matrix(A_local, b=matrix_probe_rhs, use_int64_indices=True)


def _pack_sharded_matrix(
    A_bcsr: jsp.BCSR,
    comm: Comm,
    mesh: Mesh,
    axis_name: str,
    partition_info: tuple[int, int],
    row_counts: tuple[int, ...],
    max_local_size: int,
    max_nnz: int | None = None,
) -> ShardedMatrix:
    """Pack normalized local values into a global sharded array."""
    local_nnz = int(A_bcsr.data.shape[0])
    if max_nnz is None:
        max_nnz = max(int(value) for value in comm.allgather(local_nnz))
    local_packed_data = (
        A_bcsr.data
        if max_nnz == local_nnz
        else jnp.pad(A_bcsr.data, (0, max_nnz - local_nnz))
    )
    data_sharding = NamedSharding(mesh, P(axis_name))
    data = jax.make_array_from_process_local_data(
        data_sharding,
        local_packed_data,
        global_shape=(comm.Get_size() * max_nnz,),
    )
    return ShardedMatrix(
        A_bcsr,
        data,
        comm,
        mesh,
        axis_name,
        partition_info,
        row_counts,
        max_local_size,
    )


def make_sharded_matrix(
    A_local: MatrixOrOperator,
    b: jax.Array,
    *,
    comm: Comm | None = None,
    mesh: Mesh | None = None,
    axis_name: str = "rank",
) -> ShardedMatrix:
    """Create a distributed CSR container without replicating the global matrix.

    The CSR column indices and row pointers remain rank-local static structure.
    Matrix values are packed into a global JAX array sharded over the same mesh
    axis as ``b``. Unequal local nonzero counts are padded to the largest count;
    the padding is ignored by solves and gradients.

    Args:
        A_local: This process's CSR row partition with shape
            ``(n_local, n_global)`` and global column indices.
        b: Global row-sharded RHS used to validate the matrix partition, mesh,
            and numerical dtype.
        comm: MPI communicator whose rank order matches the JAX process order.
            Defaults to ``MPI.COMM_WORLD``.
        mesh: One-dimensional JAX device mesh. If omitted, use the mesh from
            ``b.sharding``.
        axis_name: Name of the mesh axis that partitions rows and packed values.

    Returns:
        A :class:`ShardedMatrix` whose ``data`` attribute contains the global
        sharded values. The original global matrix is never materialized.
    """
    _require_shard_map()
    if not isinstance(b, jax.Array):
        raise TypeError("b must be a global jax.Array")
    comm = _resolve_comm(comm)
    mesh = _resolve_mesh(b, mesh, axis_name)
    _validate_runtime(comm, mesh, axis_name)

    matrix_shape = getattr(A_local, "shape", None)
    if matrix_shape is None or len(matrix_shape) != 2:
        raise ValueError("A_local must have a two-dimensional shape")
    n_local = int(matrix_shape[0])
    nglobal = int(matrix_shape[1])
    partition_info, row_counts, max_local_size = _local_partition(
        b, mesh, axis_name, comm, n_local, nglobal
    )
    A_bcsr = _normalize_local_matrix(A_local, b, partition_info)
    return _pack_sharded_matrix(
        A_bcsr,
        comm,
        mesh,
        axis_name,
        partition_info,
        row_counts,
        max_local_size,
    )


def _validate_operand(
    value: jax.Array,
    template: jax.Array,
    mesh: Mesh,
    axis_name: str,
    name: str,
) -> None:
    if value.shape != template.shape:
        raise ValueError(f"{name} must have shape {template.shape}; got {value.shape}")
    if value.dtype != template.dtype:
        raise ValueError(f"{name} must have dtype {template.dtype}; got {value.dtype}")
    # During an outer JAX transform ``value`` is a tracer and its concrete
    # sharding is supplied by shard_map. Validate eager calls here.
    if not isinstance(value, jax.core.Tracer):
        sharding = getattr(value, "sharding", None)
        if not isinstance(sharding, NamedSharding):
            raise ValueError(f"{name} must use NamedSharding")
        if sharding.mesh != mesh or sharding.spec != _row_partition_spec(
            value.ndim, axis_name
        ):
            raise ValueError(f"{name} must use the same NamedSharding as its template")


def _transpose_distributed_matrix(
    indices: np.ndarray,
    indptr: np.ndarray,
    recvcounts: tuple[int, ...],
    partition_info: tuple[int, int],
    comm: Comm,
    local_device: jax.Device,
) -> tuple[_CSRStructure, _TransposePlan]:
    """Transpose fixed CSR structure and build a sparse value-exchange plan."""
    from mpi4py import MPI

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
    coordinate_send_counts = 2 * send_counts
    coordinate_recv_counts = 2 * recv_counts
    coordinate_send_displs = 2 * send_displs
    coordinate_recv_displs = 2 * recv_displs
    comm.Alltoallv(
        [
            send_coordinates,
            coordinate_send_counts,
            coordinate_send_displs,
            MPI.INT64_T,
        ],
        [
            recv_coordinates,
            coordinate_recv_counts,
            coordinate_recv_displs,
            MPI.INT64_T,
        ],
    )
    recv_rows = recv_coordinates[:, 0]
    recv_cols = recv_coordinates[:, 1]
    local_rows = recv_rows - row_start
    order = np.lexsort((recv_cols, local_rows))
    local_rows = local_rows[order]
    recv_cols = recv_cols[order]
    row_counts = np.bincount(local_rows, minlength=n_local)
    transpose_indptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int32)

    # The dynamic transpose communicates only off-rank values. Entries whose
    # transpose rows remain on this rank are gathered locally, so the padded
    # all-to-all chunk is governed by the largest remote rank pair rather than
    # by the (usually much larger) on-rank diagonal block.
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

    send_ids_2d = np.zeros((nranks, max_per_rank), dtype=np.int32)
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

    with temp_enable_x64():
        transpose_indices = jax.device_put(
            np.asarray(recv_cols, dtype=np.int64), local_device
        )
        transpose_structure = _CSRStructure(
            transpose_indices,
            jax.device_put(transpose_indptr, local_device),
            (n_local, n_global),
            recv_nnz,
        )
    return transpose_structure, _TransposePlan(
        local_source_ids,
        local_target_ids,
        send_ids_2d,
        recv_target_ids_2d,
        max_nnz,
    )


def make_sharded_solver(
    A: ShardedMatrix,
    b: jax.Array,
    *,
    config: dict[str, Any] | None = None,
    is_symmetric: bool = False,
    block_dim: int = 1,
    reuse_setup: bool = False,
) -> ShardedSolve:
    """Create a JIT-compiled solver for a globally sharded RHS.

    This interface complements, rather than replaces, ``solve(..., comm=...)``.
    JAX manages the global input and output arrays through ``shard_map`` while
    the AmgX solve itself uses the supplied MPI communicator. Version one uses
    a one-dimensional mesh and requires one MPI process with one mesh-local GPU
    per rank.

    ``jax.distributed.initialize()`` must be called before this function in a
    multi-process job. The matrix owns the communicator, mesh, local CSR
    structure, and globally sharded packed values. Pass ``A.data`` through
    ``solver(..., A_data=A_data)`` to differentiate matrix values. Use
    ``jax.set_mesh(A.mesh)`` around outer transforms such as ``jax.grad``.

    Args:
        A: Distributed matrix created with :func:`make_sharded_matrix`.
        b: Global JAX array with shape ``(n_global,)`` or
            ``(n_global, nrhs)`` using row ``NamedSharding``. For unequal row
            counts, construct it with :func:`make_sharded_vector`; physical
            shards are padded to the largest local partition. Batched RHS
            columns are replicated within each row shard.
        config: AmgX configuration. Defaults to the JAX-AMG MPI configuration.
        is_symmetric: Whether the global matrix is symmetric. When ``False``
            (the default), the distributed transpose is prepared once during
            solver creation for the reverse-mode adjoint.
        block_dim: AmgX block dimension. The matrix retains its scalar CSR
            representation, and every local row partition must be divisible by
            this value.
        reuse_setup: Reuse the cached AmgX hierarchy across solves with the
            same sparsity pattern.

    Returns:
        A callable ``solver(b, x0=None, *, A_data=None)``. Use
        ``A.local_matrix(gradient)`` to convert packed matrix gradients to this
        rank's unpadded BCSR structure. ``solver.local_vector(value)`` removes
        vector padding. ``A_data`` may be omitted for a direct call but must be
        explicit under ``jax.jit``, ``jax.grad``, or another JAX transform. For
        a vector RHS, info values have one entry per rank. For a batched RHS
        they have shape ``(nranks, nrhs)`` and ``residual_history`` has an
        additional trailing ``max_iters + 1`` axis.
    """
    _require_shard_map()
    if not isinstance(A, ShardedMatrix):
        raise TypeError("A must be created with jaxamg.make_sharded_matrix")
    if not isinstance(b, jax.Array):
        raise TypeError("b must be a global jax.Array")
    comm = A._comm
    mesh = A.mesh
    axis_name = A.axis_name
    _validate_runtime(comm, mesh, axis_name)

    n_local = A.local_size
    nglobal = A.global_size
    partition_info = A.partition_info
    max_n_local = A.max_local_size
    _validate_vector_layout(b, mesh, axis_name, comm.Get_size(), max_n_local)

    block_dim = int(block_dim)
    is_batched = b.ndim == 2
    local_rhs = b.addressable_shards[0].data[:n_local]
    matrix_probe_rhs = local_rhs[:, 0] if is_batched else local_rhs
    A_bcsr = A._local_bcsr
    if get_preferred_dtype(A_bcsr, matrix_probe_rhs) != A.data.dtype:
        raise ValueError(
            "the sharded matrix dtype is incompatible with b; rebuild it "
            "with make_sharded_matrix(A_local, b)"
        )

    _validate_matrix(A_bcsr, nglobal, partition_info, block_dim)
    local_device = mesh.local_devices[0]
    local_hardware_id = getattr(local_device, "local_hardware_id", local_device.id)

    # MPI setup needs CSR structure on the host. Materialize it exactly once;
    # the halo and transpose plans below share these arrays.
    indices_host = np.asarray(A_bcsr.indices, dtype=np.int64)
    indptr_host = np.asarray(A_bcsr.indptr, dtype=np.int64)
    halo_plan = build_halo_plan(indices_host, A.row_counts, partition_info, comm)

    rhs_spec = _row_partition_spec(b.ndim, axis_name)
    A_data_spec = P(axis_name)
    max_nnz = A.max_local_nnz
    local_nnz = A.local_nnz
    A_data = A.data
    A_structure = _CSRStructure(
        A_bcsr.indices,
        A_bcsr.indptr,
        tuple(A_bcsr.shape),
        local_nnz,
    )

    if is_symmetric:
        nnz_out = None
        transpose_structure = None
        transpose_cache = None
        max_transpose_nnz = None
        transpose_local_source_ids = None
        transpose_local_target_ids = None
        transpose_send_ids = None
        transpose_recv_target_ids = None
    else:
        transpose_structure, transpose_plan = _transpose_distributed_matrix(
            indices_host,
            indptr_host,
            A.row_counts,
            partition_info,
            comm,
            local_device,
        )
        nnz_out = transpose_structure.nnz
        max_transpose_nnz = transpose_plan.max_nnz
        # Explicit single-device placement keeps rank-local constants local even
        # when solver construction happens inside a global ``jax.set_mesh``
        # context.
        transpose_local_source_ids = jax.device_put(
            transpose_plan.local_source_ids, local_device
        )
        transpose_local_target_ids = jax.device_put(
            transpose_plan.local_target_ids, local_device
        )
        transpose_send_ids = jax.device_put(transpose_plan.send_ids_2d, local_device)
        transpose_recv_target_ids = jax.device_put(
            transpose_plan.recv_target_ids_2d, local_device
        )

    mpi_cache = _build_mpi_cache(
        config or {},
        comm,
        nglobal,
        A.row_counts,
        max_nnz,
        nnz_out,
        halo_plan,
        block_dim=block_dim,
        lrank=int(local_hardware_id),
    )
    if not is_symmetric:
        # The MPI solve's primal does not consume halo operands, and sharding's
        # outer VJP computes its matrix gradient from A's existing halo plan.
        # Therefore the nested A^T solve needs only placeholder halo operands.
        transpose_cache = {
            **mpi_cache,
            "max_nnz": max_transpose_nnz,
            "nnz_out": local_nnz,
            "halo_plan": _primal_halo_placeholder(n_local, len(A.row_counts)),
        }

    if is_batched:
        info_specs = {
            "iterations": P(axis_name, None),
            "residual": P(axis_name, None),
            "status": P(axis_name, None),
            "residual_history": P(axis_name, None, None),
        }
    else:
        info_specs = {
            "iterations": P(axis_name),
            "residual": P(axis_name),
            "status": P(axis_name),
            "residual_history": P(axis_name, None),
        }

    def pack_info(info: ShardedInfo) -> ShardedInfo:
        if is_batched:
            return {
                "iterations": jnp.asarray(info["iterations"], dtype=jnp.int32)[None, :],
                "residual": jnp.asarray(info["residual"])[None, :],
                "status": jnp.asarray(info["status"], dtype=jnp.int32)[None, :],
                "residual_history": jnp.asarray(info["residual_history"])[None, :, :],
            }
        return {
            "iterations": jnp.asarray(info["iterations"], dtype=jnp.int32)[None],
            "residual": jnp.asarray(info["residual"])[None],
            "status": jnp.asarray(info["status"], dtype=jnp.int32)[None],
            "residual_history": jnp.asarray(info["residual_history"])[None, :],
        }

    def matrix_with_data(
        data_local: jax.Array,
        structure: _CSRStructure,
        cache: dict[str, Any],
        symmetric: bool,
    ) -> jsp.BCSR:
        matrix = jsp.BCSR(
            (
                data_local[: structure.nnz],
                structure.indices,
                structure.indptr,
            ),
            shape=structure.shape,
        )
        return with_cache(matrix, mpi=cache, is_symmetric=symmetric)

    def pad_local_vector(value: jax.Array) -> jax.Array:
        padding = ((0, max_n_local - n_local),) + ((0, 0),) * (value.ndim - 1)
        return jnp.pad(value, padding)

    def solve_one_rhs(
        A_dynamic: jsp.BCSR,
        rhs_local: jax.Array,
        x0_local: jax.Array | None = None,
        *,
        reuse: bool = reuse_setup,
    ) -> tuple[jax.Array, ShardedInfo]:
        return solve(
            A_dynamic,
            rhs_local[:n_local],
            x0=None if x0_local is None else x0_local[:n_local],
            block_dim=block_dim,
            reuse_setup=reuse,
        )

    def solve_local_rhs(
        A_dynamic: jsp.BCSR,
        rhs_local: jax.Array,
        x0_local: jax.Array | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        if not is_batched:
            return solve_one_rhs(A_dynamic, rhs_local, x0_local)

        solutions: list[jax.Array] = []
        column_info: list[ShardedInfo] = []
        for column in range(rhs_local.shape[1]):
            rhs_column = rhs_local[:, column]
            x0_column = None if x0_local is None else x0_local[:, column]
            # XLA schedules each rank independently. Tie every column to the
            # previous result so all ranks enter AmgX collectives in the same
            # order.
            if solutions:
                if x0_column is None:
                    rhs_column, _ = jax.lax.optimization_barrier(
                        (rhs_column, solutions[-1])
                    )
                else:
                    rhs_column, x0_column, _ = jax.lax.optimization_barrier(
                        (rhs_column, x0_column, solutions[-1])
                    )
            solution, info = solve_one_rhs(
                A_dynamic,
                rhs_column,
                x0_column,
                # All columns use identical matrix values. Once the first
                # column has prepared the hierarchy, resetting it again is
                # unnecessary even when cross-call reuse was not requested.
                reuse=reuse_setup or column > 0,
            )
            solutions.append(solution)
            column_info.append(info)
        return jnp.stack(solutions, axis=1), jax.tree.map(
            lambda *values: jnp.stack(values), *column_info
        )

    def local_solve(
        A_data_local: jax.Array, rhs_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_structure, mpi_cache, is_symmetric)
        x_local, info = solve_local_rhs(A_dynamic, rhs_local)
        return pad_local_vector(x_local), pack_info(info)

    def local_solve_x0(
        A_data_local: jax.Array, rhs_local: jax.Array, x0_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_structure, mpi_cache, is_symmetric)
        x_local, info = solve_local_rhs(A_dynamic, rhs_local, x0_local)
        return pad_local_vector(x_local), pack_info(info)

    def local_transpose_values(A_data_local: jax.Array) -> jax.Array:
        assert transpose_structure is not None
        assert max_transpose_nnz is not None
        assert transpose_local_source_ids is not None
        assert transpose_local_target_ids is not None
        assert transpose_send_ids is not None
        assert transpose_recv_target_ids is not None

        transpose_values = jnp.zeros(
            transpose_structure.nnz + 1, dtype=A_data_local.dtype
        )
        transpose_values = transpose_values.at[transpose_local_target_ids].set(
            A_data_local[transpose_local_source_ids]
        )
        send_buffer = A_data_local[transpose_send_ids]
        recv_buffer = jax.lax.all_to_all(
            send_buffer,
            axis_name,
            split_axis=0,
            concat_axis=0,
        )
        transpose_values = transpose_values.at[
            transpose_recv_target_ids.reshape(-1)
        ].set(recv_buffer.reshape(-1))
        return jnp.pad(
            transpose_values[:-1],
            (0, max_transpose_nnz - transpose_structure.nnz),
        )

    def make_local_adjoint(reuse: bool):
        def local_adjoint(
            adjoint_data_local: jax.Array, g_local: jax.Array
        ) -> jax.Array:
            if is_symmetric:
                structure = A_structure
                cache = mpi_cache
            else:
                assert transpose_structure is not None
                assert transpose_cache is not None
                structure = transpose_structure
                cache = transpose_cache
            A_adjoint = matrix_with_data(
                adjoint_data_local, structure, cache, symmetric=is_symmetric
            )
            adjoint_local, _ = solve_one_rhs(A_adjoint, g_local, reuse=reuse)
            return pad_local_vector(adjoint_local)

        return local_adjoint

    halo_plan = mpi_cache["halo_plan"]
    max_n_ghost = halo_plan.max_n_ghost
    local_row_indices = np.repeat(
        np.arange(A_bcsr.shape[0], dtype=np.int32),
        np.diff(indptr_host),
    )

    # Keep rank-local halo metadata inside the shard_map bodies. Making these
    # arrays global and closing over them in the custom VJP prevents an outer
    # multi-process jax.jit from lowering because their remote shards are not
    # addressable by the current process.
    row_indices = jax.device_put(local_row_indices, local_device)
    col_to_combined = jax.device_put(halo_plan.col_to_combined, local_device)
    send_ids = jax.device_put(halo_plan.send_ids_2d, local_device)
    recv_ghost_slot = jax.device_put(halo_plan.recv_ghost_slot_2d, local_device)

    def gather_solution_halo(
        x_local: jax.Array,
        send_ids_local: jax.Array,
        recv_ghost_slot_local: jax.Array,
    ) -> jax.Array:
        send_buffer = x_local[send_ids_local]
        recv_buffer = jax.lax.all_to_all(
            send_buffer,
            axis_name,
            split_axis=0,
            concat_axis=0,
        )
        trailing_shape = x_local.shape[1:]
        x_ghost = jnp.zeros((max_n_ghost + 1, *trailing_shape), dtype=x_local.dtype)
        x_ghost = x_ghost.at[recv_ghost_slot_local.reshape(-1)].set(
            recv_buffer.reshape((-1, *trailing_shape))
        )
        return jnp.concatenate([x_local, x_ghost[:max_n_ghost]], axis=0)

    def local_matrix_gradient(
        x_local: jax.Array, adjoint_local: jax.Array
    ) -> jax.Array:
        # Order the JAX halo exchange after the preceding AmgX adjoint solve.
        x_ordered, adjoint_ordered = jax.lax.optimization_barrier(
            (x_local, adjoint_local)
        )
        x_combined = gather_solution_halo(
            x_ordered[:n_local], send_ids, recv_ghost_slot
        )
        grad_values = (
            -adjoint_ordered[:n_local][row_indices] * x_combined[col_to_combined]
        )
        return jnp.pad(grad_values, (0, max_nnz - local_nnz))

    mapped_solve = jax.jit(
        jax.shard_map(
            local_solve,
            mesh=mesh,
            in_specs=(A_data_spec, rhs_spec),
            out_specs=(rhs_spec, info_specs),
        )
    )
    mapped_solve_x0 = jax.jit(
        jax.shard_map(
            local_solve_x0,
            mesh=mesh,
            in_specs=(A_data_spec, rhs_spec, rhs_spec),
            out_specs=(rhs_spec, info_specs),
        )
    )

    scalar_spec = P(axis_name)
    if is_symmetric:
        mapped_transpose_values = None
    else:
        mapped_transpose_values = jax.jit(
            jax.shard_map(
                local_transpose_values,
                mesh=mesh,
                in_specs=(A_data_spec,),
                out_specs=A_data_spec,
            )
        )
    mapped_adjoint = jax.jit(
        jax.shard_map(
            make_local_adjoint(reuse_setup),
            mesh=mesh,
            in_specs=(A_data_spec, scalar_spec),
            out_specs=scalar_spec,
        )
    )
    mapped_adjoint_reuse = (
        mapped_adjoint
        if reuse_setup or not is_batched
        else jax.jit(
            jax.shard_map(
                make_local_adjoint(True),
                mesh=mesh,
                in_specs=(A_data_spec, scalar_spec),
                out_specs=scalar_spec,
            )
        )
    )
    mapped_matrix_gradient = jax.jit(
        jax.shard_map(
            local_matrix_gradient,
            mesh=mesh,
            in_specs=(scalar_spec, scalar_spec),
            out_specs=A_data_spec,
        )
    )

    # Place the custom VJP outside shard_map. Its cotangent is then a global
    # sharded array, and the adjoint shard_map receives the correct local shard
    # on every rank. Keeping the custom rule inside shard_map would make the
    # FFI result appear rank-invariant to JAX's varying-manual-axis analysis.
    @jax.custom_vjp
    def differentiated_solve(
        matrix_data: jax.Array, rhs: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        return mapped_solve(matrix_data, rhs)

    def differentiated_solve_fwd(matrix_data: jax.Array, rhs: jax.Array):
        result = mapped_solve(matrix_data, rhs)
        x, _ = result
        return result, (matrix_data, x)

    def differentiated_solve_backward(
        matrix_data: jax.Array, x: jax.Array, g_x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if is_symmetric:
            adjoint_data = matrix_data
        else:
            assert mapped_transpose_values is not None
            # Matrix values are identical for every RHS column, so prepare the
            # distributed transpose exactly once per backward pass. Tie the
            # exchange to the completed forward solution so XLA cannot overlap
            # its JAX collective with AmgX's preceding MPI collectives.
            matrix_data_ordered, _ = jax.lax.optimization_barrier((matrix_data, x))
            adjoint_data = mapped_transpose_values(matrix_data_ordered)

        if not is_batched:
            adjoint = mapped_adjoint(adjoint_data, g_x)
            return adjoint, mapped_matrix_gradient(x, adjoint)

        adjoint_columns: list[jax.Array] = []
        for column in range(x.shape[1]):
            g_column = g_x[:, column]
            # Preserve one cross-rank order for the sequential AmgX solves.
            if adjoint_columns:
                g_column, _ = jax.lax.optimization_barrier(
                    (g_column, adjoint_columns[-1])
                )
            adjoint = (mapped_adjoint if column == 0 else mapped_adjoint_reuse)(
                adjoint_data,
                g_column,
            )
            adjoint_columns.append(adjoint)

        # Run matrix-gradient halo exchanges only after all AmgX adjoint
        # collectives have completed. Accumulating immediately avoids an
        # ``(nrhs, nnz)`` stack of temporary matrix gradients.
        matrix_gradient_sum = None
        last_adjoint = adjoint_columns[-1]
        for column, adjoint in enumerate(adjoint_columns):
            x_column = x[:, column]
            order_dependency = (
                last_adjoint if matrix_gradient_sum is None else matrix_gradient_sum
            )
            x_column, adjoint, _ = jax.lax.optimization_barrier(
                (x_column, adjoint, order_dependency)
            )
            matrix_gradient = mapped_matrix_gradient(x_column, adjoint)
            matrix_gradient_sum = (
                matrix_gradient
                if matrix_gradient_sum is None
                else matrix_gradient_sum + matrix_gradient
            )
        assert matrix_gradient_sum is not None
        return jnp.stack(adjoint_columns, axis=1), matrix_gradient_sum

    def differentiated_solve_bwd(residuals, cotangents):
        matrix_data, x = residuals
        g_x, _ = cotangents
        adjoint, grad_A_data = differentiated_solve_backward(matrix_data, x, g_x)
        return grad_A_data, adjoint

    differentiated_solve.defvjp(differentiated_solve_fwd, differentiated_solve_bwd)

    @jax.custom_vjp
    def differentiated_solve_x0(
        matrix_data: jax.Array, rhs: jax.Array, x0: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        return mapped_solve_x0(matrix_data, rhs, x0)

    def differentiated_solve_x0_fwd(
        matrix_data: jax.Array, rhs: jax.Array, x0: jax.Array
    ):
        result = mapped_solve_x0(matrix_data, rhs, x0)
        x, _ = result
        return result, (matrix_data, x)

    def differentiated_solve_x0_bwd(residuals, cotangents):
        matrix_data, x = residuals
        g_x, _ = cotangents
        adjoint, grad_A_data = differentiated_solve_backward(matrix_data, x, g_x)
        return grad_A_data, adjoint, jnp.zeros_like(adjoint)

    differentiated_solve_x0.defvjp(
        differentiated_solve_x0_fwd, differentiated_solve_x0_bwd
    )

    def sharded_solver(
        rhs: jax.Array,
        x0: jax.Array | None = None,
        *,
        A_data_override: jax.Array | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        _validate_operand(rhs, b, mesh, axis_name, "b")
        if A_data_override is None:
            if isinstance(rhs, jax.core.Tracer):
                raise ValueError(
                    "A_data must be passed explicitly when a sharded solver is "
                    "used inside jax.jit, jax.grad, or another JAX transform"
                )
            matrix_data = A_data
        else:
            matrix_data = A_data_override
        _validate_operand(matrix_data, A_data, mesh, axis_name, "A_data")
        if x0 is None:
            return differentiated_solve(matrix_data, rhs)
        _validate_operand(x0, b, mesh, axis_name, "x0")
        return differentiated_solve_x0(matrix_data, rhs, x0)

    def unpad_local_vector(value: jax.Array) -> jax.Array:
        _validate_operand(value, b, mesh, axis_name, "solver vector")
        return value.addressable_shards[0].data[:n_local]

    def solve_fn(
        rhs: jax.Array,
        x0: jax.Array | None = None,
        *,
        A_data: jax.Array | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        return sharded_solver(rhs, x0, A_data_override=A_data)

    return ShardedSolve(
        solve_fn,
        unpad_local_vector,
        nglobal,
        n_local,
    )
