"""JAX sharding integration for distributed AmgX solves.

This module provides an additive interface on top of JAX-AMG's MPI backend.
JAX owns the global arrays and ``shard_map`` execution, while AmgX continues
to use one MPI rank per GPU for the distributed solve.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .cache import cache_mpi_metadata, with_cache
from .jaxamg import solve
from .utils import MatrixOrOperator, temp_enable_x64, to_bcsr_matrix

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

ShardedInfo = dict[str, jax.Array]


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
        local_packed_data: jax.Array,
        comm: Comm,
        mesh: Mesh,
        axis_name: str,
        partition_info: tuple[int, int],
        max_local_size: int,
    ) -> None:
        self._local_bcsr = local_bcsr
        self._local_packed_data = local_packed_data
        self._comm = comm
        self.mesh = mesh
        self.axis_name = axis_name
        self.partition_info = partition_info
        self.max_local_size = max_local_size
        self.data = data
        self.global_size = int(local_bcsr.shape[1])
        self.local_size = int(local_bcsr.shape[0])
        self.local_nnz = int(local_bcsr.data.shape[0])
        self.max_local_nnz = int(local_packed_data.shape[0])
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
        matrix: ShardedMatrix,
        local_vector_fn: Callable[[jax.Array], jax.Array],
        global_size: int,
        local_size: int,
    ) -> None:
        self._solve_fn = solve_fn
        self.matrix = matrix
        # Backward-compatible alias for the matrix container's sharded values.
        self.A_data = matrix.data
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
        """Solve with the cached values or a differentiable ``A_data`` operand."""
        return self._solve_fn(b, x0, A_data=A_data)

    def local_matrix_gradient(self, gradient: jax.Array) -> jsp.BCSR:
        """Convert a packed global value gradient to this rank's local BCSR."""
        return self.matrix.local_matrix(gradient)

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


def _local_partition(
    b: jax.Array,
    mesh: Mesh,
    axis_name: str,
    comm: Comm,
    n_local: int,
    n_global: int,
) -> tuple[tuple[int, int], int]:
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

    local_sizes = tuple(int(size) for size in comm.allgather(n_local))
    if sum(local_sizes) != n_global:
        raise ValueError(
            "the distributed matrix row counts must sum to its global column "
            f"count; got row counts {local_sizes} and {n_global} columns"
        )
    max_local_size = max(local_sizes)
    expected_shape = (comm.Get_size() * max_local_size, *b.shape[1:])
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

    comm_rank = comm.Get_rank()
    row_start = sum(local_sizes[:comm_rank])
    return (row_start, row_start + n_local), max_local_size


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
        local_packed_data,
        comm,
        mesh,
        axis_name,
        partition_info,
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
    partition_info, max_local_size = _local_partition(
        b, mesh, axis_name, comm, n_local, nglobal
    )
    A_bcsr = _normalize_local_matrix(A_local, b, partition_info)
    return _pack_sharded_matrix(
        A_bcsr,
        comm,
        mesh,
        axis_name,
        partition_info,
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
    A: jsp.BCSR,
    recvcounts: tuple[int, ...],
    partition_info: tuple[int, int],
    max_nnz: int,
    comm: Comm,
) -> tuple[jsp.BCSR, np.ndarray]:
    """Transpose fixed CSR structure and return its packed value-source map."""
    from mpi4py import MPI

    row_start, row_end = partition_info
    n_local = row_end - row_start
    n_global = sum(recvcounts)
    nranks = len(recvcounts)

    indices = np.asarray(A.indices, dtype=np.int64)
    indptr = np.asarray(A.indptr, dtype=np.int64)
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

    send_rows = np.ascontiguousarray(indices[send_order])
    send_cols = np.ascontiguousarray(source_rows[send_order])
    packed_ids = comm.Get_rank() * max_nnz + np.arange(len(indices), dtype=np.int64)
    send_ids = np.ascontiguousarray(packed_ids[send_order])
    recv_nnz = int(recv_counts.sum())
    recv_rows = np.empty(recv_nnz, dtype=np.int64)
    recv_cols = np.empty(recv_nnz, dtype=np.int64)
    recv_ids = np.empty(recv_nnz, dtype=np.int64)
    comm.Alltoallv(
        [send_rows, send_counts, send_displs, MPI.INT64_T],
        [recv_rows, recv_counts, recv_displs, MPI.INT64_T],
    )
    comm.Alltoallv(
        [send_cols, send_counts, send_displs, MPI.INT64_T],
        [recv_cols, recv_counts, recv_displs, MPI.INT64_T],
    )
    comm.Alltoallv(
        [send_ids, send_counts, send_displs, MPI.INT64_T],
        [recv_ids, recv_counts, recv_displs, MPI.INT64_T],
    )

    local_rows = recv_rows - row_start
    order = np.lexsort((recv_cols, local_rows))
    local_rows = local_rows[order]
    recv_cols = recv_cols[order]
    recv_ids = recv_ids[order]
    row_counts = np.bincount(local_rows, minlength=n_local)
    transpose_indptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int32)

    with temp_enable_x64():
        transpose_data = jnp.zeros(recv_nnz, dtype=A.data.dtype)
        transpose_indices = jnp.asarray(recv_cols, dtype=jnp.int64)
        transpose = jsp.BCSR(
            (
                transpose_data,
                transpose_indices,
                jnp.asarray(transpose_indptr),
            ),
            shape=(n_local, n_global),
        )
    return transpose, recv_ids


def make_sharded_solver(
    A_local: MatrixOrOperator | ShardedMatrix,
    b: jax.Array,
    *,
    comm: Comm | None = None,
    mesh: Mesh | None = None,
    axis_name: str = "rank",
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
    multi-process job. The CSR structure is captured as static per-process state,
    while the packed values are available as ``solver.A_data``. Pass that array
    back through ``solver(..., A_data=A_data)`` to differentiate matrix values.
    Use ``jax.set_mesh(mesh)`` around outer transforms such as ``jax.grad`` that
    consume the solver's global output.

    Args:
        A_local: Local row partition with shape ``(n_local, n_global)`` and
            global column indices, or a :class:`ShardedMatrix` created with
            :func:`make_sharded_matrix`.
        b: Global JAX array with shape ``(n_global,)`` or
            ``(n_global, nrhs)`` using row ``NamedSharding``. For unequal row
            counts, construct it with :func:`make_sharded_vector`; physical
            shards are padded to the largest local partition. Batched RHS
            columns are replicated within each row shard.
        comm: MPI communicator whose rank order matches the JAX process order.
            Defaults to ``MPI.COMM_WORLD``.
        mesh: One-dimensional JAX device mesh. If omitted, use the mesh from
            ``b.sharding``.
        axis_name: Name of the mesh axis that partitions rows.
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
        A callable ``solver(b, x0=None, *, A_data=None)``. ``solver.A_data`` is the
        global sharded array of packed CSR values, padded to the largest local
        nonzero count. ``solver.local_matrix_gradient(gradient)`` converts its
        gradient back to this rank's unpadded BCSR structure, while
        ``solver.local_vector(value)`` removes vector padding. The solve returns
        a global sharded solution and an info dictionary. For a vector RHS,
        info values have one entry per rank. For a batched RHS they have shape
        ``(nranks, nrhs)`` and ``residual_history`` has an additional trailing
        ``max_iters + 1`` axis.
    """
    _require_shard_map()
    if not isinstance(b, jax.Array):
        raise TypeError("b must be a global jax.Array")
    if isinstance(A_local, ShardedMatrix) and comm is None:
        comm = A_local._comm
    comm = _resolve_comm(comm)
    mesh = _resolve_mesh(b, mesh, axis_name)
    _validate_runtime(comm, mesh, axis_name)

    matrix_shape = (
        A_local.local_shape
        if isinstance(A_local, ShardedMatrix)
        else getattr(A_local, "shape", None)
    )
    if matrix_shape is None or len(matrix_shape) != 2:
        raise ValueError("A_local must have a two-dimensional shape")
    n_local = int(matrix_shape[0])
    nglobal = int(matrix_shape[1])
    partition_info, max_n_local = _local_partition(
        b, mesh, axis_name, comm, n_local, nglobal
    )

    block_dim = int(block_dim)
    is_batched = b.ndim == 2
    local_rhs = b.addressable_shards[0].data[:n_local]
    matrix_probe_rhs = local_rhs[:, 0] if is_batched else local_rhs
    sharded_matrix: ShardedMatrix | None
    if isinstance(A_local, ShardedMatrix):
        sharded_matrix = A_local
        if sharded_matrix.mesh != mesh or sharded_matrix.axis_name != axis_name:
            raise ValueError(
                "the sharded matrix and RHS must use the same mesh and axis name"
            )
        if (
            sharded_matrix.partition_info != partition_info
            or sharded_matrix.max_local_size != max_n_local
        ):
            raise ValueError(
                "the sharded matrix partition does not match the current RHS"
            )
        normalized_matrix = to_bcsr_matrix(
            sharded_matrix._local_bcsr,
            b=matrix_probe_rhs,
            use_int64_indices=True,
        )
        if normalized_matrix.data.dtype != sharded_matrix.data.dtype:
            raise ValueError(
                "the sharded matrix dtype is incompatible with b; rebuild it "
                "with make_sharded_matrix(A_local, b)"
            )
        A_bcsr = sharded_matrix._local_bcsr
    else:
        sharded_matrix = None
        A_bcsr = _normalize_local_matrix(A_local, b, partition_info)

    _validate_matrix(A_bcsr, nglobal, partition_info, block_dim)
    mpi_cache = cache_mpi_metadata(
        config or {},
        comm,
        nglobal,
        partition_info,
        A_bcsr,
        is_symmetric=is_symmetric,
        block_dim=block_dim,
    )
    # cache_mpi_metadata's lrank preserves the existing MPI API's device
    # selection. In sharded mode JAX has already selected this process's one
    # mesh-local GPU, so pass its CUDA-local hardware ordinal to AmgX.
    local_device = mesh.local_devices[0]
    local_hardware_id = getattr(local_device, "local_hardware_id", local_device.id)
    mpi_cache["lrank"] = int(local_hardware_id)

    rhs_spec = _row_partition_spec(b.ndim, axis_name)
    A_data_spec = P(axis_name)
    max_nnz = int(mpi_cache["max_nnz"])
    if sharded_matrix is None:
        sharded_matrix = _pack_sharded_matrix(
            A_bcsr,
            comm,
            mesh,
            axis_name,
            partition_info,
            max_n_local,
            max_nnz=max_nnz,
        )
    elif max_nnz != sharded_matrix.max_local_nnz:
        raise RuntimeError(
            "matrix packing and MPI cache disagree about the maximum local nnz"
        )
    local_nnz = sharded_matrix.local_nnz
    local_packed_data = sharded_matrix._local_packed_data
    A_data = sharded_matrix.data

    if is_symmetric:
        A_transpose = None
        transpose_cache = None
        transpose_source_ids = None
    else:
        A_transpose, source_ids = _transpose_distributed_matrix(
            A_bcsr,
            mpi_cache["recvcounts_tuple"],
            partition_info,
            max_nnz,
            comm,
        )
        transpose_cache = cache_mpi_metadata(
            config or {},
            comm,
            nglobal,
            partition_info,
            A_transpose,
            is_symmetric=False,
            block_dim=block_dim,
        )
        transpose_cache["lrank"] = int(local_hardware_id)
        transpose_source_ids = jnp.asarray(source_ids, dtype=jnp.int32)

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
        template: jsp.BCSR,
        cache: dict[str, Any],
        symmetric: bool,
    ) -> jsp.BCSR:
        matrix = jsp.BCSR(
            (data_local[: template.data.shape[0]], template.indices, template.indptr),
            shape=template.shape,
        )
        return with_cache(matrix, mpi=cache, is_symmetric=symmetric)

    def pad_local_vector(value: jax.Array) -> jax.Array:
        padding = ((0, max_n_local - n_local),) + ((0, 0),) * (value.ndim - 1)
        return jnp.pad(value, padding)

    def solve_one_rhs(
        A_dynamic: jsp.BCSR,
        rhs_local: jax.Array,
        x0_local: jax.Array | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        return solve(
            A_dynamic,
            rhs_local[:n_local],
            x0=None if x0_local is None else x0_local[:n_local],
            block_dim=block_dim,
            reuse_setup=reuse_setup,
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
            )
            solutions.append(solution)
            column_info.append(info)
        return jnp.stack(solutions, axis=1), jax.tree.map(
            lambda *values: jnp.stack(values), *column_info
        )

    def local_solve(
        A_data_local: jax.Array, rhs_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_bcsr, mpi_cache, is_symmetric)
        x_local, info = solve_local_rhs(A_dynamic, rhs_local)
        return pad_local_vector(x_local), pack_info(info)

    def local_solve_x0(
        A_data_local: jax.Array, rhs_local: jax.Array, x0_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_bcsr, mpi_cache, is_symmetric)
        x_local, info = solve_local_rhs(A_dynamic, rhs_local, x0_local)
        return pad_local_vector(x_local), pack_info(info)

    def local_adjoint(A_data_local: jax.Array, g_local: jax.Array) -> jax.Array:
        if is_symmetric:
            A_adjoint = matrix_with_data(
                A_data_local, A_bcsr, mpi_cache, symmetric=True
            )
        else:
            assert A_transpose is not None
            assert transpose_cache is not None
            assert transpose_source_ids is not None
            all_A_data = jax.lax.all_gather(A_data_local, axis_name, axis=0, tiled=True)
            transpose_data = all_A_data[transpose_source_ids]
            A_adjoint = matrix_with_data(
                transpose_data, A_transpose, transpose_cache, symmetric=False
            )
        adjoint_local, _ = solve_one_rhs(A_adjoint, g_local)
        return adjoint_local

    halo_plan = mpi_cache["halo_plan"]
    max_n_ghost = max(comm.allgather(halo_plan.n_ghost))
    local_row_indices = np.repeat(
        np.arange(A_bcsr.shape[0], dtype=np.int32),
        np.diff(np.asarray(A_bcsr.indptr, dtype=np.int64)),
    )
    padded_row_indices = np.zeros(max_nnz, dtype=np.int32)
    padded_row_indices[:local_nnz] = local_row_indices
    padded_col_to_combined = np.zeros(max_nnz, dtype=np.int32)
    padded_col_to_combined[:local_nnz] = halo_plan.col_to_combined
    local_valid_values = np.arange(max_nnz) < local_nnz

    # Keep rank-local halo metadata inside the shard_map bodies. Making these
    # arrays global and closing over them in the custom VJP prevents an outer
    # multi-process jax.jit from lowering because their remote shards are not
    # addressable by the current process.
    row_indices = jnp.asarray(padded_row_indices)
    col_to_combined = jnp.asarray(padded_col_to_combined)
    valid_values = jnp.asarray(local_valid_values)
    send_ids = jnp.asarray(halo_plan.send_ids_2d)
    recv_ghost_slot = jnp.asarray(halo_plan.recv_ghost_slot_2d)

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

    def local_gather_matrix_columns(x_local: jax.Array) -> jax.Array:
        x_combined = gather_solution_halo(x_local[:n_local], send_ids, recv_ghost_slot)
        return jnp.where(valid_values, x_combined[col_to_combined], 0)

    def local_backward(
        A_data_local: jax.Array,
        x_at_columns: jax.Array,
        g_local: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        A_data_ordered, g_ordered, x_at_columns = jax.lax.optimization_barrier(
            (A_data_local, g_local, x_at_columns)
        )
        adjoint_local = local_adjoint(A_data_ordered, g_ordered)
        grad_values = -adjoint_local[row_indices] * x_at_columns
        grad_A_data_local = jnp.where(valid_values, grad_values, 0)
        return pad_local_vector(adjoint_local), grad_A_data_local

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

    def local_cached_matrix_data(rhs_local: jax.Array) -> jax.Array:
        del rhs_local
        return jnp.asarray(local_packed_data)

    mapped_cached_matrix_data = jax.jit(
        jax.shard_map(
            local_cached_matrix_data,
            mesh=mesh,
            in_specs=(rhs_spec,),
            out_specs=A_data_spec,
        )
    )
    scalar_spec = P(axis_name)
    mapped_gather_matrix_columns = jax.jit(
        jax.shard_map(
            local_gather_matrix_columns,
            mesh=mesh,
            in_specs=(scalar_spec,),
            out_specs=scalar_spec,
        )
    )
    mapped_backward = jax.jit(
        jax.shard_map(
            local_backward,
            mesh=mesh,
            in_specs=(A_data_spec, scalar_spec, scalar_spec),
            out_specs=(scalar_spec, A_data_spec),
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
        if not is_batched:
            x_at_columns = mapped_gather_matrix_columns(x)
            return mapped_backward(matrix_data, x_at_columns, g_x)

        adjoint_columns: list[jax.Array] = []
        matrix_gradients: list[jax.Array] = []
        for column in range(x.shape[1]):
            x_column = x[:, column]
            g_column = g_x[:, column]
            # Preserve the same collective ordering for adjoint solves and
            # their preceding halo exchanges.
            if adjoint_columns:
                x_column, g_column, _ = jax.lax.optimization_barrier(
                    (x_column, g_column, adjoint_columns[-1])
                )
            x_at_columns = mapped_gather_matrix_columns(x_column)
            adjoint, matrix_gradient = mapped_backward(
                matrix_data, x_at_columns, g_column
            )
            adjoint_columns.append(adjoint)
            matrix_gradients.append(matrix_gradient)
        return jnp.stack(adjoint_columns, axis=1), jnp.sum(
            jnp.stack(matrix_gradients), axis=0
        )

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
            # A non-addressable global array cannot be captured as a constant
            # by an outer multi-process jax.jit. Materialize the same cached
            # rank-local values through shard_map while tracing instead.
            matrix_data = (
                mapped_cached_matrix_data(rhs)
                if isinstance(rhs, jax.core.Tracer)
                else A_data
            )
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
        sharded_matrix,
        unpad_local_vector,
        nglobal,
        n_local,
    )


def solve_sharded(
    A_local: MatrixOrOperator | ShardedMatrix,
    b: jax.Array,
    x0: jax.Array | None = None,
    *,
    comm: Comm | None = None,
    mesh: Mesh | None = None,
    axis_name: str = "rank",
    config: dict[str, Any] | None = None,
    is_symmetric: bool = False,
    block_dim: int = 1,
    reuse_setup: bool = False,
) -> tuple[jax.Array, ShardedInfo]:
    """Solve ``Ax=b`` while preserving JAX global-array sharding.

    This convenience function creates a sharded solver and calls it once. For
    repeated solves, use :func:`make_sharded_solver` so MPI metadata and the
    compiled ``shard_map`` are reused.
    """
    solver = make_sharded_solver(
        A_local,
        b,
        comm=comm,
        mesh=mesh,
        axis_name=axis_name,
        config=config,
        is_symmetric=is_symmetric,
        block_dim=block_dim,
        reuse_setup=reuse_setup,
    )
    return solver(b, x0)
