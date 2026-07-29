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


class ShardedSolve:
    """Callable sharded solver with differentiable packed matrix values."""

    def __init__(
        self,
        solve_fn: Callable[..., tuple[jax.Array, ShardedInfo]],
        A_data: jax.Array,
        local_matrix_gradient_fn: Callable[[jax.Array], jsp.BCSR],
        local_vector_fn: Callable[[jax.Array], jax.Array],
        global_size: int,
        local_size: int,
    ) -> None:
        self._solve_fn = solve_fn
        self.A_data = A_data
        self._local_matrix_gradient_fn = local_matrix_gradient_fn
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
        return self._local_matrix_gradient_fn(gradient)

    def local_vector(self, value: jax.Array) -> jax.Array:
        """Return this rank's unpadded portion of a solver vector."""
        return self._local_vector_fn(value)


def make_sharded_vector(
    local_values: Any,
    *,
    comm: Comm,
    mesh: Mesh,
    global_size: int | None = None,
    axis_name: str = "rank",
) -> jax.Array:
    """Create a row-sharded vector, padding unequal local partitions.

    The returned JAX array has equal physical shard sizes, as required by
    ``NamedSharding``. ``make_sharded_solver`` ignores each shard's padding and
    uses the true row counts from ``A_local``.

    Args:
        local_values: This rank's unpadded one-dimensional values.
        comm: MPI communicator whose rank order matches ``mesh``.
        mesh: One-dimensional JAX device mesh with one device per MPI rank.
        global_size: Optional true global length. When provided, it is checked
            against the sum of local lengths.
        axis_name: Mesh axis used to partition the vector.

    Returns:
        A global JAX array with ``NamedSharding(mesh, P(axis_name))``. Its
        physical length is ``comm.size * max(local_sizes)``.
    """
    values = np.asarray(local_values)
    if values.ndim != 1:
        raise ValueError(f"local_values must be one-dimensional; got {values.shape}")
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

    local_sizes = tuple(int(size) for size in comm.allgather(values.shape[0]))
    inferred_global_size = sum(local_sizes)
    if global_size is not None and int(global_size) != inferred_global_size:
        raise ValueError(
            f"global_size is {global_size}, but local sizes sum to "
            f"{inferred_global_size}"
        )

    max_local_size = max(local_sizes)
    padded_values = np.zeros(max_local_size, dtype=values.dtype)
    padded_values[: values.shape[0]] = values
    sharding = NamedSharding(mesh, P(axis_name))
    return jax.make_array_from_process_local_data(
        sharding,
        padded_values,
        global_shape=(comm_size * max_local_size,),
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
    if b.ndim != 1:
        raise ValueError(f"b must be one-dimensional; got shape {b.shape}")

    sharding = getattr(b, "sharding", None)
    expected_spec = P(axis_name)
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
    expected_shape = (comm.Get_size() * max_local_size,)
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
    if len(index) != 1 or not isinstance(index[0], slice):
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
        if sharding.mesh != mesh or sharding.spec != P(axis_name):
            raise ValueError(
                f"{name} must use the same NamedSharding as the template RHS"
            )


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

    data = np.asarray(A.data)
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

    send_data = np.ascontiguousarray(data[send_order])
    send_rows = np.ascontiguousarray(indices[send_order])
    send_cols = np.ascontiguousarray(source_rows[send_order])
    packed_ids = comm.Get_rank() * max_nnz + np.arange(len(data), dtype=np.int64)
    send_ids = np.ascontiguousarray(packed_ids[send_order])
    recv_nnz = int(recv_counts.sum())
    recv_data = np.empty(recv_nnz, dtype=data.dtype)
    recv_rows = np.empty(recv_nnz, dtype=np.int64)
    recv_cols = np.empty(recv_nnz, dtype=np.int64)
    recv_ids = np.empty(recv_nnz, dtype=np.int64)
    value_mpi_type = MPI.FLOAT if data.dtype == np.float32 else MPI.DOUBLE

    comm.Alltoallv(
        [send_data, send_counts, send_displs, value_mpi_type],
        [recv_data, recv_counts, recv_displs, value_mpi_type],
    )
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
    recv_data = recv_data[order]
    recv_ids = recv_ids[order]
    row_counts = np.bincount(local_rows, minlength=n_local)
    transpose_indptr = np.concatenate(([0], np.cumsum(row_counts))).astype(np.int32)

    with temp_enable_x64():
        transpose_indices = jnp.asarray(recv_cols, dtype=jnp.int64)
        transpose = jsp.BCSR(
            (
                jnp.asarray(recv_data),
                transpose_indices,
                jnp.asarray(transpose_indptr),
            ),
            shape=(n_local, n_global),
        )
    return transpose, recv_ids


def make_sharded_solver(
    A_local: MatrixOrOperator,
    b: jax.Array,
    *,
    comm: Comm,
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
            global column indices.
        b: Global one-dimensional JAX array using
            ``NamedSharding(mesh, PartitionSpec(axis_name))``. For unequal row
            counts, construct it with :func:`make_sharded_vector`; physical
            shards are padded to the largest local partition.
        comm: MPI communicator whose rank order matches the JAX process order.
        mesh: One-dimensional JAX device mesh. If omitted, use the mesh from
            ``b.sharding``.
        axis_name: Name of the mesh axis that partitions rows.
        config: AmgX configuration. Defaults to the JAX-AMG MPI configuration.
        is_symmetric: Whether the global matrix is symmetric. When ``False``
            (the default), the distributed transpose is prepared once during
            solver creation for the reverse-mode adjoint.
        block_dim: AmgX block dimension.
        reuse_setup: Reuse the cached AmgX hierarchy across solves with the
            same sparsity pattern.

    Returns:
        A callable ``solver(b, x0=None, *, A_data=None)``. ``solver.A_data`` is the
        global sharded array of packed CSR values, padded to the largest local
        nonzero count. ``solver.local_matrix_gradient(gradient)`` converts its
        gradient back to this rank's unpadded BCSR structure, while
        ``solver.local_vector(value)`` removes vector padding. The solve returns
        a global sharded solution and an info dictionary. Info values are global
        arrays with one entry per rank; ``residual_history`` has shape
        ``(nranks, max_iters + 1)``.
    """
    _require_shard_map()
    if not isinstance(b, jax.Array):
        raise TypeError("b must be a global jax.Array")
    mesh = _resolve_mesh(b, mesh, axis_name)
    _validate_runtime(comm, mesh, axis_name)

    matrix_shape = getattr(A_local, "shape", None)
    if matrix_shape is None or len(matrix_shape) != 2:
        raise ValueError("A_local must have a two-dimensional shape")
    n_local = int(matrix_shape[0])
    nglobal = int(matrix_shape[1])
    partition_info, max_n_local = _local_partition(
        b, mesh, axis_name, comm, n_local, nglobal
    )

    block_dim = int(block_dim)
    _validate_matrix(A_local, nglobal, partition_info, block_dim)

    local_rhs = b.addressable_shards[0].data[:n_local]
    A_bcsr = to_bcsr_matrix(A_local, b=local_rhs, use_int64_indices=True)
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

    row_spec = P(axis_name)
    A_data_sharding = NamedSharding(mesh, row_spec)
    local_nnz = int(A_bcsr.data.shape[0])
    max_nnz = int(mpi_cache["max_nnz"])
    local_packed_data = np.zeros(max_nnz, dtype=np.asarray(A_bcsr.data).dtype)
    local_packed_data[:local_nnz] = np.asarray(A_bcsr.data)
    A_data = jax.make_array_from_process_local_data(
        A_data_sharding,
        local_packed_data,
        global_shape=(comm.Get_size() * max_nnz,),
    )

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

    info_specs = {
        "iterations": row_spec,
        "residual": row_spec,
        "status": row_spec,
        "residual_history": P(axis_name, None),
    }

    def pack_info(info: ShardedInfo) -> ShardedInfo:
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
        return jnp.pad(value, (0, max_n_local - n_local))

    def local_solve(
        A_data_local: jax.Array, rhs_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_bcsr, mpi_cache, is_symmetric)
        x_local, info = solve(
            A_dynamic,
            rhs_local[:n_local],
            block_dim=block_dim,
            reuse_setup=reuse_setup,
        )
        return pad_local_vector(x_local), pack_info(info)

    def local_solve_x0(
        A_data_local: jax.Array, rhs_local: jax.Array, x0_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_bcsr, mpi_cache, is_symmetric)
        x_local, info = solve(
            A_dynamic,
            rhs_local[:n_local],
            x0=x0_local[:n_local],
            block_dim=block_dim,
            reuse_setup=reuse_setup,
        )
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
        adjoint_local, _ = solve(
            A_adjoint,
            g_local[:n_local],
            block_dim=block_dim,
            reuse_setup=reuse_setup,
        )
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
        x_ghost = jnp.zeros(max_n_ghost + 1, dtype=x_local.dtype)
        x_ghost = x_ghost.at[recv_ghost_slot_local.reshape(-1)].set(
            recv_buffer.reshape(-1)
        )
        return jnp.concatenate([x_local, x_ghost[:max_n_ghost]])

    def local_matrix_gradient(
        A_data_local: jax.Array,
        x_at_columns: jax.Array,
        adjoint_local: jax.Array,
    ) -> jax.Array:
        del A_data_local
        grad_values = -adjoint_local[row_indices] * x_at_columns
        return jnp.where(valid_values, grad_values, 0)

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
        grad_A_data_local = local_matrix_gradient(
            A_data_ordered,
            x_at_columns,
            adjoint_local,
        )
        return pad_local_vector(adjoint_local), grad_A_data_local

    mapped_solve = jax.jit(
        jax.shard_map(
            local_solve,
            mesh=mesh,
            in_specs=(row_spec, row_spec),
            out_specs=(row_spec, info_specs),
        )
    )
    mapped_solve_x0 = jax.jit(
        jax.shard_map(
            local_solve_x0,
            mesh=mesh,
            in_specs=(row_spec, row_spec, row_spec),
            out_specs=(row_spec, info_specs),
        )
    )

    def local_cached_matrix_data(rhs_local: jax.Array) -> jax.Array:
        del rhs_local
        return jnp.asarray(local_packed_data)

    mapped_cached_matrix_data = jax.jit(
        jax.shard_map(
            local_cached_matrix_data,
            mesh=mesh,
            in_specs=(row_spec,),
            out_specs=row_spec,
        )
    )
    mapped_gather_matrix_columns = jax.jit(
        jax.shard_map(
            local_gather_matrix_columns,
            mesh=mesh,
            in_specs=(row_spec,),
            out_specs=row_spec,
        )
    )
    mapped_backward = jax.jit(
        jax.shard_map(
            local_backward,
            mesh=mesh,
            in_specs=(row_spec, row_spec, row_spec),
            out_specs=(row_spec, row_spec),
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

    def differentiated_solve_bwd(residuals, cotangents):
        matrix_data, x = residuals
        g_x, _ = cotangents
        x_at_columns = mapped_gather_matrix_columns(x)
        adjoint, grad_A_data = mapped_backward(matrix_data, x_at_columns, g_x)
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
        x_at_columns = mapped_gather_matrix_columns(x)
        adjoint, grad_A_data = mapped_backward(matrix_data, x_at_columns, g_x)
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

    def unpad_matrix_gradient(gradient: jax.Array) -> jsp.BCSR:
        _validate_operand(gradient, A_data, mesh, axis_name, "A_data gradient")
        local_gradient = gradient.addressable_shards[0].data[:local_nnz]
        return jsp.BCSR(
            (local_gradient, A_bcsr.indices, A_bcsr.indptr), shape=A_bcsr.shape
        )

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
        A_data,
        unpad_matrix_gradient,
        unpad_local_vector,
        nglobal,
        n_local,
    )


def solve_sharded(
    A_local: MatrixOrOperator,
    b: jax.Array,
    x0: jax.Array | None = None,
    *,
    comm: Comm,
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
