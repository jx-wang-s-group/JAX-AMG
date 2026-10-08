"""JAX sharding integration for distributed AmgX solves.

This module provides an additive interface on top of JAX-AMG's MPI backend.
JAX owns the global arrays and ``shard_map`` execution, while AmgX continues
to use one MPI rank per GPU for the distributed solve.
"""

from __future__ import annotations

import functools
import os
import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from . import config as amgx_config
from .jaxamg import _capture_and_save_stats
from .linear_map import LinearMap
from .mpi_utils import (
    build_halo_plan,
    build_transpose_plan,
    register_comm,
)
from .mpi_utils import transpose_values as mpi_transpose_values
from .nullspace import (
    _DENSE_LU_MSG,
    _MISSING_NULLSPACE_MSG,
    _MISSING_TRANSPOSE_MSG,
    NullSpaceWarning,
    as_labels,
    as_nullspace_basis,
    label_scale_stats,
    label_sums,
    remove_label_sums,
    scale_label_basis,
    unit_bases,
    validate_basis,
    validate_label_values,
)
from .transport import halo_gather
from .utils import (
    MatrixOrOperator,
    get_preferred_dtype,
    to_bcsr_matrix,
)

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

# Private API; tells whether XLA_FLAGS can still take effect.
try:
    from jax._src.xla_bridge import (
        backends_are_initialized as _backends_are_initialized,
    )
except Exception:  # pragma: no cover - exercised only if the private API moves

    def _backends_are_initialized() -> bool:
        return True  # never report a flag as effective without knowing


def _disable_shard_autotuning() -> bool:
    """Disable XLA's cross-process sharded autotuning; return whether it is off.

    That autotuning assumes every process compiles the identical program and
    can deadlock when they differ (a local or halo operator override).
    Disabling it only affects compile time. XLA reads
    ``XLA_FLAGS`` when its backend initializes, so an explicit setting is
    respected and a late import cannot take effect.
    """
    flags = os.environ.get("XLA_FLAGS", "")
    if "xla_gpu_shard_autotuning" in flags:
        return "xla_gpu_shard_autotuning=false" in flags.lower()
    if _backends_are_initialized():
        return False
    os.environ["XLA_FLAGS"] = f"{flags} --xla_gpu_shard_autotuning=false".strip()
    return True


_SHARD_AUTOTUNING_DISABLED = _disable_shard_autotuning()


def _check_shard_autotuning() -> None:
    if jax.process_count() > 1 and not _SHARD_AUTOTUNING_DISABLED:
        warnings.warn(
            "XLA's sharded autotuning is enabled, which can deadlock a "
            "multi-process sharded solve. Import jaxamg before the first JAX "
            "device call, set XLA_FLAGS=--xla_gpu_shard_autotuning=false, or "
            "pass compiler_options={'xla_gpu_shard_autotuning': False} to the "
            "outer jax.jit.",
            stacklevel=3,
        )


ShardedInfo = dict[str, jax.Array]


class _CSRStructure(NamedTuple):
    """Rank-local CSR structure without an otherwise unused values buffer."""

    indices: jax.Array
    indptr: jax.Array
    shape: tuple[int, int]
    nnz: int


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


class ShardedMatrix:
    """Distributed CSR matrix created by :func:`make_sharded_matrix`."""

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
        coloring: tuple | None = None,
        nullspace: jax.Array | None = None,
        transpose_nullspace: jax.Array | None = None,
        labels: tuple[int, jax.Array] | None = None,
        label_sum: Callable[[jax.Array], jax.Array] | None = None,
    ) -> None:
        local_nnz = int(local_bcsr.data.shape[0])
        # Keep only the rank-local CSR structure. The matrix values live solely
        # in the padded shard of ``data``; ``local_matrix`` re-slices them on
        # demand, so no second per-rank value buffer is retained. These arrays
        # stay uncommitted: an array pinned to this process's single device
        # cannot be captured by a shard_map spanning the whole mesh.
        self._structure = _CSRStructure(
            jnp.asarray(local_bcsr.indices),
            jnp.asarray(local_bcsr.indptr),
            tuple(local_bcsr.shape),
            local_nnz,
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
        # Coloring of the source operator, if any, for materializing
        # operators with this sparsity.
        self._coloring = coloring
        # Null-space bases and labels of the local rows: each solve's
        # defaults.
        self._nullspace = nullspace
        self._transpose_nullspace = transpose_nullspace
        self._labels = labels
        self._label_sum = label_sum

    def local_matrix(self, data: jax.Array | None = None) -> jsp.BCSR:
        """Return this rank's unpadded BCSR matrix for ``data`` or cached values.

        This is an eager helper: it reads the addressable shard of the packed
        values (typically to unpack a computed matrix gradient), so it cannot
        be applied to traced values inside ``jax.jit`` or another JAX
        transformation.
        """
        values = self.data if data is None else data
        _validate_operand(values, self.data, self.mesh, self.axis_name, "matrix data")
        # The caller may be inside jax.set_mesh over the whole mesh, which
        # rejects single-device slicing.
        with jax.set_mesh(_local_mesh(self.mesh, self.axis_name)):
            local_values = values.addressable_shards[0].data[: self.local_nnz]
        return jsp.BCSR(
            (local_values, self._structure.indices, self._structure.indptr),
            shape=self._structure.shape,
        )


class ShardedSolve:
    """Callable sharded solver created by :func:`make_sharded_solver`."""

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
        A: MatrixOrOperator | jax.Array | None = None,
        save_stats_file: str | os.PathLike | None = None,
        nullspace: Any = None,
        transpose_nullspace: Any = None,
        labels: Any = None,
        label_sum: Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        """Solve with the cached values or an explicit ``A``.

        ``A`` is this rank's ``(n_local, n_global)`` operator or matrix with the
        sparsity fixed at solver creation, as ``solve(A, b)`` takes it, or the
        packed global values in the layout of ``ShardedMatrix.data``. Operator
        parameters must be identical on every rank and receive the gradient of
        the global loss; packed values receive their per-entry gradient. ``A``
        is required inside a JAX transformation. ``save_stats_file`` writes
        AmgX statistics after a direct call (rank 0 writes the file; requires
        ``save_stats=True`` at creation).

        ``nullspace``, ``transpose_nullspace`` and ``labels`` (as for
        ``solve``, here global arrays sharded like ``b``: ``"constant"``, a
        vector or ``(rows, k)`` bases, and ``(count, labels)``) replace, for
        this solve, those attached to the matrix (the defaults); changing them
        reuses the AmgX setup. ``label_sum`` reduces the
        labels' partial sums when the labels span ranks in the caller's own
        numbering: a linear function of a global array holding each rank's
        ``count`` rows of partial sums (sharded like ``b``), returning their
        totals in the same layout. By default the labels are global numbers,
        summed by an all-reduce.
        """
        return self._solve_fn(
            b,
            x0,
            A=A,
            save_stats_file=save_stats_file,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            labels=labels,
            label_sum=label_sum,
        )

    def local_vector(self, value: jax.Array) -> jax.Array:
        """Return this rank's unpadded rows of a solver vector.

        This is an eager helper: it reads the vector's addressable shard, so
        it cannot be applied to traced values inside ``jax.jit`` or another
        JAX transformation.
        """
        return self._local_vector_fn(value)


def make_sharded_vector(
    local_values: Any,
    *,
    comm: Comm | None = None,
    mesh: Mesh | None = None,
    global_size: int | None = None,
    axis_name: str | tuple[str, ...] = "rank",
) -> jax.Array:
    """Create a row-sharded vector, padding unequal partitions.

    The returned JAX array has equal physical shard sizes, as required by
    ``NamedSharding``. ``make_sharded_solver`` ignores each shard's padding and
    uses the true row counts from ``A_local``. JAX array inputs are padded and
    assembled on device; other array-like inputs use a NumPy host staging path.

    Args:
        local_values: This rank's unpadded values with shape ``(n_local,)``.
        comm: MPI communicator spanning every JAX process, with rank order
            matching ``mesh``. Defaults to ``MPI.COMM_WORLD``.
        mesh: JAX device mesh with one device per MPI rank, in rank order.
            Its ``axis_name`` axes partition the rows; any other axis must have
            size one, so a program's multi-axis mesh (e.g. a Cartesian domain
            decomposition's) can be shared. Defaults to a one-dimensional mesh
            over the first ``comm.size`` JAX devices (all devices in a typical
            multi-process job).
        global_size: Optional true global length. When provided, it is checked
            against the sum of local lengths.
        axis_name: Mesh axis, or tuple of mesh axes in mesh order, partitioning
            the vector.

    Returns:
        A global JAX array whose axis uses ``P(axis_name)``. Its physical
        length is ``comm.size * max(local_sizes)``; values are cast to
        ``float32`` unless already ``float32`` or ``float64``.
    """
    comm = _resolve_comm(comm)
    axis_name = _axis_spec(axis_name)
    if mesh is None:
        if not isinstance(axis_name, str):
            raise ValueError("pass the mesh whose axes partition the rows")
        # One device per MPI rank. In a multi-process job this covers all JAX
        # devices; a process with extra local devices uses the leading ones.
        comm_size = comm.Get_size()
        mesh = jax.make_mesh(
            (comm_size,), (axis_name,), devices=jax.devices()[:comm_size]
        )
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
    if values.ndim != 1:
        raise ValueError(f"local_values must be one-dimensional; got {values.shape}")
    if values.dtype not in (jnp.float32, jnp.float64):
        # The solver's precision rule, so solutions share the RHS dtype.
        values = values.astype(jnp.float32)
    _row_axes(mesh, axis_name)

    comm_size = comm.Get_size()
    if mesh.size != comm_size or _row_count(mesh, axis_name) != comm_size:
        raise ValueError(
            "the mesh axis must contain exactly one device per MPI rank; got "
            f"mesh size {mesh.size} and communicator size {comm_size}"
        )

    local_shapes = tuple(tuple(shape) for shape in comm.allgather(values.shape))
    local_sizes = tuple(int(shape[0]) for shape in local_shapes)
    if min(local_sizes) == 0:
        raise ValueError(
            f"every rank must own at least one row; got local sizes {local_sizes}"
        )
    inferred_global_size = sum(local_sizes)
    if global_size is not None and int(global_size) != inferred_global_size:
        raise ValueError(
            f"global_size is {global_size}, but local sizes sum to "
            f"{inferred_global_size}"
        )

    max_local_size = max(local_sizes)
    if max_local_size == values.shape[0]:
        padded_values = values
    elif isinstance(values, jax.Array):
        padded_values = jnp.pad(values, (0, max_local_size - values.shape[0]))
    else:
        padded_values = np.pad(values, (0, max_local_size - values.shape[0]))
    sharding = NamedSharding(mesh, P(axis_name))
    return jax.make_array_from_process_local_data(
        sharding,
        padded_values,
        global_shape=(comm_size * max_local_size,),
    )


def _local_mesh(mesh: Mesh, axis_name: str | tuple[str, ...]) -> Mesh:
    """This process's mesh device as a one-device mesh with the mesh's axes."""
    del axis_name  # every axis of the mesh, each of size one
    names = tuple(mesh.axis_names)
    return jax.make_mesh((1,) * len(names), names, devices=[mesh.local_devices[0]])


def _axis_spec(axis_name: str | tuple[str, ...]) -> str | tuple[str, ...]:
    """The row axes as a PartitionSpec entry: a name, or a tuple of two or more."""
    if isinstance(axis_name, str):
        return axis_name
    names = tuple(axis_name)
    if not names or not all(isinstance(name, str) for name in names):
        raise ValueError(
            f"axis_name must be a mesh axis name or a tuple of names; got {axis_name!r}"
        )
    return names[0] if len(names) == 1 else names


def _row_axes(mesh: Mesh, axis_name: str | tuple[str, ...]) -> tuple[str, ...]:
    """The mesh axes partitioning rows, validated: distinct existing axes in
    mesh order, every other mesh axis of size one (so the flattened device
    order is the row-partition order)."""
    names = (axis_name,) if isinstance(axis_name, str) else tuple(axis_name)
    mesh_names = tuple(mesh.axis_names)
    if len(set(names)) != len(names) or any(name not in mesh_names for name in names):
        raise ValueError(
            f"the row axes {names!r} must be distinct axes of the mesh {mesh_names!r}"
        )
    if tuple(name for name in mesh_names if name in names) != names:
        raise ValueError(
            f"the row axes {names!r} must follow the mesh axis order {mesh_names!r}"
        )
    others = [
        name for name in mesh_names if name not in names and mesh.shape[name] != 1
    ]
    if others:
        raise ValueError(
            f"mesh axes {others!r} do not partition rows and must have size one"
        )
    return names


def _row_count(mesh: Mesh, axis_name: str | tuple[str, ...]) -> int:
    """Number of row partitions: the product of the row axes' sizes."""
    return int(
        np.prod([mesh.shape[name] for name in _row_axes(mesh, axis_name)], dtype=int)
    )


# Older releases default meshes to Auto axes, which drop the sharding spec of
# derived arrays; the interface relies on explicit sharding throughout.
def _local_basis(
    A_local: MatrixOrOperator, name: str, n_local: int, n_global: int, dtype: Any
) -> jax.Array | None:
    """Normalized ``(n_local, k)`` null-space basis attached to ``A_local``."""
    basis = as_nullspace_basis(
        getattr(A_local, f"_{name}", None), n_local, dtype, name, n_global
    )
    # Uncommitted, so a shard_map spanning the mesh can capture it.
    return None if basis is None else jnp.asarray(np.asarray(basis))


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

    _row_axes(mesh, axis_name)
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
    if mesh.size != comm_size or _row_count(mesh, axis_name) != comm_size:
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
    if comm_size > 1:
        # Heterogeneous models can get different compiler fusion/halo plans and
        # silently corrupt stencil results (JAX 0.9.1). Compare each rank's own
        # local model: remote device descriptions can repeat the querying one's.
        from mpi4py import MPI

        model = local_devices[0].device_kind
        root_model = comm.bcast(model if comm_rank == 0 else None, root=0)
        if comm.allreduce(int(model != root_model), op=MPI.MAX):
            raise ValueError(
                "the sharding interface requires the same GPU model on every MPI rank"
            )


def _validate_vector_layout(
    b: jax.Array,
    mesh: Mesh,
    axis_name: str,
    comm_size: int,
    max_local_size: int,
) -> None:
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

    expected_shape = (comm_size * max_local_size,)
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


def _local_partition(
    b: jax.Array,
    mesh: Mesh,
    axis_name: str,
    comm: Comm,
    n_local: int,
    n_global: int,
) -> tuple[tuple[int, int], tuple[int, ...], int]:
    local_sizes = tuple(int(size) for size in comm.allgather(n_local))
    if min(local_sizes) == 0:
        raise ValueError(
            f"every rank must own at least one row; got row counts {local_sizes}"
        )
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


def _local_matrix_shape(A_local: MatrixOrOperator) -> tuple[int, int]:
    """Resolve the ``(n_local, n_global)`` shape of a local matrix or operator.

    A matrix-free operator is a plain callable with no ``shape``, so its shape
    comes from the coloring cache that the distributed solve needs anyway.
    """
    shape = getattr(A_local, "shape", None)
    if shape is None and callable(A_local):
        coloring = getattr(A_local, "_coloring_info", None)
        if coloring is None:
            raise ValueError(
                "a callable A_local must carry cached coloring information so "
                "its shape is known; attach it with jaxamg.with_cache(op, "
                "coloring=jaxamg.cache_coloring(op, shape=(n_local, n_global)))"
            )
        shape = coloring[4]
    if shape is None or len(shape) != 2:
        raise ValueError("A_local must have a two-dimensional shape")
    return int(shape[0]), int(shape[1])


def _validate_matrix(
    A_local: MatrixOrOperator,
    nglobal: int,
    partition_info: tuple[int, int],
    block_dim: int,
) -> None:
    shape = _local_matrix_shape(A_local)

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
    nglobal = _local_matrix_shape(A_local)[1]
    _validate_matrix(A_local, nglobal, partition_info, block_dim=1)

    local_rhs = b.addressable_shards[0].data[:n_local]
    return to_bcsr_matrix(A_local, b=local_rhs, use_int64_indices=True)


def _pack_sharded_matrix(
    A_bcsr: jsp.BCSR,
    comm: Comm,
    mesh: Mesh,
    axis_name: str,
    partition_info: tuple[int, int],
    row_counts: tuple[int, ...],
    max_local_size: int,
    max_nnz: int | None = None,
    coloring: tuple | None = None,
    nullspace: jax.Array | None = None,
    transpose_nullspace: jax.Array | None = None,
    labels: tuple[int, jax.Array] | None = None,
    label_sum: Callable[[jax.Array], jax.Array] | None = None,
) -> ShardedMatrix:
    """Pack normalized local values into a global sharded array."""
    local_nnz = int(A_bcsr.data.shape[0])
    if max_nnz is None:
        from mpi4py import MPI

        max_nnz = int(comm.allreduce(local_nnz, op=MPI.MAX))
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
        coloring,
        nullspace,
        transpose_nullspace,
        labels,
        label_sum,
    )


def make_sharded_matrix(
    A_local: MatrixOrOperator,
    b: jax.Array,
    *,
    comm: Comm | None = None,
    mesh: Mesh | None = None,
    axis_name: str | tuple[str, ...] = "rank",
) -> ShardedMatrix:
    """Create a distributed CSR container without replicating the global matrix.

    The CSR column indices and row pointers remain rank-local static structure.
    Matrix values are packed into a global JAX array sharded over the same mesh
    axis as ``b``. Unequal local nonzero counts are padded to the largest count;
    the padding is ignored by solves and gradients.

    Args:
        A_local: This process's CSR row partition with shape
            ``(n_local, n_global)`` and global column indices. A matrix-free
            operator is also accepted, provided it carries cached coloring
            information (``jaxamg.with_cache(op, coloring=...)``) so its shape
            and sparsity pattern are known; it is materialized once here.
            Null-space bases and labels attached with ``with_cache(A_local,
            nullspace=..., transpose_nullspace=..., labels=..., label_sum=...)``
            (local rows) are kept as every solve's defaults.
        b: Global row-sharded RHS used to validate the matrix partition, mesh,
            and numerical dtype.
        comm: MPI communicator spanning every JAX process, with rank order
            matching the JAX process order. Defaults to ``MPI.COMM_WORLD``.
        mesh: JAX device mesh (see :func:`make_sharded_vector`). If omitted,
            use the mesh from ``b.sharding``.
        axis_name: Mesh axis, or tuple of mesh axes in mesh order, that
            partitions rows and packed values (as for ``b``).

    Returns:
        A :class:`ShardedMatrix` whose ``data`` attribute contains the global
        sharded values. The original global matrix is never materialized.
    """
    if not isinstance(b, jax.Array):
        raise TypeError("b must be a global jax.Array")
    comm = _resolve_comm(comm)
    axis_name = _axis_spec(axis_name)
    mesh = _resolve_mesh(b, mesh, axis_name)
    _validate_runtime(comm, mesh, axis_name)

    from .global_operator import GlobalOperator
    from .halo import HaloOperator

    if isinstance(A_local, GlobalOperator):
        # Probed once here with the distributed colouring; rows as a matrix.
        A_local = A_local._local_matrix(b)
    if isinstance(A_local, HaloOperator):
        from mpi4py import MPI

        # The global size is the sum of the owned rows (one O(1) reduction).
        n_local = A_local.n_local
        nglobal = int(comm.allreduce(n_local, op=MPI.SUM))
    else:
        n_local, nglobal = _local_matrix_shape(A_local)
    partition_info, row_counts, max_local_size = _local_partition(
        b, mesh, axis_name, comm, n_local, nglobal
    )
    if isinstance(A_local, HaloOperator):
        A_bcsr = A_local._local_matrix(
            partition_info[0], nglobal, b.dtype, traced=False
        )
    else:
        A_bcsr = _normalize_local_matrix(A_local, b, partition_info)
    dtype = A_bcsr.data.dtype
    return _pack_sharded_matrix(
        A_bcsr,
        comm,
        mesh,
        axis_name,
        partition_info,
        row_counts,
        max_local_size,
        coloring=(
            getattr(A_local, "_coloring_info", None) if callable(A_local) else None
        ),
        nullspace=_local_basis(A_local, "nullspace", n_local, nglobal, dtype),
        transpose_nullspace=_local_basis(
            A_local, "transpose_nullspace", n_local, nglobal, dtype
        ),
        labels=as_labels(getattr(A_local, "_labels", None), n_local, comm=comm),
        label_sum=getattr(A_local, "_label_sum", None),
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
            raise ValueError(f"{name} must use the same NamedSharding as its template")


def _global_operator_type():
    from .global_operator import GlobalOperator

    return GlobalOperator


def make_sharded_solver(
    A: ShardedMatrix,
    b: jax.Array,
    *,
    config: dict[str, Any] | None = None,
    is_symmetric: bool = False,
    block_dim: int = 1,
    reuse_setup: bool = False,
    save_stats: bool = False,
) -> ShardedSolve:
    """Create a solver for a globally sharded RHS.

    Wrap repeated solves in ``jax.jit``; an untransformed call runs the
    rank-local pipeline on this process's shard.

    This interface complements, rather than replaces, ``solve(..., comm=...)``.
    JAX manages the global input and output arrays through ``shard_map`` while
    the AmgX solve itself uses the supplied MPI communicator. The interface is
    experimental; it partitions rows over the matrix's mesh axes and requires
    one MPI process with one mesh-local GPU per rank and a communicator
    spanning every JAX process.

    ``jax.distributed.initialize()`` must be called before this function in a
    multi-process job. The matrix owns the communicator, mesh, local CSR
    structure, and globally sharded packed values. Pass ``A.data`` through
    ``solver(..., A=A.data)`` to differentiate matrix values. Use
    ``jax.set_mesh(A.mesh)`` around outer transforms such as ``jax.grad``.

    .. note::
        Importing jaxamg disables XLA's cross-process sharded autotuning,
        which can deadlock when the ranks' programs differ. Import it before
        the first JAX device call; see :doc:`sharding`.

    Args:
        A: Distributed matrix created with :func:`make_sharded_matrix`.
        b: Global row-sharded JAX vector. Construct it with
            :func:`make_sharded_vector`, which handles unequal row counts.
        config: AmgX configuration. Defaults to the JAX-AMG MPI configuration.
        is_symmetric: Whether the global matrix is symmetric. When ``False``
            (the default), the distributed transpose is prepared once during
            solver creation for the reverse-mode adjoint.
        block_dim: AmgX block dimension. The matrix retains its scalar CSR
            representation, and every local row partition must be divisible by
            this value.
        reuse_setup: Reuse the cached AmgX hierarchy across solves with the
            same sparsity pattern.
        save_stats: Prepare the AmgX configuration with solver-statistics
            output enabled so a later direct call with
            ``solver(..., save_stats_file=...)`` produces a complete stats
            file.

    Returns:
        A callable ``solver(b, x0=None, *, A=None, save_stats_file=None)``.
        ``A`` is this rank's operator or matrix with the sparsity fixed here
        (its closed-over parameters must be identical on every rank) or the
        packed global values ``A.data``; it may be omitted for a direct call
        but is required under a JAX transform. ``A.local_matrix(gradient)``
        unpacks a packed matrix gradient and ``solver.local_vector(value)``
        removes vector padding. Info values have one entry per rank.
    """
    if not isinstance(A, ShardedMatrix):
        raise TypeError("A must be created with jaxamg.make_sharded_matrix")
    if not isinstance(b, jax.Array):
        raise TypeError("b must be a global jax.Array")
    comm = A._comm
    mesh = A.mesh
    axis_name = A.axis_name
    _validate_runtime(comm, mesh, axis_name)
    _check_shard_autotuning()

    n_local = A.local_size
    nglobal = A.global_size
    partition_info = A.partition_info
    max_n_local = A.max_local_size
    _validate_vector_layout(b, mesh, axis_name, comm.Get_size(), max_n_local)

    block_dim = int(block_dim)
    A_structure = A._structure
    if get_preferred_dtype(A.data, b) != A.data.dtype:
        raise ValueError(
            "the sharded matrix dtype is incompatible with b; rebuild it "
            "with make_sharded_matrix(A_local, b)"
        )

    _validate_matrix(A_structure, nglobal, partition_info, block_dim)

    # Null spaces as in solve(): a symmetric matrix shares one basis.
    nullspace = A._nullspace
    transpose_nullspace = A._transpose_nullspace
    if transpose_nullspace is None and nullspace is not None:
        if is_symmetric:
            transpose_nullspace = nullspace
        else:
            warnings.warn(_MISSING_TRANSPOSE_MSG, NullSpaceWarning, stacklevel=2)
    elif nullspace is None and transpose_nullspace is not None:
        if is_symmetric:
            nullspace = transpose_nullspace
        else:
            warnings.warn(_MISSING_NULLSPACE_MSG, NullSpaceWarning, stacklevel=2)
    # Every rank must take the same basis-dependent branches and collectives.
    schema = tuple(
        None if basis is None else basis.shape[1]
        for basis in (nullspace, transpose_nullspace)
    )
    # Agreement by one O(1) reduction of (min, -max) over the encoded schema.
    from mpi4py import MPI

    encoded = np.array([-1 if c is None else c for c in schema], dtype=np.int64)
    bounds = np.concatenate([encoded, -encoded])
    comm.Allreduce(MPI.IN_PLACE, bounds, op=MPI.MIN)
    if np.any(bounds[:2] != -bounds[2:]):
        raise ValueError(
            "null-space bases must be declared on every rank with the same "
            "column counts; this rank's (nullspace, transpose_nullspace) "
            f"columns: {schema}"
        )
    labels = A._labels
    default_label_sum = A._label_sum
    if labels is not None and nullspace is None and transpose_nullspace is None:
        raise ValueError(
            "labels apply to the declared nullspace/transpose_nullspace columns; "
            "declare them (e.g. nullspace='constant')"
        )
    singular = nullspace is not None or transpose_nullspace is not None

    local_device = mesh.local_devices[0]
    local_hardware_id = getattr(local_device, "local_hardware_id", local_device.id)

    # MPI setup needs CSR structure on the host. Materialize it exactly once;
    # the halo and transpose plans below share these arrays.
    indices_host = np.asarray(A_structure.indices, dtype=np.int64)
    indptr_host = np.asarray(A_structure.indptr, dtype=np.int64)
    # Padded to the global maxima: every rank's shard_map program has one shape.
    halo_plan = build_halo_plan(
        indices_host, A.row_counts, partition_info, comm, pad=True
    )

    rhs_spec = P(axis_name)
    A_data_spec = P(axis_name)
    max_nnz = A.max_local_nnz
    local_nnz = A.local_nnz
    A_data = A.data

    if is_symmetric:
        max_transpose_nnz = None
        transpose_exchange = None
    else:
        transpose_plan = build_transpose_plan(
            indices_host,
            indptr_host,
            A.row_counts,
            partition_info,
            comm,
            pad=True,
        )
        max_transpose_nnz = transpose_plan.max_nnz
        transpose_exchange = transpose_plan.exchange

    # The local CSR row index of every nonzero (for the SpMV maps).
    local_row_indices = np.repeat(
        np.arange(A_structure.shape[0], dtype=np.int32),
        np.diff(indptr_host),
    ).astype(np.int32)

    comm_ptr = register_comm(comm)
    config_str = amgx_config.prepare_config(
        config or {},
        save_stats=save_stats,
        mpi=True,
        block_dim=block_dim,
        singular=singular,
    )
    # Every rank compiles one program: rank-local arrays enter at run time
    # (``jaxamg.local_arrays``), padded to the largest local sizes.
    from mpi4py import MPI

    from .jaxamg import _amgx_solve_mpi_impl
    from .local_arrays import load_local, register_local
    from .nullspace import project_out, relative_norm

    def register(array: np.ndarray):
        """Register this rank's ``array`` for the services (collective)."""
        return register_local(comm, array, local_device)

    max_n_ghost = halo_plan.max_n_ghost
    n_max = max_n_local

    def pad_to(array: np.ndarray, size: int, value: Any = 0) -> np.ndarray:
        return np.pad(array, (0, size - len(array)), constant_values=value)

    def register_structure(indptr: np.ndarray, indices: np.ndarray, nnz_max: int):
        nnz = int(indptr[-1])
        return (
            register(pad_to(indptr.astype(np.int32), n_max + 1, int(indptr[-1]))),
            register(pad_to(indices.astype(np.int64), nnz_max)),
            register(np.array([n_local, nnz], dtype=np.int32)),
        )

    local_comm = register(
        np.array(
            [
                np.int32(np.uint32(comm_ptr & 0xFFFFFFFF)),
                np.int32(np.uint32((comm_ptr >> 32) & 0xFFFFFFFF)),
                int(local_hardware_id),
            ],
            dtype=np.int32,
        ),
    )
    nglobal_arr = np.array([nglobal], dtype=np.int32)
    primal_structure = register_structure(indptr_host, indices_host, max_nnz)
    if is_symmetric:
        adjoint_structure = primal_structure
    else:
        adjoint_structure = register_structure(
            np.asarray(transpose_plan.indptr, dtype=np.int64),
            np.asarray(transpose_plan.indices, dtype=np.int64),
            max_transpose_nnz,
        )
        # Padded routing. Sentinels move zeros: a source index max_nnz reads
        # the zero appended to the values, a target index max_transpose_nnz
        # writes the extra slot, and the plan's own sentinels land in padding.
        width = max(
            int(comm.allreduce(len(transpose_plan.local_source_ids), op=MPI.MAX)), 1
        )
        transpose_routing = (
            register(
                pad_to(transpose_plan.local_source_ids, width, max_nnz).astype(
                    np.int32
                ),
            ),
            register(
                pad_to(
                    transpose_plan.local_target_ids, width, max_transpose_nnz
                ).astype(np.int32),
            ),
            register(np.asarray(transpose_plan.send_ids, dtype=np.int32)),
            register(np.asarray(transpose_plan.recv_target_ids, dtype=np.int32)),
        )
    # SpMV maps (the matrix gradient is their transpose): ghost slots follow
    # the padded local block.
    pad_row = n_max - 1  # after every real row, so the rows stay sorted
    local_slots = np.asarray(halo_plan.col_to_combined, dtype=np.int64)
    local_slots = np.where(
        local_slots >= n_local, local_slots - n_local + n_max, local_slots
    )
    spmv_maps = (
        register(pad_to(local_row_indices, max_nnz, pad_row).astype(np.int32)),
        register(pad_to(local_slots, max_nnz).astype(np.int32)),
        register(pad_to(np.ones(local_nnz, dtype=np.int32), max_nnz).astype(np.int32)),
        register(np.asarray(halo_plan.send_ids, dtype=np.int32)),
    )

    def register_basis(basis):
        if basis is None:
            return None
        values = np.asarray(basis)[:n_local]
        return register(
            np.pad(values, ((0, n_max - n_local), (0, 0))).astype(A_data.dtype)
        )

    # The null spaces given at construction, each solve's defaults: this
    # rank's rows, registered per rank like the matrix structure (a jit
    # cannot close over an array spanning processes) and loaded as a solve's
    # operands.
    default_bases = [register_basis(nullspace)]
    default_bases.append(
        default_bases[0]
        if transpose_nullspace is nullspace
        else register_basis(transpose_nullspace)
    )
    default_labels = (
        None
        if labels is None
        else register(pad_to(np.asarray(labels[1], np.int32), n_max, -1))
    )

    res_history_len = amgx_config.outer_max_iters(config_str) + 1
    configs = {singular: config_str}

    def config_for(singular_: bool) -> str:
        """The AmgX configuration of a solve with (or without) a null space:
        a declared one selects the singular coarse solve, as in solve()."""
        if singular_ not in configs:
            configs[singular_] = amgx_config.prepare_config(
                config or {},
                save_stats=save_stats,
                mpi=True,
                block_dim=block_dim,
                singular=singular_,
            )
            if singular_ and amgx_config.uses_dense_lu_coarse_solver(
                configs[singular_]
            ):
                warnings.warn(_DENSE_LU_MSG, NullSpaceWarning, stacklevel=4)
        return configs[singular_]

    def loader(in_map: bool) -> Callable[[Any], jax.Array]:
        axis = axis_name if in_map else None
        return lambda handle: load_local(handle, varying_axis=axis)

    def native_solve(load, structure, values, rhs, x0, singular_):
        indptr_h, indices_h, sizes_h = structure
        comm_lrank = load(local_comm)
        return _amgx_solve_mpi_impl(
            load(indptr_h),
            load(indices_h),
            values,
            rhs,
            x0,
            jnp.asarray(nglobal_arr),
            comm_lrank[:2],
            comm_lrank[2:],
            config_str=config_for(singular_),
            return_stats=int(save_stats),
            reuse_setup=reuse_setup,
            res_history_len=res_history_len,
            use_x0=x0 is not None,
            block_dim=block_dim,
            ordered=False,
            local_sizes=load(sizes_h),
        )

    def info_specs_for(has_M: bool) -> dict:
        specs = {
            "iterations": P(axis_name),
            "residual": P(axis_name),
            "status": P(axis_name),
            "residual_history": P(axis_name, None),
        }
        if has_M:
            specs["rhs_inconsistency"] = P(axis_name)
        return specs

    def pack_info(stats: jax.Array, inconsistency) -> ShardedInfo:
        packed = {
            "iterations": stats[0].astype(jnp.int32)[None],
            "residual": stats[1][None],
            "status": stats[2].astype(jnp.int32)[None],
            "residual_history": stats[3:][None, :],
        }
        if inconsistency is not None:
            packed["rhs_inconsistency"] = jnp.asarray(
                inconsistency, dtype=A_data.dtype
            )[None]
        return packed

    def make_local_transpose_values(ordered: bool, in_map: bool):
        load = loader(in_map)

        def local_transpose_values(A_data_local):
            return mpi_transpose_values(
                A_data_local,
                tuple(load(h) for h in transpose_routing),
                transpose_exchange,
                max_transpose_nnz,
                ordered=ordered,
            )

        return local_transpose_values

    def gather_solution_halo(load, x_padded, ordered: bool) -> jax.Array:
        # [x (padded) | ghosts (zero-padded to the global maximum)].
        return halo_gather(
            x_padded,
            load(spmv_maps[3]),
            halo_plan.exchange,
            n_ghost=max_n_ghost,
            ordered=ordered,
        )

    nranks = comm.Get_size()

    def lax_reduce(value: jax.Array) -> jax.Array:
        return jax.lax.psum(value, axis_name)

    def lax_max(value: jax.Array) -> jax.Array:
        return jax.lax.pmax(value, axis_name)

    if nranks == 1:

        def mpi_reduce(value: jax.Array) -> jax.Array:
            return value

    else:

        def mpi_reduce(value: jax.Array) -> jax.Array:
            import mpi4jax
            from mpi4py import MPI

            return mpi4jax.allreduce(value, op=MPI.SUM, comm=comm)

    # The mesh-local device as a one-device mesh. The eager local branch runs
    # under it so that its single-device operations dispatch cleanly even when
    # the caller sits inside jax.set_mesh over the whole multi-process mesh
    # (as an eager outer transform requires for its own global-array ops).
    local_mesh = _local_mesh(mesh, axis_name)

    def local_shard(value: jax.Array) -> jax.Array:
        # Typed as replicated on the one-device mesh (no copy): JAX 0.11
        # rejects gathers on an untyped shard under jax.set_mesh.
        return jax.device_put(
            value.addressable_shards[0].data, NamedSharding(local_mesh, P())
        )

    def assemble_global(local_value: jax.Array) -> jax.Array:
        spec = P(axis_name, *(None,) * (local_value.ndim - 1))
        return jax.make_array_from_single_device_arrays(
            (nranks * local_value.shape[0], *local_value.shape[1:]),
            NamedSharding(mesh, spec),
            [jax.device_put(local_value, local_device)],
        )

    def dispatch(
        local_fn: Callable[..., Any], shard_mapped: Callable[..., Any]
    ) -> Callable[..., Any]:
        # Concrete operands bypass shard_map: outside a trace, JAX executes a
        # shard_map body primitive by primitive across the mesh, an order of
        # magnitude slower than running the body on the local shard directly.
        def invoke(*args: Any) -> Any:
            if any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(args)):
                return shard_mapped(*args)
            with jax.set_mesh(local_mesh):
                outputs = local_fn(*(local_shard(arg) for arg in args))
                return jax.tree.map(assemble_global, outputs)

        return invoke

    # The null-space programs, each one jitted map over the mesh (like the
    # SpMV's), so they run alike eagerly and under a transformation.
    columns_spec = P(axis_name, None)
    in_map = loader(True)

    def from_ranks(local, spec):
        """A global operand sharded like b, from each rank's block
        ``local()``."""
        return jax.jit(jax.shard_map(local, mesh=mesh, in_specs=(), out_specs=spec))

    # The construction's null spaces as operands, and a solve's "constant".
    default_operands = [
        None if h is None else from_ranks(lambda h=h: in_map(h), columns_spec)
        for h in default_bases
    ]
    if default_bases[1] is default_bases[0]:
        default_operands[1] = default_operands[0]
    default_rows = (
        None
        if default_labels is None
        else from_ranks(lambda: in_map(default_labels), rhs_spec)
    )
    constant_operand = from_ranks(
        lambda: jnp.ones((n_max, 1), A_data.dtype), columns_spec
    )

    @jax.jit
    def prepare(bases: tuple, rows):
        """A solve's bases and labels on this rank's rows (the padding belongs
        to no component). Unlabelled bases use unit_bases; labelled columns
        are scaled per component by the projection programs below."""

        def local(bases, *rows):
            real = jnp.arange(n_max) < in_map(primal_structure[2])[0]
            masked = tuple(jnp.where(real[:, None], B, 0) for B in bases)
            # Labelled columns must be scaled per component. A global column
            # scale can erase a smaller component before it is projected.
            if not rows:
                masked = unit_bases(masked, lax_max)
            return masked, [jnp.where(real, r, -1) for r in rows]

        rows = [] if rows is None else [rows]
        specs = (columns_spec,) * len(bases)
        bases, rows = jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(specs,) + (rhs_spec,) * len(rows),
            out_specs=(specs, [rhs_spec] * len(rows)),
        )(tuple(bases), *rows)
        return bases, (rows[0] if rows else None)

    def solve_map(structure, warm: bool, dense: tuple, singular_: bool):
        """The mapped solve ``(values, rhs[, x0], *bases) -> (x, info)`` of
        ``structure`` (the primal or the adjoint's), with the unlabelled bases
        present in ``dense`` (nullspace, transpose_nullspace) projected in the
        map: b onto range(A) (the removed fraction reported), then x out of
        null(A)."""

        def body(reduce, load):
            def local(values, rhs, *rest):
                x0 = rest[0] if warm else None
                bases = list(rest[int(warm) :])
                N = bases.pop(0) if dense[0] else None
                M = bases.pop(0) if dense[1] else None
                inconsistency = None
                if M is not None:
                    projected = project_out(rhs, M, reduce)
                    inconsistency = relative_norm(rhs - projected, rhs, reduce)
                    rhs = projected
                x, stats = native_solve(load, structure, values, rhs, x0, singular_)
                if N is not None:
                    x = project_out(x, N, reduce)
                return x, pack_info(stats, inconsistency)

            return local

        in_specs = (A_data_spec,) + (rhs_spec,) * (1 + int(warm))
        in_specs += (columns_spec,) * sum(dense)
        return dispatch(
            body(mpi_reduce, loader(False)),
            jax.shard_map(
                body(lax_reduce, in_map),
                mesh=mesh,
                in_specs=in_specs,
                out_specs=(rhs_spec, info_specs_for(dense[1])),
            ),
        )

    @functools.partial(jax.jit, static_argnums=4)
    def label_sums_map(v, basis, rows, scales, count):
        """Scale columns and form partial sums in the same mesh program."""

        def local(v, basis, rows, scales):
            basis = scale_label_basis(basis, (count, rows), scales)
            return basis, label_sums(v, basis, (count, rows))

        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(rhs_spec, columns_spec, rhs_spec, columns_spec),
            out_specs=(columns_spec, columns_spec),
        )(v, basis, rows, scales)

    @functools.partial(jax.jit, static_argnums=4)
    def label_removed_map(v, totals, basis, rows, count):
        """``v`` less each label's projection from the labels' totals, and the
        removed fraction."""

        def local(v, totals, basis, rows):
            projected = remove_label_sums(v, basis, (count, rows), totals)
            return projected, relative_norm(v - projected, v, lax_reduce)[None]

        return jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(rhs_spec, columns_spec, columns_spec, rhs_spec),
            out_specs=(rhs_spec, P(axis_name)),
        )(v, totals, basis, rows)

    def label_projection(count: int, reduce_sum):
        """``(v, basis, labels) -> (v less each label's projection onto the
        basis, the removed fraction)``: each rank's per-label partial sums,
        their totals by ``reduce_sum`` (a map of global arrays holding each
        rank's ``count`` rows, sharded like b), then the removal."""

        def project(v, basis, rows):
            stats = label_scale_map(basis, rows, count)
            basis, sums = label_sums_map(v, basis, rows, reduce_sum(stats), count)
            return label_removed_map(v, reduce_sum(sums), basis, rows, count)

        return project

    @functools.partial(jax.jit, static_argnums=2)
    def label_scale_map(basis, rows, count):
        return jax.shard_map(
            lambda basis, rows: label_scale_stats(basis, (count, rows)),
            mesh=mesh,
            in_specs=(columns_spec, rhs_spec),
            out_specs=columns_spec,
        )(basis, rows)

    # The labels' default reduction: an all-reduce of the ranks' sums (labels
    # numbered globally).
    sum_over_ranks = jax.jit(
        jax.shard_map(
            lax_reduce, mesh=mesh, in_specs=(columns_spec,), out_specs=columns_spec
        )
    )

    if is_symmetric:
        mapped_transpose_values = None
    else:
        mapped_transpose_values = dispatch(
            # The eager local branch is a per-rank program (ordered exchange);
            # inside shard_map one SPMD program fixes the order.
            make_local_transpose_values(True, False),
            jax.shard_map(
                make_local_transpose_values(False, True),
                mesh=mesh,
                in_specs=(A_data_spec,),
                out_specs=A_data_spec,
            ),
        )

    def validate_nulls(basis, name, labels_, label_sum_):
        """``validate_basis`` over the ranks, on this rank's rows: the
        all-reduce, or a caller's label reduction applied to the ranks' sums
        as one global array."""
        with jax.set_mesh(local_mesh):
            if labels_ is None or label_sum_ is None:
                validate_basis(basis, name, comm, labels_)
                return

            def reduce(sums):
                local = np.asarray(sums.addressable_shards[0].data)
                with jax.set_mesh(mesh):
                    totals = label_sum_(assemble_global(jnp.asarray(local)))
                return jnp.asarray(np.asarray(totals.addressable_shards[0].data))

            validate_basis(basis, name, None, labels_, reduce)

    # Validate here, collectively, on the schedule the schema check fixed.
    if nullspace is not None:
        validate_nulls(nullspace, "nullspace", labels, default_label_sum)
    if transpose_nullspace is not None and transpose_nullspace is not nullspace:
        validate_nulls(
            transpose_nullspace, "transpose_nullspace", labels, default_label_sum
        )

    coloring = A._coloring

    fixed_indices = np.asarray(A_structure.indices)
    fixed_indptr = np.asarray(A_structure.indptr)

    def require_fixed_structure(problem: str | None) -> None:
        """Raise on every rank if any rank's override has another structure
        (``problem`` is this rank's reason): the rejection is agreed before
        any materialization or solve communication."""
        from mpi4py import MPI

        if not comm.allreduce(problem is None, op=MPI.LAND):
            raise ValueError(
                problem
                or "A must have the sparsity structure fixed when the solver "
                "was created (it differs on another rank)"
            )

    def structure_problem(indices: Any, indptr: Any) -> str | None:
        """Why this rank's CSR structure differs from the fixed one, or None."""
        for given, fixed in ((indices, fixed_indices), (indptr, fixed_indptr)):
            if isinstance(given, jax.core.Tracer):
                return (
                    "A's sparsity structure must be concrete; pass traced matrix "
                    "values in the packed A.data layout instead"
                )
            given = np.asarray(given)
            if given.shape != fixed.shape or not np.array_equal(given, fixed):
                return (
                    "A must have the sparsity structure fixed when the solver "
                    "was created"
                )
        return None

    def check_structure(matrix: jsp.BCSR) -> None:
        """Reject a matrix whose CSR structure differs from the fixed one."""
        if tuple(matrix.shape) != tuple(A_structure.shape):
            problem = (
                f"A must have local shape {tuple(A_structure.shape)}; got "
                f"{tuple(matrix.shape)}"
            )
        elif matrix.indices is A_structure.indices and (
            matrix.indptr is A_structure.indptr
        ):
            problem = None
        else:
            problem = structure_problem(matrix.indices, matrix.indptr)
        require_fixed_structure(problem)

    def local_values_of(A_local: MatrixOrOperator) -> jax.Array:
        """This rank's padded values of ``A_local`` in the packed layout."""
        from .halo import HaloOperator

        if isinstance(A_local, HaloOperator):
            indices, indptr = A_local._host_structure(
                partition_info[0], A_data.dtype, traced=True
            )
            require_fixed_structure(structure_problem(indices, indptr))
            values = A_local._local_matrix(
                partition_info[0], nglobal, A_data.dtype, traced=True
            ).data
        elif callable(A_local):
            info = getattr(A_local, "_coloring_info", None) or coloring
            discovered = getattr(info, "dtype", None)
            if discovered is not None and discovered != A_data.dtype:
                by_dtype = getattr(A_local, "_coloring_by_dtype", None) or {}
                info = by_dtype.get(jnp.dtype(A_data.dtype))
                if info is None:
                    raise ValueError(
                        f"A's colouring was discovered at {discovered}, but this "
                        f"solver's values are {A_data.dtype}; attach "
                        "jaxamg.cache_coloring(op, shape=(n_local, n_global), "
                        f"dtype={A_data.dtype})"
                    )
            if info is None:
                raise ValueError(
                    "A is a matrix-free operator without coloring information; "
                    "build the sharded matrix from the operator, or attach "
                    "coloring with jaxamg.with_cache(op, coloring="
                    "jaxamg.cache_coloring(op, shape=(n_local, n_global)))"
                )
            rows, cols, column_colors, n_colors, shape = info
            if tuple(shape) != (n_local, nglobal):
                problem: str | None = (
                    f"A must have local shape {(n_local, nglobal)}; its "
                    f"coloring describes shape {tuple(shape)}"
                )
            else:
                from .sparsity import csr_structure

                # The recovered structure, in materialization order (row,
                # then column), must be the fixed one.
                _, recovered, recovered_indptr = csr_structure(
                    np.asarray(rows).astype(np.int64),
                    np.asarray(cols).astype(np.int64),
                    n_local,
                )
                problem = structure_problem(recovered, recovered_indptr)
            require_fixed_structure(problem)
            from .sparsity import materialize_sparse_matrix

            values = materialize_sparse_matrix(
                A_local,
                shape,
                rows,
                cols,
                column_colors,
                n_colors,
                dtype=A_data.dtype,
            ).data
        else:
            matrix = to_bcsr_matrix(
                A_local,
                b=jnp.zeros(n_local, dtype=A_data.dtype),
                use_int64_indices=True,
            )
            check_structure(matrix)
            values = matrix.data
        if values.shape[0] != local_nnz:
            raise ValueError(
                "A must have the sparsity structure fixed when the solver was "
                f"created; got {values.shape[0]} local nonzeros, expected "
                f"{local_nnz}"
            )
        return jnp.pad(values.astype(A_data.dtype), (0, max_nnz - local_nnz))

    def require_replicated(value: jax.Array) -> None:
        sharding = getattr(getattr(value, "aval", None), "sharding", None)
        spec = getattr(sharding, "spec", None)
        if spec is not None and any(entry is not None for entry in spec):
            raise ValueError(
                "a differentiable value that A closes over is sharded across "
                "ranks; such values must be identical on every rank. Pass "
                "rank-local matrix values as the packed array A.data instead."
            )

    def check_replicated(value: Any) -> None:
        if (
            isinstance(value, jax.Array)
            and not isinstance(value, jax.core.Tracer)
            and not value.sharding.is_fully_replicated
        ):
            raise ValueError(
                "a value that A closes over is sharded across ranks; such "
                "values must be identical on every rank. Pass rank-local "
                "matrix values as the packed array A.data instead."
            )

    def local_copy(value: Any) -> Any:
        """This process's copy of a value: for an array spanning the mesh,
        its addressable shard (the value is replicated)."""
        if isinstance(value, jax.Array) and not isinstance(value, jax.core.Tracer):
            return value.addressable_shards[0].data
        return value

    def replicated_over_mesh(local_value: jax.Array) -> jax.Array:
        """This rank's value typed as replicated over the mesh; ranks may
        differ, and only this rank's copy is read back."""
        return jax.make_array_from_single_device_arrays(
            local_value.shape,
            NamedSharding(mesh, P()),
            [jax.device_put(local_value, local_device)],
        )

    # The materialization runs outside shard_map (its per-process constants
    # would break shard_map's identical-body assumption), bracketed by two
    # linear maps with declared transposes: local_to_global (no inserted psum)
    # and reduce_cotangent (sums the ranks' parameter cotangents).
    to_shards = jax.shard_map(
        lambda values: values,
        mesh=mesh,
        in_specs=(P(),),
        out_specs=A_data_spec,
        check_vma=False,
    )
    from_shards = jax.shard_map(
        lambda values: values,
        mesh=mesh,
        in_specs=(A_data_spec,),
        out_specs=P(),
        check_vma=False,
    )
    sum_across_ranks = jax.shard_map(
        lax_reduce, mesh=mesh, in_specs=(P(),), out_specs=P(), check_vma=False
    )

    # Linear ownership maps with explicit transposes (every derivative order,
    # forward and reverse). Eager cotangents take the primal's typing: mesh-
    # typed (replicated over the mesh) or local, fixed when the map is bound.
    def local_to_global_forward(values: jax.Array) -> jax.Array:
        if isinstance(values, jax.core.Tracer):
            return to_shards(values)
        return assemble_global(local_copy(values))

    def global_to_local(mesh_typed: bool) -> Callable[[jax.Array], jax.Array]:
        def transpose(ct: jax.Array) -> jax.Array:
            if isinstance(ct, jax.core.Tracer):
                return from_shards(ct)
            local = local_shard(ct)
            return replicated_over_mesh(local) if mesh_typed else local

        return transpose

    def sum_to_ranks(mesh_typed: bool) -> Callable[[jax.Array], jax.Array]:
        def transpose(ct: jax.Array) -> jax.Array:
            if isinstance(ct, jax.core.Tracer):
                return sum_across_ranks(ct)
            with jax.set_mesh(local_mesh):
                reduced = mpi_reduce(local_copy(ct))
            return replicated_over_mesh(reduced) if mesh_typed else reduced

        return transpose

    local_to_global_maps = {
        typed: LinearMap(
            local_to_global_forward, global_to_local(typed), "local_to_global"
        )
        for typed in (False, True)
    }
    # One logical parameter expanded into every rank's use: the identity,
    # whose transpose sums the ranks' contributions (never a forward sum).
    expansion_maps = {
        typed: LinearMap(lambda value: value, sum_to_ranks(typed), "expand")
        for typed in (False, True)
    }

    def mesh_typed(value: jax.Array) -> bool:
        # From the JAX type, so a traced primal of an eager linearization is
        # typed as its value is.
        mesh_of = getattr(getattr(jax.typeof(value), "sharding", None), "mesh", None)
        return mesh_of is not None and not mesh_of.empty

    def local_to_global(values: jax.Array) -> jax.Array:
        return local_to_global_maps[mesh_typed(values)](values)

    def reduce_cotangent(value: jax.Array) -> jax.Array:
        return expansion_maps[mesh_typed(value)](value)

    def pack_operator(A_local: MatrixOrOperator, traced: bool) -> jax.Array:
        """Global packed values of a local operator or matrix, differentiable
        with respect to the traced values it closes over."""
        closed = jax.make_jaxpr(lambda: local_values_of(A_local))()
        # Traced closed-over values become explicit inputs; the rest stay
        # jaxpr constants.
        hoisted = tuple(
            const
            for const in closed.consts
            if isinstance(const, jax.core.Tracer)
            and jnp.issubdtype(const.dtype, jnp.inexact)
        )
        for value in hoisted:
            require_replicated(value)
        slots = {id(const): index for index, const in enumerate(hoisted)}
        values = tuple(reduce_cotangent(value) for value in hoisted)
        consts = []
        for const in closed.consts:
            if id(const) in slots:
                consts.append(values[slots[id(const)]])
                continue
            check_replicated(const)
            # A jit cannot close over an array spanning the mesh; eager
            # evaluation needs it as is.
            consts.append(local_copy(const) if traced else const)
        return local_to_global(jax.core.eval_jaxpr(closed.jaxpr, consts)[0])

    # The implicit core (``jaxamg.core``), outside shard_map so its cotangents
    # are global sharded arrays.
    from .core import implicit_solver

    def local_spmv(values_local: jax.Array, x_local: jax.Array) -> jax.Array:
        load = loader(True)
        rows, slots, valid = (load(h) for h in spmv_maps[:3])
        x_combined = gather_solution_halo(load, x_local, False)
        return jax.ops.segment_sum(
            jnp.where(valid > 0, values_local * x_combined[slots], 0),
            rows,
            num_segments=n_max,
            indices_are_sorted=True,
        )

    # Jitted, so an eager (un-jitted) reverse pass linearizes it as one call:
    # eager linearization of a bare shard_map stages its integer residuals with
    # a mismatched float0 cotangent spec.
    mapped_spmv = jax.jit(
        jax.shard_map(
            local_spmv,
            mesh=mesh,
            in_specs=(A_data_spec, rhs_spec),
            out_specs=rhs_spec,
        )
    )

    def build_core(dense: tuple, count: int | None, reduce_sum):
        """The implicit core of a solve whose null-space operands ``aux`` =
        (nullspace, transpose_nullspace, labels) are prepared: ``dense``, the
        bases present; with ``count`` labels, the labelled projections run
        around the singular solve, their sums reduced by ``reduce_sum``. For
        Aᵀ the two null spaces exchange roles."""
        singular_ = any(dense)
        project = None if count is None else label_projection(count, reduce_sum)

        def run(structure, values, rhs, x0, N, M, rows):
            warm = () if x0 is None else (x0,)
            if project is None:
                present = (N is not None, M is not None)
                solve = solve_map(structure, x0 is not None, present, singular_)
                bases = tuple(B for B in (N, M) if B is not None)
                return solve(values, rhs, *warm, *bases)
            inconsistency = None
            if M is not None:
                rhs, inconsistency = project(rhs, M, rows)
            solve = solve_map(structure, x0 is not None, (False, False), True)
            x, info = solve(values, rhs, *warm)
            if N is not None:
                x, _ = project(x, N, rows)
            if inconsistency is not None:
                info = {**info, "rhs_inconsistency": inconsistency}
            return x, info

        def sharded_native(indptr, indices, values, rhs_, x0_, aux):
            N, M, rows = aux
            return run(primal_structure, values, rhs_, x0_, N, M, rows)

        def sharded_zero_start(indptr, indices, values, rhs_, aux):
            return sharded_native(indptr, indices, values, rhs_, None, aux)[0]

        def sharded_transpose(indptr, indices, values, rhs_, aux):
            N, M, rows = aux
            if not is_symmetric:
                assert mapped_transpose_values is not None
                if isinstance(values, jax.core.Tracer) or isinstance(
                    rhs_, jax.core.Tracer
                ):
                    # Order the transpose exchange after the cotangent (hence
                    # after the forward solve) within one program.
                    values, _ = jax.lax.optimization_barrier((values, rhs_))
                values = mapped_transpose_values(values)
                N, M = M, N
            return run(adjoint_structure, values, rhs_, None, N, M, rows)[0]

        def sharded_spmv(values, indices, indptr, aux, x_):
            return mapped_spmv(values, x_)

        # The symmetric shortcut lives in sharded_transpose, so the core runs
        # with custom_linear_solve's general (non-symmetric) mode.
        return implicit_solver(
            sharded_native,
            sharded_zero_start,
            sharded_transpose,
            False,
            sharded_spmv,
            collective=True,
        )

    no_structure = jnp.zeros(0, dtype=jnp.int32)

    class _PackedValues(NamedTuple):
        data: jax.Array
        indices: jax.Array
        indptr: jax.Array

    def solve_basis(spec, name: str):
        """A solve's own basis as a global ``(rows, k)`` operand sharded like
        b: ``"constant"`` (built per rank) or the caller's array (a vector or
        ``(rows, k)``); None when not given."""
        if isinstance(spec, str) and spec.lower() == "constant":
            return constant_operand()
        return as_nullspace_basis(spec, nranks * n_max, A_data.dtype, name)

    def solve_labels(spec):
        """A solve's own labels: ``(count, a global integer vector sharded
        like b)``."""
        if not isinstance(spec, tuple) or len(spec) != 2:
            raise ValueError("labels must be a (count, labels) pair")
        count, rows = spec
        if (
            isinstance(count, bool)
            or not isinstance(count, (int, np.integer))
            or count < 1
        ):
            raise ValueError("labels' count must be a positive Python integer")
        # Preserve host integer widths until after range validation, even
        # when JAX's x64 mode is disabled.
        rows = rows if isinstance(rows, jax.Array) else np.asarray(rows)
        if rows.shape != (nranks * n_max,) or not jnp.issubdtype(
            rows.dtype, jnp.integer
        ):
            raise ValueError(
                f"labels must be a global integer vector of {nranks * n_max} rows sharded like b"
            )
        # Check local, unpadded values before narrowing; all ranks reject
        # together, before any basis-validation or solve collectives.
        if isinstance(rows, jax.core.Tracer):
            checked = rows
        elif isinstance(rows, jax.Array):
            checked = local_rows(rows)
        else:
            start = comm.Get_rank() * n_max
            checked = rows[start : start + n_local]
        validate_label_values((int(count), checked), comm)
        return int(count), jnp.asarray(rows, dtype=jnp.int32)

    def local_rows(value: jax.Array) -> np.ndarray:
        return np.asarray(value.addressable_shards[0].data)[:n_local]

    def sharded_solver(
        rhs: jax.Array,
        x0: jax.Array | None = None,
        *,
        A_override: MatrixOrOperator | jax.Array | None = None,
        save_stats_file: str | os.PathLike | None = None,
        nullspace: Any = None,
        transpose_nullspace: Any = None,
        labels: Any = None,
        label_sum: Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        _validate_operand(rhs, b, mesh, axis_name, "b")
        traced = isinstance(rhs, jax.core.Tracer) or isinstance(x0, jax.core.Tracer)
        if A_override is None:
            if traced:
                raise ValueError(
                    "A must be passed explicitly (this rank's operator or "
                    "matrix, or the packed values A.data) when a sharded solver "
                    "is used inside jax.jit, jax.grad, or another JAX transform"
                )
            matrix_data = A_data
        elif isinstance(A_override, jax.Array) and A_override.ndim == 1:
            # The packed global values themselves.
            matrix_data = A_override
        elif isinstance(A_override, _global_operator_type()):
            require_fixed_structure(
                structure_problem(A_override.indices, A_override.indptr)
            )
            matrix_data = A_override._packed_values(rhs)
        else:
            matrix_data = pack_operator(A_override, traced)
        _validate_operand(matrix_data, A_data, mesh, axis_name, "A_data")
        if save_stats_file is not None:
            if any(
                isinstance(operand, jax.core.Tracer)
                for operand in (matrix_data, rhs, x0)
            ):
                raise ValueError(
                    "save_stats_file requires a direct solver call outside "
                    "jax.jit, jax.grad, and other JAX transforms"
                )
            if not save_stats:
                warnings.warn(
                    "save_stats_file was passed, but the solver was created "
                    "without stats output; the stats file will be missing "
                    "solver statistics. Pass save_stats=True to "
                    "make_sharded_solver().",
                    stacklevel=4,
                )
        if x0 is not None:
            _validate_operand(x0, b, mesh, axis_name, "x0")
        # The null spaces, operands: this solve's, each defaulting to the one
        # given at construction (a symmetric matrix shares one basis).
        N = solve_basis(nullspace, "nullspace")
        M = solve_basis(transpose_nullspace, "transpose_nullspace")
        if is_symmetric:
            N, M = (M if N is None else N), (N if M is None else M)
        own = isinstance(N, jax.Array) and not isinstance(nullspace, str)
        own |= isinstance(M, jax.Array) and not isinstance(transpose_nullspace, str)
        if N is None and default_operands[0] is not None:
            N = default_operands[0]()
        if M is None and default_operands[1] is not None:
            M = (
                N
                if default_operands[1] is default_operands[0]
                else default_operands[1]()
            )
        if labels is not None:
            count, rows = solve_labels(labels)
            own = True
        elif default_labels is not None:
            assert A._labels is not None and default_rows is not None
            count, rows = A._labels[0], default_rows()
        else:
            count = rows = None
        if rows is not None and N is None and M is None:
            raise ValueError(
                "labels apply to the declared nullspace/transpose_nullspace "
                "columns; declare them (e.g. nullspace='constant')"
            )
        label_sum = default_label_sum if label_sum is None else label_sum
        # A solve's own concrete null space: finite, and independent on each
        # label's rows.
        if own and not any(isinstance(v, jax.core.Tracer) for v in (N, M, rows)):
            local_labels = None if rows is None else (count, local_rows(rows))
            for basis, name in ((N, "nullspace"), (M, "transpose_nullspace")):
                if basis is not None and not (basis is N and name != "nullspace"):
                    validate_nulls(local_rows(basis), name, local_labels, label_sum)
        if N is not None or M is not None:
            shared = M is N
            bases = (N,) if shared else tuple(B for B in (N, M) if B is not None)
            prepared, rows = prepare(bases, rows)
            prepared = list(prepared)
            N = prepared.pop(0) if N is not None else None
            M = N if shared else (prepared.pop(0) if M is not None else None)
        reduce_sum = sum_over_ranks if label_sum is None else label_sum
        core = build_core(
            (N is not None, M is not None), None if rows is None else count, reduce_sum
        )
        result = core(
            _PackedValues(matrix_data, no_structure, no_structure),
            rhs,
            x0,
            (N, M, rows),
        )
        if save_stats_file is not None:
            # Statistics are captured during execution, so wait for the solve
            # to finish before reading them back from the extension.
            result[0].block_until_ready()
            _capture_and_save_stats(save_stats_file, comm=comm)
        return result

    def unpad_local_vector(value: jax.Array) -> jax.Array:
        _validate_operand(value, b, mesh, axis_name, "solver vector")
        with jax.set_mesh(local_mesh):
            return value.addressable_shards[0].data[:n_local]

    def solve_fn(
        rhs: jax.Array,
        x0: jax.Array | None = None,
        *,
        A: MatrixOrOperator | jax.Array | None = None,
        save_stats_file: str | os.PathLike | None = None,
        nullspace: Any = None,
        transpose_nullspace: Any = None,
        labels: Any = None,
        label_sum: Callable[[jax.Array], jax.Array] | None = None,
    ) -> tuple[jax.Array, ShardedInfo]:
        return sharded_solver(
            rhs,
            x0,
            A_override=A,
            save_stats_file=save_stats_file,
            nullspace=nullspace,
            transpose_nullspace=transpose_nullspace,
            labels=labels,
            label_sum=label_sum,
        )

    return ShardedSolve(
        solve_fn,
        unpad_local_vector,
        nglobal,
        n_local,
    )
