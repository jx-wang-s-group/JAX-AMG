"""JAX sharding integration for distributed AmgX solves.

This module provides an additive interface on top of JAX-AMG's MPI backend.
JAX owns the global arrays and ``shard_map`` execution, while AmgX continues
to use one MPI rank per GPU for the distributed solve.
"""

from __future__ import annotations

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
from .cache import _build_mpi_cache, with_cache
from .jaxamg import _capture_and_save_stats, solve
from .mpi_utils import (
    HaloPlan,
    _apply_transpose_plan,
    build_halo_plan,
    build_transpose_plan,
)
from .utils import (
    MatrixOrOperator,
    get_preferred_dtype,
    temp_enable_x64,
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

    That autotuning assumes every process compiles the identical program, but
    a sharded solve compiles rank-local structure into each process's program
    and deadlocks under it. Disabling it only affects compile time. XLA reads
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
            "XLA's sharded autotuning is enabled, so compiling a multi-process "
            "sharded solve will deadlock. Import jaxamg before the first JAX "
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
        """
        return self._solve_fn(b, x0, A=A, save_stats_file=save_stats_file)

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
    axis_name: str = "rank",
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
        mesh: One-dimensional JAX device mesh with one device per MPI rank.
            Defaults to a mesh over the first ``comm.size`` JAX devices (all
            devices in a typical multi-process job).
        global_size: Optional true global length. When provided, it is checked
            against the sum of local lengths.
        axis_name: Mesh axis used to partition the vector.

    Returns:
        A global JAX array whose axis uses ``P(axis_name)``. Its physical
        length is ``comm.size * max(local_sizes)``; values are cast to
        ``float32`` unless already ``float32`` or ``float64``.
    """
    comm = _resolve_comm(comm)
    if mesh is None:
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


def _local_mesh(mesh: Mesh, axis_name: str) -> Mesh:
    """This process's mesh device as a one-device mesh."""
    return jax.make_mesh((1,), (axis_name,), devices=[mesh.local_devices[0]])


def _require_shard_map() -> None:
    if not hasattr(jax, "shard_map"):
        raise RuntimeError(
            "JAX-AMG's sharding interface requires a JAX version that exposes "
            "jax.shard_map. Upgrade JAX or use solve(..., comm=...) instead."
        )


def _reject_nullspace(A_local: MatrixOrOperator) -> None:
    if any(
        getattr(A_local, attr, None) is not None
        for attr in ("_nullspace", "_transpose_nullspace")
    ):
        raise ValueError(
            "null spaces are not supported by the sharding interface yet; use "
            "solve(..., comm=...) for singular systems"
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
        coloring,
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
            ``(n_local, n_global)`` and global column indices. A matrix-free
            operator is also accepted, provided it carries cached coloring
            information (``jaxamg.with_cache(op, coloring=...)``) so its shape
            and sparsity pattern are known; it is materialized once here.
        b: Global row-sharded RHS used to validate the matrix partition, mesh,
            and numerical dtype.
        comm: MPI communicator spanning every JAX process, with rank order
            matching the JAX process order. Defaults to ``MPI.COMM_WORLD``.
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

    _reject_nullspace(A_local)
    n_local, nglobal = _local_matrix_shape(A_local)
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
        coloring=(
            getattr(A_local, "_coloring_info", None) if callable(A_local) else None
        ),
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

    Whether the solve is compiled is entirely the caller's choice: this function
    adds no ``jax.jit`` of its own. A caller that wraps its own function in
    ``jax.jit`` gets the whole pipeline compiled as part of that single program
    (through ``jax.shard_map``). A direct, untransformed call instead executes
    the rank-local pipeline on this process's shard -- the same code path as
    ``solve(..., comm=...)``, with mpi4jax for the gradient exchanges -- and
    reassembles global arrays, so it matches the MPI interface's eager speed.

    This interface complements, rather than replaces, ``solve(..., comm=...)``.
    JAX manages the global input and output arrays through ``shard_map`` while
    the AmgX solve itself uses the supplied MPI communicator. The interface is
    experimental; it uses a one-dimensional mesh and requires one MPI process
    with one mesh-local GPU per rank and a communicator spanning every JAX
    process.

    ``jax.distributed.initialize()`` must be called before this function in a
    multi-process job. The matrix owns the communicator, mesh, local CSR
    structure, and globally sharded packed values. Pass ``A.data`` through
    ``solver(..., A=A.data)`` to differentiate matrix values. Use
    ``jax.set_mesh(A.mesh)`` around outer transforms such as ``jax.grad``.

    .. note::
        Importing jaxamg disables XLA's cross-process sharded autotuning,
        which deadlocks on the per-process programs a sharded solve compiles
        to. Import it before the first JAX device call; see :doc:`sharding`.

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
    _require_shard_map()
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
    local_device = mesh.local_devices[0]
    local_hardware_id = getattr(local_device, "local_hardware_id", local_device.id)

    # MPI setup needs CSR structure on the host. Materialize it exactly once;
    # the halo and transpose plans below share these arrays.
    indices_host = np.asarray(A_structure.indices, dtype=np.int64)
    indptr_host = np.asarray(A_structure.indptr, dtype=np.int64)
    halo_plan = build_halo_plan(indices_host, A.row_counts, partition_info, comm)

    rhs_spec = P(axis_name)
    A_data_spec = P(axis_name)
    max_nnz = A.max_local_nnz
    local_nnz = A.local_nnz
    A_data = A.data

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
        transpose_plan = build_transpose_plan(
            indices_host,
            indptr_host,
            A.row_counts,
            partition_info,
            comm,
        )
        with temp_enable_x64():
            transpose_structure = _CSRStructure(
                jnp.asarray(transpose_plan.indices),
                jnp.asarray(transpose_plan.indptr),
                (n_local, nglobal),
                transpose_plan.nnz,
            )
        nnz_out = transpose_plan.nnz
        max_transpose_nnz = transpose_plan.max_nnz
        # Uncommitted for the same reason as the CSR structure above.
        transpose_local_source_ids = jnp.asarray(transpose_plan.local_source_ids)
        transpose_local_target_ids = jnp.asarray(transpose_plan.local_target_ids)
        transpose_send_ids = jnp.asarray(transpose_plan.send_ids_2d)
        transpose_recv_target_ids = jnp.asarray(transpose_plan.recv_target_ids_2d)

    # The local CSR row index of every nonzero, used by both the nested MPI
    # solve and this module's matrix-gradient body.
    local_row_indices = np.repeat(
        np.arange(A_structure.shape[0], dtype=np.int32),
        np.diff(indptr_host),
    ).astype(np.int32)

    mpi_cache = _build_mpi_cache(
        config or {},
        comm,
        nglobal,
        A.row_counts,
        max_nnz,
        nnz_out,
        halo_plan,
        row_indices=local_row_indices,
        save_stats=save_stats,
        block_dim=block_dim,
        lrank=int(local_hardware_id),
        device=local_device,
        commit=False,
    )
    if not is_symmetric:
        # The MPI solve's primal does not consume halo operands, and sharding's
        # outer VJP computes its matrix gradient from A's existing halo plan.
        # Therefore the nested A^T solve needs only placeholder halo operands.
        transpose_cache = {
            **mpi_cache,
            "halo_plan": _primal_halo_placeholder(n_local, len(A.row_counts)),
        }

    info_specs = {
        "iterations": P(axis_name),
        "residual": P(axis_name),
        "status": P(axis_name),
        "residual_history": P(axis_name, None),
    }

    res_history_len = amgx_config.outer_max_iters(mpi_cache["config_str"]) + 1

    def pack_info(info: ShardedInfo) -> ShardedInfo:
        history = jnp.asarray(info["residual_history"])
        if history.shape[0] < res_history_len:
            # A direct (untraced) solve trims the history to its own iteration
            # count, which differs per rank; restore the static traced length
            # (NaN-padded) so every rank's info shard has one common shape.
            history = jnp.pad(
                history,
                (0, res_history_len - history.shape[0]),
                constant_values=jnp.nan,
            )
        return {
            "iterations": jnp.asarray(info["iterations"], dtype=jnp.int32)[None],
            "residual": jnp.asarray(info["residual"])[None],
            "status": jnp.asarray(info["status"], dtype=jnp.int32)[None],
            "residual_history": history[None, :],
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
        return jnp.pad(value, (0, max_n_local - n_local))

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
            # os.devnull enables stats capture in the FFI call without writing
            # anything: solve() never writes a file for traced results and
            # skips the write for os.devnull on direct local calls. The actual
            # file is written by ``sharded_solver`` after execution.
            save_stats_file=os.devnull if save_stats else None,
        )

    def local_solve(
        A_data_local: jax.Array, rhs_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_structure, mpi_cache, is_symmetric)
        x_local, info = solve_one_rhs(A_dynamic, rhs_local)
        return pad_local_vector(x_local), pack_info(info)

    def local_solve_x0(
        A_data_local: jax.Array, rhs_local: jax.Array, x0_local: jax.Array
    ) -> tuple[jax.Array, ShardedInfo]:
        A_dynamic = matrix_with_data(A_data_local, A_structure, mpi_cache, is_symmetric)
        x_local, info = solve_one_rhs(A_dynamic, rhs_local, x0_local)
        return pad_local_vector(x_local), pack_info(info)

    def make_local_transpose_values(
        exchange: Callable[[jax.Array], jax.Array],
    ) -> Callable[[jax.Array], jax.Array]:
        def local_transpose_values(A_data_local: jax.Array) -> jax.Array:
            assert transpose_structure is not None
            assert max_transpose_nnz is not None
            assert transpose_local_source_ids is not None
            assert transpose_local_target_ids is not None
            assert transpose_send_ids is not None
            assert transpose_recv_target_ids is not None

            transpose_values = _apply_transpose_plan(
                A_data_local,
                transpose_local_source_ids,
                transpose_local_target_ids,
                transpose_send_ids,
                transpose_recv_target_ids,
                transpose_structure.nnz,
                exchange,
            )
            return jnp.pad(
                transpose_values,
                (0, max_transpose_nnz - transpose_structure.nnz),
            )

        return local_transpose_values

    def local_adjoint(adjoint_data_local: jax.Array, g_local: jax.Array) -> jax.Array:
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
        adjoint_local, _ = solve_one_rhs(A_adjoint, g_local)
        return pad_local_vector(adjoint_local)

    halo_plan = mpi_cache["halo_plan"]
    max_n_ghost = halo_plan.max_n_ghost
    # Keep rank-local halo metadata inside the shard_map bodies. Making these
    # arrays global and closing over them in the custom VJP prevents an outer
    # multi-process jax.jit from lowering because their remote shards are not
    # addressable by the current process.
    row_indices = mpi_cache["row_indices"]
    col_to_combined = halo_plan.col_to_combined
    send_ids = halo_plan.send_ids_2d
    recv_ghost_slot = halo_plan.recv_ghost_slot_2d

    def gather_solution_halo(
        x_local: jax.Array,
        exchange: Callable[[jax.Array], jax.Array],
    ) -> jax.Array:
        send_buffer = x_local[send_ids]
        recv_buffer = exchange(send_buffer)
        x_ghost = jnp.zeros(max_n_ghost + 1, dtype=x_local.dtype)
        x_ghost = x_ghost.at[recv_ghost_slot.reshape(-1)].set(recv_buffer.reshape(-1))
        return jnp.concatenate([x_local, x_ghost[:max_n_ghost]], axis=0)

    def make_local_matrix_gradient(
        exchange: Callable[[jax.Array], jax.Array],
    ) -> Callable[[jax.Array, jax.Array], jax.Array]:
        def local_matrix_gradient(
            x_local: jax.Array, adjoint_local: jax.Array
        ) -> jax.Array:
            # Order the halo exchange after the preceding AmgX adjoint solve.
            x_ordered, adjoint_ordered = jax.lax.optimization_barrier(
                (x_local, adjoint_local)
            )
            x_combined = gather_solution_halo(x_ordered[:n_local], exchange)
            grad_values = (
                -adjoint_ordered[:n_local][row_indices] * x_combined[col_to_combined]
            )
            return jnp.pad(grad_values, (0, max_nnz - local_nnz))

        return local_matrix_gradient

    nranks = comm.Get_size()

    def lax_exchange(values: jax.Array) -> jax.Array:
        return jax.lax.all_to_all(values, axis_name, split_axis=0, concat_axis=0)

    if nranks == 1:
        # A single-rank all-to-all is the identity.
        def mpi_exchange(values: jax.Array) -> jax.Array:
            return values

    else:

        def mpi_exchange(values: jax.Array) -> jax.Array:
            import mpi4jax

            return mpi4jax.alltoall(values, comm=comm)

    # The mesh-local device as a one-device mesh. The eager local branch runs
    # under it so that its single-device operations dispatch cleanly even when
    # the caller sits inside jax.set_mesh over the whole multi-process mesh
    # (as an eager outer transform requires for its own global-array ops).
    local_mesh = _local_mesh(mesh, axis_name)

    def local_shard(value: jax.Array) -> jax.Array:
        return value.addressable_shards[0].data

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

    mapped_solve = dispatch(
        local_solve,
        jax.shard_map(
            local_solve,
            mesh=mesh,
            in_specs=(A_data_spec, rhs_spec),
            out_specs=(rhs_spec, info_specs),
        ),
    )
    mapped_solve_x0 = dispatch(
        local_solve_x0,
        jax.shard_map(
            local_solve_x0,
            mesh=mesh,
            in_specs=(A_data_spec, rhs_spec, rhs_spec),
            out_specs=(rhs_spec, info_specs),
        ),
    )
    scalar_spec = P(axis_name)
    if is_symmetric:
        mapped_transpose_values = None
    else:
        mapped_transpose_values = dispatch(
            make_local_transpose_values(mpi_exchange),
            jax.shard_map(
                make_local_transpose_values(lax_exchange),
                mesh=mesh,
                in_specs=(A_data_spec,),
                out_specs=A_data_spec,
            ),
        )
    mapped_adjoint = dispatch(
        local_adjoint,
        jax.shard_map(
            local_adjoint,
            mesh=mesh,
            in_specs=(A_data_spec, scalar_spec),
            out_specs=scalar_spec,
        ),
    )
    mapped_matrix_gradient = dispatch(
        make_local_matrix_gradient(mpi_exchange),
        jax.shard_map(
            make_local_matrix_gradient(lax_exchange),
            mesh=mesh,
            in_specs=(scalar_spec, scalar_spec),
            out_specs=A_data_spec,
        ),
    )

    coloring = A._coloring

    def check_structure(matrix: jsp.BCSR) -> None:
        """Reject a matrix whose CSR structure differs from the fixed one."""
        for given, fixed in (
            (matrix.indices, A_structure.indices),
            (matrix.indptr, A_structure.indptr),
        ):
            if given is fixed:
                continue
            if isinstance(given, jax.core.Tracer):
                raise ValueError(
                    "A's sparsity structure must be concrete; pass traced matrix "
                    "values in the packed A.data layout instead"
                )
            if given.shape != fixed.shape or not np.array_equal(
                np.asarray(given), np.asarray(fixed)
            ):
                raise ValueError(
                    "A must have the sparsity structure fixed when the solver "
                    "was created"
                )

    def local_values_of(A_local: MatrixOrOperator) -> jax.Array:
        """This rank's padded values of ``A_local`` in the packed layout."""
        if callable(A_local):
            info = getattr(A_local, "_coloring_info", None) or coloring
            if info is None:
                raise ValueError(
                    "A is a matrix-free operator without coloring information; "
                    "build the sharded matrix from the operator, or attach "
                    "coloring with jaxamg.with_cache(op, coloring="
                    "jaxamg.cache_coloring(op, shape=(n_local, n_global)))"
                )
            rows, cols, column_colors, n_colors, shape = info
            if tuple(shape) != (n_local, nglobal):
                raise ValueError(
                    f"A must have local shape {(n_local, nglobal)}; its "
                    f"coloring describes shape {tuple(shape)}"
                )
            from .sparsity import materialize_sparse_matrix

            values = materialize_sparse_matrix(
                A_local, shape, rows, cols, column_colors, n_colors
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

    def lax_reduce(value: jax.Array) -> jax.Array:
        return jax.lax.psum(value, axis_name)

    if nranks == 1:

        def mpi_reduce(value: jax.Array) -> jax.Array:
            return value

    else:

        def mpi_reduce(value: jax.Array) -> jax.Array:
            import mpi4jax
            from mpi4py import MPI

            return mpi4jax.allreduce(value, op=MPI.SUM, comm=comm)

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

    def typed_like(cotangent: jax.Array, primal: jax.Array) -> jax.Array:
        if isinstance(getattr(primal, "sharding", None), NamedSharding):
            return replicated_over_mesh(cotangent)
        return cotangent

    def in_mesh_context() -> bool:
        return not jax.sharding.get_abstract_mesh().empty

    # The materialization is ordinary JAX code evaluated outside shard_map
    # (JAX assumes a shard_map body is identical on every device; per-process
    # constants violate that). Traced, each process computes its own values
    # typed replicated (P()); eagerly it runs in the caller's context, which
    # its arrays' types are tied to. Two custom VJPs bracket it: local_to_global
    # avoids the psum JAX would insert when transposing replicated to sharded,
    # and reduce_cotangent sums parameter cotangents across ranks.
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

    @jax.custom_vjp
    def local_to_global(values: jax.Array) -> jax.Array:
        if isinstance(values, jax.core.Tracer):
            return to_shards(values)
        return assemble_global(local_copy(values))

    def local_to_global_fwd(values: jax.Array):
        return local_to_global(values), values

    def local_to_global_bwd(values: jax.Array, ct: jax.Array):
        if isinstance(ct, jax.core.Tracer):
            return (from_shards(ct),)
        return (typed_like(local_shard(ct), values),)

    local_to_global.defvjp(local_to_global_fwd, local_to_global_bwd)

    @jax.custom_vjp
    def reduce_cotangent(value: jax.Array) -> jax.Array:
        return value

    def reduce_cotangent_fwd(value: jax.Array):
        return value, value

    def reduce_cotangent_bwd(value: jax.Array, ct: jax.Array):
        if isinstance(ct, jax.core.Tracer):
            return (sum_across_ranks(ct),)
        with jax.set_mesh(local_mesh):
            reduced = mpi_reduce(local_copy(ct))
        return (typed_like(reduced, value),)

    reduce_cotangent.defvjp(reduce_cotangent_fwd, reduce_cotangent_bwd)

    def pack_operator(A_local: MatrixOrOperator, traced: bool) -> jax.Array:
        """Global packed values of a local operator or matrix, differentiable
        with respect to the traced values it closes over."""
        _reject_nullspace(A_local)
        closed = jax.make_jaxpr(lambda: local_values_of(A_local))()
        # Traced closed-over values become explicit inputs; the rest stay
        # jaxpr constants.
        hoisted = tuple(
            const
            for const in closed.consts
            if isinstance(const, jax.core.Tracer)
            and jnp.issubdtype(const.dtype, jnp.inexact)
        )
        if traced:
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

    def differentiated_solve_backward(
        matrix_data: jax.Array, x: jax.Array, g_x: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        if is_symmetric:
            adjoint_data = matrix_data
        else:
            assert mapped_transpose_values is not None
            if isinstance(matrix_data, jax.core.Tracer) or isinstance(
                x, jax.core.Tracer
            ):
                # Within one compiled program, tie the transpose exchange to the
                # completed forward solution so XLA cannot overlap it with
                # AmgX's preceding MPI collectives. Concrete calls dispatch each
                # step in program order, so they need no barrier -- and a
                # barrier over a multi-process global array could not be
                # dispatched eagerly anyway.
                matrix_data, _ = jax.lax.optimization_barrier((matrix_data, x))
            adjoint_data = mapped_transpose_values(matrix_data)

        adjoint = mapped_adjoint(adjoint_data, g_x)
        return adjoint, mapped_matrix_gradient(x, adjoint)

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
        # The x0 cotangent is zero (the converged solution ignores the initial
        # guess). Build it shard-locally for a concrete adjoint: zeros_like on
        # a multi-process global array cannot be dispatched eagerly.
        if isinstance(adjoint, jax.core.Tracer):
            grad_x0 = jnp.zeros_like(adjoint)
        else:
            with jax.set_mesh(local_mesh):
                grad_x0 = assemble_global(jnp.zeros_like(local_shard(adjoint)))
        return grad_A_data, adjoint, grad_x0

    differentiated_solve_x0.defvjp(
        differentiated_solve_x0_fwd, differentiated_solve_x0_bwd
    )

    def sharded_solver(
        rhs: jax.Array,
        x0: jax.Array | None = None,
        *,
        A_override: MatrixOrOperator | jax.Array | None = None,
        save_stats_file: str | os.PathLike | None = None,
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
        if x0 is None:
            result = differentiated_solve(matrix_data, rhs)
        else:
            _validate_operand(x0, b, mesh, axis_name, "x0")
            result = differentiated_solve_x0(matrix_data, rhs, x0)
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
    ) -> tuple[jax.Array, ShardedInfo]:
        return sharded_solver(rhs, x0, A_override=A, save_stats_file=save_stats_file)

    return ShardedSolve(
        solve_fn,
        unpad_local_vector,
        nglobal,
        n_local,
    )
