"""Global-view operators for the sharding interface.

``fn`` maps a global row-sharded vector to one (its communication is its own),
and each rank declares its rows' pattern. Materialization probes ``fn`` with
one global vector per colour of a distributed colouring, computed once. The
rank-local probe data enters the program at run time (`jaxamg.local_arrays`),
so every rank compiles the same program. Rows must be partitioned equally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

# Bound extraction workspace independently of the matrix and colour count.
_PROBE_TILE_ENTRIES = 1048576


class GlobalOperator:
    """A global-view operator; build it with `global_operator`."""

    def __init__(
        self,
        fn: Callable[[jax.Array], jax.Array],
        indices: Any,
        indptr: Any,
        *,
        comm: Comm,
        mesh: Any,
        axis_name: str | tuple[str, ...] = "rank",
    ):
        from mpi4py import MPI

        from .distributed_coloring import (
            _canonical_rows,
            _validated_csr,
            distributed_coloring,
        )
        from .local_arrays import register_local

        self.fn = fn
        self.comm, self.mesh, self.axis_name = comm, mesh, axis_name
        try:
            n_local = len(indptr) - 1
        except TypeError:
            n_local = -1
        size = comm.Get_size()
        bounds = np.array([n_local, -n_local], dtype=np.int64)
        comm.Allreduce(MPI.IN_PLACE, bounds, op=MPI.MIN)
        if bounds[0] < 0:
            raise ValueError("global_operator needs valid CSR row pointers")
        if bounds[0] != -bounds[1]:
            raise ValueError("global_operator needs equal row partitions")
        self.n_local, self.n_global = n_local, n_local * size
        indices, indptr, _ = _validated_csr(indices, indptr, (n_local,) * size, comm)
        indices, indptr = _canonical_rows(indices, indptr)
        rows = np.repeat(np.arange(n_local), np.diff(indptr))
        # Own immutable structure, as a declared Pattern does. Caller mutation
        # must not invalidate the registered extraction layout.
        self.indices, self.indptr, self.rows = (
            np.frombuffer(a.tobytes(), dtype=a.dtype) for a in (indices, indptr, rows)
        )
        self.owned_colors, self.entry_colors, self.n_colors = distributed_coloring(
            self.indices, indptr, (n_local,) * size, comm
        )
        self.nnz = len(self.indices)
        self.max_nnz = int(comm.allreduce(self.nnz, op=MPI.MAX))
        self.shape = (n_local, self.n_global)
        # Entries grouped by colour, extracted in fixed-size tiles. Only the
        # small tile schedule is common to ranks; row IDs and CSR positions
        # remain rank-local. Unlike a (colours, largest colour) table, storage
        # does not multiply a skewed colour's size by the number of colours.
        counts = np.bincount(self.entry_colors, minlength=self.n_colors)
        largest = counts.astype(np.int64)
        comm.Allreduce(MPI.IN_PLACE, largest, op=MPI.MAX)
        self._tile_size = max(1, min(_PROBE_TILE_ENTRIES, int(largest.max(initial=0))))
        if max(n_local, self.max_nnz + self._tile_size) > np.iinfo(np.int32).max:
            raise ValueError("global_operator's local layout exceeds 32-bit indexing")
        tiles = (largest + self._tile_size - 1) // self._tile_size
        starts = np.r_[0, np.cumsum(tiles)]
        self._tile_colors = np.repeat(np.arange(self.n_colors, dtype=np.int32), tiles)
        self._tile_offsets = (
            (np.arange(starts[-1]) - np.repeat(starts[:-1], tiles)) * self._tile_size
        ).astype(np.int32)
        order = np.argsort(self.entry_colors, kind="stable")
        pad = self.max_nnz + self._tile_size - self.nnz
        device = mesh.local_devices[0]
        self._local = tuple(
            register_local(comm, array.astype(np.int32), device)
            for array in (
                self.owned_colors,
                np.pad(self.rows[order], (0, pad)),
                np.pad(order, (0, pad), constant_values=self.max_nnz),
                np.r_[0, np.cumsum(counts)],
            )
        )

    def with_fn(self, fn: Callable[[jax.Array], jax.Array]) -> GlobalOperator:
        """The same pattern and colouring with another function (new
        parameters), without repeating the setup collectives."""
        other = object.__new__(GlobalOperator)
        other.__dict__.update(self.__dict__)
        other.fn = fn
        return other

    def _packed_values(self, template: jax.Array) -> jax.Array:
        """Every rank's values, padded to the largest local count, as a global
        row-sharded array (the sharding interface's packed layout). The probes
        take ``template``'s dtype and sharding."""
        from jax._src import core as jax_core

        from .local_arrays import load_local

        spec = P(self.axis_name)
        axis = self.axis_name

        def packed(t):
            # Load metadata once, outside the loop. Each array has the same
            # shape, but its contents are owned by the executing rank.
            owned, rows, positions, pointers = jax.shard_map(
                lambda: tuple(load_local(h, varying_axis=axis) for h in self._local),
                mesh=self.mesh,
                in_specs=(),
                out_specs=(spec,) * 4,
            )()
            result_type = jax.eval_shape(self.fn, t)
            if result_type.shape != t.shape:
                raise ValueError(
                    "global_operator must return a vector of the input shape"
                )
            zeros = jax.shard_map(
                lambda _: jnp.zeros(self.max_nnz, dtype=result_type.dtype),
                mesh=self.mesh,
                in_specs=spec,
                out_specs=spec,
            )(t)
            if self.max_nnz == 0:
                return zeros

            def evaluate(c):
                probe = jax.shard_map(
                    lambda colors, c: (colors == c).astype(t.dtype),
                    mesh=self.mesh,
                    in_specs=(spec, P()),
                    out_specs=spec,
                )(owned, c)
                return self.fn(probe)

            def extract(out, y, rows, positions, pointers, c, offset):
                count = pointers[c + 1] - pointers[c]
                start = pointers[c] + jnp.minimum(offset, count)
                take = jax.lax.dynamic_slice_in_dim(rows, start, self._tile_size)
                ids = jax.lax.dynamic_slice_in_dim(positions, start, self._tile_size)
                lane = jnp.arange(self._tile_size)
                valid = lane < count - offset
                # Dropped updates leave the packed tail zero and never overwrite
                # another colour. Each real entry is written exactly once.
                # Keep even the dropped indices distinct for unique_indices.
                ids = jnp.where(valid, ids, self.max_nnz + lane)
                return out.at[ids].add(y[take], mode="drop", unique_indices=True)

            scatter = jax.shard_map(
                extract,
                mesh=self.mesh,
                in_specs=(spec, spec, spec, spec, spec, P(), P()),
                out_specs=spec,
            )

            def step(carry, tile):
                previous, out = carry
                c, offset = tile
                # All ranks use the same tile schedule, hence the same sequence
                # of operator collectives. Checkpoint from the colour index so
                # reverse mode can recreate probes instead of saving every one.
                y = jax.lax.cond(offset == 0, evaluate, lambda _: previous, c)
                return (y, scatter(out, y, rows, positions, pointers, c, offset)), None

            (_, result), _ = jax.lax.scan(
                jax.checkpoint(step),
                (jnp.zeros_like(t, dtype=result_type.dtype), zeros),
                (jnp.asarray(self._tile_colors), jnp.asarray(self._tile_offsets)),
            )
            return result

        # Always one jitted call: an eager linearization of the bare shard_maps
        # (under an eager grad) mismatches their integer residuals' cotangent
        # specs. Outside any transformation it runs under the operator's mesh
        # and owns the loaded data only while it lives. set_mesh is refused
        # while any transformation traces, even when ``template`` itself is
        # concrete (closed over), so test the trace state as set_mesh does.
        if not jax_core.trace_state_clean():
            return jax.jit(packed)(template)
        with jax.set_mesh(self.mesh):
            return jax.jit(packed)(template)

    def _local_matrix(self, template: jax.Array) -> jsp.BCSR:
        """This rank's rows (eager): values from the probes, global columns."""
        from .utils import temp_enable_x64

        packed = self._packed_values(template)
        values = packed.addressable_shards[0].data[: self.nnz]
        with temp_enable_x64():
            indices = jnp.asarray(self.indices, dtype=jnp.int64)
        return jsp.BCSR(
            (values, indices, jnp.asarray(self.indptr, dtype=jnp.int32)),
            shape=self.shape,
        )


def global_operator(
    fn: Callable[[jax.Array], jax.Array],
    indices: Any,
    indptr: Any,
    *,
    comm: Comm,
    mesh: Any,
    axis_name: str | tuple[str, ...] = "rank",
) -> GlobalOperator:
    """A global-view operator for the sharding interface (collective).

    ``fn`` maps a global row-sharded vector to a global row-sharded vector and
    performs its own communication. Each rank declares the pattern of the rows
    it owns; the columns are coloured once by the distributed colouring, and
    ``fn`` is probed with one global vector per colour.

    Args:
        fn: The operator, linear in its input.
        indices: Global column index of every entry of this rank's rows (CSR).
        indptr: CSR row pointers of this rank's rows.
        comm: The MPI communicator spanning the mesh's processes.
        mesh: The device mesh of the global vectors.
        axis_name: The mesh axis the rows are sharded over.

    Returns:
        A `GlobalOperator`, accepted by `jaxamg.make_sharded_matrix` and as a
        sharded solver's ``A=`` (``op.with_fn(new_fn)`` for new parameters,
        eagerly or inside transforms). Row partitions must be equal.
    """
    return GlobalOperator(
        fn, indices, indptr, comm=comm, mesh=mesh, axis_name=axis_name
    )
