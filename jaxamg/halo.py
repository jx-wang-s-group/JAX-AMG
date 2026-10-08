"""Halo-form operators: a rank-local operator ``y_local = fn(x_local, x_ghost)``.

It is materialized per rank over ``[x_local | x_ghost]`` with the usual
pattern discovery and colouring, so no global-length vector is formed, and
its columns are mapped to global ids for the distributed solve. ``fn`` must
not communicate, and its parameters are its own rank's.
"""

from __future__ import annotations

from collections.abc import Callable
from operator import index
from typing import Any

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np


class HaloOperator:
    """A halo-form operator; build it with `halo_operator`."""

    def __init__(
        self,
        fn: Callable[[jax.Array, jax.Array], jax.Array],
        n_local: int,
        ghost_ids: Any,
        pattern: Any = None,
    ):
        if isinstance(n_local, bool) or index(n_local) < 0:
            raise ValueError("n_local must be a nonnegative integer")
        ghost_ids = np.asarray(ghost_ids)
        if ghost_ids.ndim != 1 or (ghost_ids.size and ghost_ids.dtype.kind not in "iu"):
            raise ValueError("ghost_ids must be a one-dimensional integer array")
        if ghost_ids.size and (
            np.any(ghost_ids < 0) or np.any(ghost_ids > np.iinfo(np.int64).max)
        ):
            raise ValueError("ghost_ids must be nonnegative and fit in int64")
        ghost_ids = np.frombuffer(ghost_ids.astype(np.int64).tobytes(), dtype=np.int64)
        if ghost_ids.size and np.any(np.diff(ghost_ids) <= 0):
            raise ValueError("ghost_ids must be ascending and unique")
        self.fn = fn
        self.n_local = int(n_local)
        self.ghost_ids = ghost_ids
        self.combined_shape = (self.n_local, self.n_local + int(ghost_ids.size))

        n = self.n_local

        def combined(z: jax.Array) -> jax.Array:
            return fn(z[:n], z[n:])

        self.combined = combined
        if pattern is not None:
            from .cache import with_cache

            # A declared pattern in the combined slot space.
            self.combined = with_cache(combined, pattern=pattern)

    def with_fn(self, fn: Callable[[jax.Array, jax.Array], jax.Array]) -> HaloOperator:
        """The same structure with another function (new parameters): the
        colouring discovered or declared for this operator carries over, so it
        is found once, eagerly, and reused inside traced solves."""
        other = HaloOperator(fn, self.n_local, self.ghost_ids)
        for attr in ("_coloring_info", "_coloring_by_dtype"):
            if hasattr(self.combined, attr):
                setattr(other.combined, attr, getattr(self.combined, attr))
        return other

    def _host_layout(self, row_start: int, dtype: Any, traced: bool):
        """The colouring and, on the host, the global-column CSR layout: the
        permutation from materialization order, sorted global columns and
        row pointers."""
        from .sparsity import coloring_for, csr_structure

        info = coloring_for(self.combined, self.combined_shape, dtype, traced=traced)
        rows, cols = info[0], info[1]
        # The order materialize_sparse_matrix returns (row, then slot).
        _, slot, indptr = csr_structure(
            np.asarray(rows).astype(np.int32),
            np.asarray(cols).astype(np.int32),
            self.n_local,
        )
        slot = np.asarray(slot).astype(np.int64)
        indptr = np.asarray(indptr).astype(np.int64)
        n = self.n_local
        global_cols = int(row_start) + slot
        ghost = slot >= n
        global_cols[ghost] = self.ghost_ids[slot[ghost] - n]
        row_of = np.repeat(np.arange(n), np.diff(indptr))
        order = np.lexsort((global_cols, row_of))
        return info, order, global_cols[order], indptr

    def _host_structure(
        self, row_start: int, dtype: Any, traced: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(indices, indptr)`` of this rank's rows, global columns, on the host."""
        _, _, indices, indptr = self._host_layout(row_start, dtype, traced)
        return indices, indptr

    def _local_matrix(
        self, row_start: int, nglobal: int, dtype: Any, traced: bool
    ) -> jsp.BCSR:
        """This rank's rows with global int64 columns, sorted within rows; the
        values stay differentiable in the operator's parameters."""
        from .sparsity import materialize_sparse_matrix
        from .utils import temp_enable_x64

        info, order, indices, indptr = self._host_layout(row_start, dtype, traced)
        rows, cols, colors, n_colors, _ = info
        values = materialize_sparse_matrix(
            self.combined,
            self.combined_shape,
            rows,
            cols,
            colors,
            n_colors,
            dtype=dtype,
        ).data
        if not np.array_equal(order, np.arange(order.size)):
            values = values[jnp.asarray(order)]
        with temp_enable_x64():
            indices = jnp.asarray(indices, dtype=jnp.int64)
        return jsp.BCSR(
            (values, indices, jnp.asarray(indptr, dtype=jnp.int32)),
            shape=(self.n_local, int(nglobal)),
        )


def halo_operator(
    fn: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    n_local: int,
    ghost_ids: Any,
    pattern: Any = None,
) -> HaloOperator:
    """A rank-local matrix-free operator for the distributed solves.

    ``fn(x_local, x_ghost)`` returns this rank's rows of ``A x`` from its owned
    entries of ``x`` and the ghost entries its rows reference. It is
    materialized per rank over ``[x_local | x_ghost]`` (no global-length vector
    and no distributed colouring) and solved like a local matrix with
    `jaxamg.solve` or the sharding interface.

    Args:
        fn: The operator, linear in its inputs; it must not communicate.
        n_local: The number of rows this rank owns.
        ghost_ids: Ascending global columns the rows reference outside this
            rank; ``x_ghost[k]`` holds column ``ghost_ids[k]``.
        pattern: Optional declared `jaxamg.Pattern` over the combined
            ``(n_local, n_local + len(ghost_ids))`` space. Without it the
            pattern is traced (else probed) on first use, which needs concrete
            parameters: derive per-step operators with `HaloOperator.with_fn`
            to reuse it inside ``jit`` and ``grad``.

    Returns:
        A `HaloOperator`.
    """
    return HaloOperator(fn, n_local, ghost_ids, pattern)
