"""Immutable declarations of an operator's possible sparse couplings."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .sparsity import _MaterializationLayout, csr_structure, get_column_coloring


def _integers(values: Any, name: str) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or (values.size and values.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return values


def _coordinates(rows, cols, shape):
    try:
        n_rows, n_cols = (operator.index(n) for n in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError("shape must contain two integer dimensions") from exc
    limit = np.iinfo(np.int32).max
    if not (0 <= n_rows <= limit and 0 <= n_cols <= limit):
        raise ValueError("pattern dimensions must be nonnegative and fit in int32")
    rows, cols = _integers(rows, "rows"), _integers(cols, "cols")
    if rows.shape != cols.shape:
        raise ValueError("rows and cols must have the same length")
    if rows.size and (
        rows.min() < 0 or rows.max() >= n_rows or cols.min() < 0 or cols.max() >= n_cols
    ):
        raise ValueError(f"pattern entries lie outside the shape {(n_rows, n_cols)}")
    rows, cols = rows.astype(np.int32), cols.astype(np.int32)
    # A declaration is a set. Canonicalize once, skipping the sort for CSR order.
    ordered = np.all(
        (rows[1:] > rows[:-1]) | ((rows[1:] == rows[:-1]) & (cols[1:] > cols[:-1]))
    )
    if not ordered:
        rows, cols = np.unique(np.stack((rows, cols)), axis=1)
    if rows.size > limit:
        raise ValueError("pattern entry count must fit in int32 row pointers")
    return rows, cols, (n_rows, n_cols)


def _immutable(array: np.ndarray) -> np.ndarray:
    # Immutable backing storage also prevents re-enabling NumPy's write flag.
    return np.frombuffer(array.astype(np.int32, copy=False).tobytes(), dtype=np.int32)


@dataclass(frozen=True, eq=False)
class Pattern:
    """Declared entries and column colouring of a linear operator.

    Coordinates use the operator's output rows and input slots, with shape
    ``(n_rows, n_columns)`` (local rows and global columns for an MPI block).
    Entries are canonicalized as a set and retained even when their values are
    zero. The declaration must cover every parameter value and precision in use.

    Prefer :func:`pattern` to compute the colouring. Direct construction accepts
    a colour per input column and a colour count; used columns must have colours
    in ``[0, n_colors)`` and columns sharing a row must have different colours.
    Both constructors copy their inputs into immutable host arrays.
    """

    rows: np.ndarray
    cols: np.ndarray
    shape: tuple[int, int]
    colors: np.ndarray
    n_colors: int
    _layout: _MaterializationLayout | None = field(default=None, init=False, repr=False)
    _layout_lock: Any = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self):
        rows, cols, shape = _coordinates(self.rows, self.cols, self.shape)
        colors = _integers(self.colors, "colors")
        if colors.size != shape[1]:
            raise ValueError("colors must contain one entry per input column")
        n_colors = operator.index(self.n_colors)
        if not 0 <= n_colors <= np.iinfo(np.int32).max:
            raise ValueError("n_colors must be nonnegative and fit in int32")
        if colors.size and (colors.min() < -1 or colors.max() >= n_colors):
            raise ValueError("colors must lie in [-1, n_colors)")
        entry_colors = colors[cols]
        if np.any(entry_colors < 0):
            raise ValueError("every used column must have a nonnegative color")
        order = np.lexsort((entry_colors, rows))
        r, c = rows[order], entry_colors[order]
        if np.any((r[1:] == r[:-1]) & (c[1:] == c[:-1])):
            raise ValueError("columns sharing a row must have different colors")
        for name, value in (("rows", rows), ("cols", cols), ("colors", colors)):
            object.__setattr__(self, name, _immutable(value))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "n_colors", n_colors)

    def _coloring(self) -> tuple:
        """The dtype-independent 5-tuple accepted by ``with_cache``."""
        return self.rows, self.cols, self.colors, self.n_colors, self.shape

    def _materialization_layout(self) -> _MaterializationLayout:
        """Create structural arrays once, including when first used in a trace.

        Do not retain tracers, probes, operator values or a floating dtype. Arrays
        are uncommitted so other JIT/device contexts can capture this declaration.
        The lock prevents concurrent first traces from capturing different copies.
        """
        if self._layout is None:
            with self._layout_lock:
                if self._layout is None:
                    with jax.ensure_compile_time_eval():
                        rows, cols, indptr = csr_structure(
                            self.rows, self.cols, self.shape[0]
                        )
                        layout = _MaterializationLayout(
                            jnp.asarray(rows),
                            jnp.asarray(cols),
                            jnp.asarray(self.colors[cols]),
                            jnp.asarray(indptr),
                            jnp.asarray(self.colors),
                        )
                    object.__setattr__(self, "_layout", layout)
        assert self._layout is not None
        return self._layout

    def __reduce__(self):
        # Serialize the declaration, not device arrays or synchronization state.
        return type(self), (
            self.rows,
            self.cols,
            self.shape,
            self.colors,
            self.n_colors,
        )


def pattern(rows: Any, cols: Any, shape: tuple[int, int]) -> Pattern:
    """Declare all possible sparse entries and compute their column colouring.

    ``rows`` and ``cols`` are one-dimensional integer coordinates in the
    operator's ``shape``. Unsorted coordinates and repeated entries are accepted.
    Attach the result with ``with_cache(op, pattern=p)``; numerical zeros do not
    remove declared entries. Construct it outside JAX transformations.
    """
    rows, cols, shape = _coordinates(rows, cols, shape)
    colors, n_colors = get_column_coloring(rows, cols, shape)
    return Pattern(rows, cols, shape, colors, n_colors)
