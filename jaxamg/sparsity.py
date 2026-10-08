"""Sparsity detection and assembly for matrix-free operators.

``cache_coloring`` detects an operator's pattern at its current values (by
tracing its jaxpr, else by one-hot probing), colours the columns, and
materializes the values with one operator evaluation per colour. Entries that
are zero at those values are dropped. Reuse a discovered pattern only while it
covers every parameter value in use.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.typing import ArrayLike

from .sparsity_tracing import trace_sparsity_pattern

# --- Probing-based detection: exhaustive one-hot basis-vector probing ---
# Device-memory budget for one batch of one-hot probes. The batch is sized from
# a per-probe footprint estimate (``_probe_batch_size``) so probing never forms
# an n_global x n_global buffer; the OOM-halving loop stays as the safety net.
_PROBE_BATCH_BYTES = 256 * 2**20


def _probe_dtype(dtype: Any = None) -> Any:
    """The floating dtype of probe vectors: ``dtype``, else the default float."""
    dtype = jnp.result_type(float) if dtype is None else jnp.dtype(dtype)
    if not jnp.issubdtype(dtype, jnp.floating):
        raise TypeError(f"probe dtype must be floating, got {dtype}")
    return dtype


class ColoringInfo(tuple):
    """``(rows, cols, colors, n_colors, shape)`` from :func:`cache_coloring`,
    remembering the input dtype the pattern was discovered at (``.dtype``).

    It unpacks as the plain 5-tuple. A dtype-dependent operator can have a
    different pattern at another dtype, so a solve at another dtype re-discovers
    it (or, inside a trace, refuses). A plain tuple attached by the caller carries
    no dtype and is used as given.
    """

    def __new__(cls, items: Any, dtype: Any):
        info = super().__new__(cls, tuple(items))
        info.dtype = jnp.dtype(dtype)
        return info

    def __reduce__(self):
        return (ColoringInfo, (tuple(self), self.dtype))


def _probe_batch_size(m: int, n: int, out_itemsize: int, in_itemsize: int = 4) -> int:
    """Initial batch size for one-hot probing from the device-memory budget.

    Per-probe footprint estimate: the one-hot input, the output, and up to
    two full-input-size intermediates at the output precision -- an operator
    partitioned from a global one (``global_op(x)[row_start:row_end]``) applies
    the global operator to the whole vector before slicing its rows. The batch
    size only sets the work per step: the (rows, cols) result is independent of it.
    """
    per_probe = in_itemsize * m + out_itemsize * (2 * m + n)
    return max(1, min(m, _PROBE_BATCH_BYTES // per_probe))


def _probe_columns(
    A_callable: Callable, shape: tuple[int, int], tol: float | None, dtype: Any = None
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive probing with one-hot basis vectors (correct for any operator).

    Probes columns in batches and extracts the non-zeros per block (the full
    (m, n) matrix is never assembled). Batches are sized to a device-memory
    budget, then halved on OOM. O(m) probes.
    """
    n, m = shape
    # Probe at the intended input dtype (default: the default float), so a
    # dtype-dependent operator is detected at the precision it will be solved at.
    probe_dtype = _probe_dtype(dtype)

    # Batch the probes with vmap when the operator supports it; fall back to a
    # sequential lax.map for operators that have no vmap rule (e.g. pure_callback
    # / FFI), matching materialize_sparse_matrix. Decide once via a cheap
    # eval_shape, which trips the missing-vmap-rule error without executing.
    try:
        jax.eval_shape(jax.vmap(A_callable), jax.ShapeDtypeStruct((1, m), probe_dtype))
        batched_A = jax.vmap(A_callable)
    except Exception:

        def batched_A(basis):
            return jax.lax.map(A_callable, basis)

    # Output precision, for sizing the probe batches (worst case if unknown).
    try:
        out_sds = jax.eval_shape(A_callable, jax.ShapeDtypeStruct((m,), probe_dtype))
        out_itemsize = int(np.dtype(out_sds.dtype).itemsize)
    except Exception:
        out_itemsize = 8

    def _eval_batch(start: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        indices = jnp.arange(start, start + size)
        basis = jax.nn.one_hot(indices, m, dtype=probe_dtype)  # (size, m)
        out = batched_A(basis)  # (size, n)
        if out.shape != (size, n):
            raise ValueError(
                f"Operator returned shape {out.shape}, expected ({size}, {n})."
            )
        # out[c, i] = A(e_{start+c})[i] = A[i, start + c].
        # Every exact nonzero is kept by default: a scale-free test, since an
        # absolute threshold drops all entries of an operator whose coefficients
        # are small (physical units, dt*nu). Extra entries only cost colours.
        values = np.abs(np.array(out))
        col_local, row = np.where(values != 0 if tol is None else values > tol)
        return row.astype(np.int32), (start + col_local).astype(np.int32)

    def _run(batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        blocks = [
            _eval_batch(start, min(batch_size, m - start))
            for start in range(0, m, batch_size)
        ]
        rows = np.concatenate([b[0] for b in blocks])
        cols = np.concatenate([b[1] for b in blocks])
        return rows, cols

    def _is_oom(e: Exception) -> bool:
        s = str(e).lower()
        return "resource exhausted" in s or "out of memory" in s or "oom" in s

    batch_size = _probe_batch_size(m, n, out_itemsize, np.dtype(probe_dtype).itemsize)
    result: tuple[np.ndarray, np.ndarray] | None = None
    while result is None and batch_size >= 1:
        try:
            result = _run(batch_size)
        except Exception as e:
            if _is_oom(e):
                batch_size //= 2
                if batch_size >= 1:
                    warnings.warn(
                        f"OOM in probe_sparsity_pattern; retrying with batch_size={batch_size}.",
                        stacklevel=2,
                    )
            else:
                raise

    if result is None:
        raise RuntimeError("OOM even with batch_size=1; operator may be too large.")
    return result


def probe_sparsity_pattern(
    A_callable: Callable,
    shape: tuple[int, int],
    tol: float | None = None,
    dtype: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine the sparsity pattern of a linear operator by one-hot probing.

    Probes the operator with batches of one-hot basis vectors and extracts the
    nonzeros per block (the full (m, n) matrix is never assembled). Batches are
    sized to a device-memory budget -- so an operator partitioned from a global
    one never forms an n_global x n_global buffer -- and halved on OOM. Correct
    for any operator; this is the fallback used when jaxpr tracing is unavailable
    (opaque or data-dependent operators).

    Probes use ``dtype`` (default: the default float dtype). ``tol=None`` keeps every exact
    nonzero; a number keeps ``|a| > tol`` (absolute, so scale-dependent).

    Must be run outside of JIT compilation. Returns (rows, cols).
    """
    n, m = shape
    if n == 0 or m == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return _probe_columns(A_callable, shape, tol, dtype)


# --- Column coloring and value materialization ---
def get_column_coloring(
    rows: np.ndarray, cols: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, int]:
    """
    Compute a coloring of the columns such that no two columns with the same color
    share a non-zero row, enabling simultaneous evaluation.

    Builds the column conflict graph via a sparse A^T A product (replaces the O(nnz²)
    Python loop), then runs the Jones-Plassman parallel greedy coloring: each round
    selects a maximal independent set (MIS) using a vectorized JAX scatter-max over
    random weights, assigns the current color to the whole MIS, and repeats.

    Returns:
        colors: array of shape (m,) where colors[j] is the color ID of column j.
        n_colors: total number of colors used.
    """
    n, m = shape

    if len(rows) == 0:
        return np.full(m, -1, dtype=np.int32), 0

    # Build column conflict graph: ATA[c1, c2] > 0 iff columns c1 and c2 share a row.
    ones = np.ones(len(rows), dtype=np.float32)
    A_bool = sp.csr_matrix((ones, (rows, cols)), shape=(n, m))
    ATA = (A_bool.T @ A_bool).tocsr()
    ATA.setdiag(0)
    ATA.eliminate_zeros()

    # Flat edge list: edge k goes from src_nodes[k] to dst_nodes[k].
    # Precomputed once; used every round for the scatter-max.
    nnz_per_col = np.diff(ATA.indptr)
    src_nodes = jnp.array(np.repeat(np.arange(m), nnz_per_col), dtype=jnp.int32)
    dst_nodes = jnp.array(ATA.indices, dtype=jnp.int32)
    has_edges = len(src_nodes) > 0

    # Jones-Plassman coloring
    # Each round: assign random weights, find MIS (nodes whose weight beats every
    # uncolored neighbor), color MIS with the current color, mark them done.
    colors = np.full(m, -1, dtype=np.int32)
    in_pattern = np.zeros(m, dtype=bool)
    in_pattern[np.unique(cols)] = True
    uncolored = in_pattern.copy()  # numpy mask; updated each round

    key = jax.random.PRNGKey(0)
    color_id = 0

    while uncolored.any():
        key, subkey = jax.random.split(key)
        # Weights in (0, 1] for uncolored nodes; 0 for already-colored nodes so
        # they can never dominate an uncolored neighbor in the max comparison.
        w = jax.random.uniform(subkey, (m,), minval=1e-7, maxval=1.0)
        w = w * jnp.array(uncolored, dtype=jnp.float32)

        # neighbor_max[c] = max weight among all neighbors of c.
        # Scatter-max over the flat edge list: O(nnz), no Python loops.
        if has_edges:
            neighbor_max = jnp.full(m, -jnp.inf).at[src_nodes].max(w[dst_nodes])
        else:
            neighbor_max = jnp.full(m, -jnp.inf)

        # MIS: uncolored nodes that beat every neighbor → valid independent set.
        mis_np = np.array(jnp.array(uncolored) & (w > neighbor_max))

        colors[mis_np] = color_id
        uncolored[mis_np] = False
        color_id += 1

    return colors, color_id


def csr_structure(
    rows: np.ndarray, cols: np.ndarray, n_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host-side CSR order of a (rows, cols) pattern: ``(rows_sorted,
    cols_sorted, indptr)``, the structure `materialize_sparse_matrix` returns."""
    # A pattern already in CSR order (row, then column, nondecreasing) is its own
    # stable sort: checking is O(nnz), the sort O(nnz log nnz). Only the
    # permutation is skipped; the gathers below return new arrays as before.
    r, c = np.asarray(rows), np.asarray(cols)
    ordered = r.size < 2 or bool(
        np.all((r[1:] > r[:-1]) | ((r[1:] == r[:-1]) & (c[1:] >= c[:-1])))
    )
    order = np.arange(r.size) if ordered else np.lexsort((cols, rows))
    rows_sorted = rows[order]
    cols_sorted = cols[order]
    indptr = np.zeros(int(n_rows) + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.bincount(rows_sorted, minlength=int(n_rows)))
    return rows_sorted, cols_sorted, indptr


class _MaterializationLayout(NamedTuple):
    """Concrete structural operands shared by materializations of one Pattern."""

    rows: jax.Array
    cols: jax.Array
    entry_colors: jax.Array
    indptr: jax.Array
    column_colors: jax.Array


def materialize_sparse_matrix(
    A_callable: Callable,
    shape: tuple[int, int],
    rows: ArrayLike,
    cols: ArrayLike,
    column_colors: ArrayLike,
    n_colors: int,
    dtype: Any = None,
    *,
    _layout: _MaterializationLayout | None = None,
) -> jsp.BCSR:
    """
    Materialize the values of a sparse matrix inside JIT using graph coloring.

    This reduces the number of operator evaluations from N (columns) to C (colors).

    Args:
        A_callable: The function A(x) -> y. Can be differentiated through.
        shape: (n, m)
        rows, cols: Fixed sparsity pattern indices (JAX or Numpy arrays).
        column_colors: Array mapping column index to color ID.
        n_colors: Number of colors.
        dtype: Floating dtype of the probe vectors, i.e. of the unknowns the
            operator is applied to (the solvers pass the RHS dtype). Defaults to
            the default float dtype (float64 under ``jax_enable_x64``). An
            operator whose arithmetic follows its input dtype is materialized at
            this precision.

    Returns:
        A_bcsr: jax.experimental.sparse.BCSR matrix containing the values from A_callable.
    """
    n, m = shape

    # The sparsity pattern (rows/cols/colours) is static -- known at cache time.
    # Compute the CSR ordering and row pointers on the host with NumPy so XLA
    # receives them as ready constants, instead of constant-folding a large
    # lexsort inside JIT (which dominates compile time at scale). Only the values
    # (the operator evaluations) stay traced. Falls back to the JAX path if the
    # indices arrive as tracers (not the normal case).
    static = False
    if _layout is None:
        try:
            rows_np = np.asarray(rows).astype(np.int32)
            cols_np = np.asarray(cols).astype(np.int32)
            colors_np = np.asarray(column_colors).astype(np.int32)
            static = True
        except Exception:
            pass
        column_colors = jnp.array(column_colors, dtype=jnp.int32)
    else:
        column_colors = _layout.column_colors
    probe_dtype = _probe_dtype(dtype)

    def evaluate_color(color_id: ArrayLike) -> jax.Array:
        # Create probe vector v_c such that v_c[j] = 1 if color[j] == c, else 0
        mask = column_colors == color_id
        v = mask.astype(probe_dtype)
        w = A_callable(v)
        return w

    # Map over all colors: (n_colors, n)
    # Use lax.map instead of vmap to support primitives without batching rules (e.g. CSR matvec)
    w_matrix = jax.lax.map(evaluate_color, jnp.arange(n_colors))

    if _layout is not None:
        values = w_matrix[_layout.entry_colors, _layout.rows]
        return jsp.BCSR((values, _layout.cols, _layout.indptr), shape=shape)

    if static:
        # Host-side static CSR construction; only `values_sorted` is traced.
        rows_sorted, cols_sorted_np, indptr_np = csr_structure(rows_np, cols_np, int(n))
        colors_for_cols_sorted = colors_np[cols_sorted_np]
        values_sorted = w_matrix[
            jnp.asarray(colors_for_cols_sorted), jnp.asarray(rows_sorted)
        ]
        return jsp.BCSR(
            (values_sorted, jnp.asarray(cols_sorted_np), jnp.asarray(indptr_np)),
            shape=shape,
        )

    # Fallback: indices are traced -> do the sort in JAX.
    rows = jnp.array(rows, dtype=jnp.int32)
    cols = jnp.array(cols, dtype=jnp.int32)
    colors_for_cols = column_colors[cols]
    values = w_matrix[colors_for_cols, rows]
    sort_idx = jnp.lexsort((cols, rows))
    cols_sorted = cols[sort_idx]
    values_sorted = values[sort_idx]
    indptr = jnp.zeros(int(n) + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows[sort_idx], length=n)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts).astype(jnp.int32))
    return jsp.BCSR((values_sorted, cols_sorted, indptr), shape=shape)


# --- Verification and orchestration (cache_coloring) ---
def _drop_zeros(
    A_bcsr: jsp.BCSR, tol: float | None = None
) -> tuple[jsp.BCSR, np.ndarray, np.ndarray]:
    """Return (BCSR, rows, cols) without the numerically zero entries.

    ``tol=None`` removes exact zeros only (a vanishing coefficient); an absolute
    threshold would also remove every entry of a small-scale operator.
    """
    data = np.asarray(A_bcsr.data)
    indices = np.asarray(A_bcsr.indices)
    indptr = np.asarray(A_bcsr.indptr)
    n_rows = A_bcsr.shape[0]
    keep = data != 0 if tol is None else np.abs(data) > tol
    row_of = np.repeat(np.arange(n_rows, dtype=np.int32), np.diff(indptr))
    new_indptr = np.zeros(n_rows + 1, dtype=np.int32)
    new_indptr[1:] = np.cumsum(np.bincount(row_of[keep], minlength=n_rows))
    A = jsp.BCSR(
        (
            jnp.asarray(data[keep]),
            jnp.asarray(indices[keep], dtype=jnp.int32),
            jnp.asarray(new_indptr),
        ),
        shape=A_bcsr.shape,
    )
    return A, row_of[keep], indices[keep].astype(np.int32)


def _verify_recovery(
    operator: Callable,
    A_bcsr: jsp.BCSR,
    n_global: int,
    n_check: int = 5,
    dtype: Any = None,
) -> bool:
    """Check the recovered matrix reproduces the operator on random vectors.

    If A_bcsr != A (entries missing because the operator was not really
    translation-invariant, or boundary couplings were not captured) then
    (A_bcsr - A) v != 0 for almost every v, so a few random probes catch it.

    Test vectors use the discovery dtype (default: the default float), with the
    tolerance loosened for float32 so a correct float32 recovery is not rejected.
    """
    dtype = _probe_dtype(dtype)
    tol = 1e-6 if dtype == jnp.float64 else 1e-4
    key = jax.random.PRNGKey(0)
    for _ in range(n_check):
        key, sub = jax.random.split(key)
        v = jax.random.normal(sub, (n_global,), dtype=dtype)
        y_op = np.asarray(operator(v))
        y_rec = np.asarray(A_bcsr @ v)
        if np.linalg.norm(y_rec - y_op) > tol * (np.linalg.norm(y_op) + 1e-30):
            return False
    return True


def _try_trace_coloring(
    operator: Callable, n_local: int, n_global: int, dtype: Any = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]] | None:
    """Sparsity from the operator's jaxpr (no probing), zero-dropped at its
    current values and checked numerically; None when the operator cannot be
    traced (opaque calls, data-dependent indexing), so the caller probes."""
    try:
        pattern = trace_sparsity_pattern(operator, (n_local, n_global), dtype=dtype)
        if pattern is None:
            return None
        rows, cols = pattern
        if rows.size == 0:
            return None
        column_colors, n_colors = get_column_coloring(rows, cols, (n_local, n_global))
        A = materialize_sparse_matrix(
            operator,
            (n_local, n_global),
            rows,
            cols,
            column_colors,
            n_colors,
            dtype=dtype,
        )
        # Entries zero at these values are dropped; the recovery check catches
        # a wrong transfer rule.
        A, final_rows, final_cols = _drop_zeros(A)
        if not _verify_recovery(operator, A, n_global, dtype=dtype):
            return None
        return (final_rows, final_cols, column_colors, n_colors, (n_local, n_global))
    except Exception:
        return None  # any failure -> probing fallback (correctness preserved)


def _dtype_matches(info: Any, dtype: Any) -> bool:
    """Whether a colouring may be used at ``dtype``: it carries no dtype (a plain
    tuple from the caller) or the same one. An explicit ``is None`` test: NumPy
    compares ``np.dtype('float64') == None`` as True."""
    discovered = getattr(info, "dtype", None)
    return discovered is None or jnp.dtype(discovered) == jnp.dtype(dtype)


def cache_coloring(
    operator: Any,
    shape: tuple[int, int] | int,
    dtype: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]]:
    """
    Compute and cache coloring information for a callable operator.

    The pattern is discovered for the operator's current values at ``dtype``:
    by tracing its jaxpr, else by one-hot probing (opaque calls, data-dependent
    indexing). Entries that are zero at these values are dropped, so reuse the
    colouring only while that pattern covers every parameter value used;
    declare it with ``with_cache(..., pattern=...)`` when couplings can appear
    later.

    Args:
        operator: A callable operator A(x) that returns ``A @ x``.
        shape: Shape of the operator (n, m) or int size (for an n×n matrix). For a
            distributed operator this is the local block ``(n_local, n_global)``.
        dtype: Floating dtype of the unknowns the operator will be applied to
            (default: the default float dtype). Discovery runs at this dtype, so a
            dtype-dependent operator gets the pattern it has at that precision.

    Returns:
        A :class:`ColoringInfo`: the 5-tuple ``(rows, cols, colors, n_colors,
        shape)`` for reattachment with ``with_cache(..., coloring=...)``, carrying
        the discovery ``dtype``. Results are cached on the operator per dtype.
    """
    dimensions: Any = (shape, shape) if np.ndim(shape) == 0 else shape
    # Static integers (a size computed with jnp, e.g. jnp.prod(grid.shape), must
    # not reach jitted comparisons as an array).
    n_local, n_global = (int(s) for s in dimensions)
    shape = (n_local, n_global)
    dtype = _probe_dtype(dtype)

    existing_cache = getattr(operator, "_coloring_info", None)
    if existing_cache is not None and existing_cache[4] != shape:
        raise ValueError(
            f"Operator already has cached coloring for shape {existing_cache[4]}, "
            f"but requested shape {shape}. Create a new operator instance."
        )
    by_dtype = dict(getattr(operator, "_coloring_by_dtype", None) or {})
    if dtype in by_dtype:
        return by_dtype[dtype]
    # A colouring attached without a dtype (a plain tuple) is used as given.
    if existing_cache is not None and _dtype_matches(existing_cache, dtype):
        return existing_cache

    n_local, n_global = shape

    # 1. Tracing (exact, any JAX operator). 2. Probing (any operator). Tracing
    # verifies before being accepted; probing is exact by construction.
    cache = _try_trace_coloring(operator, n_local, n_global, dtype)
    if cache is None:
        rows, cols = probe_sparsity_pattern(operator, shape, dtype=dtype)
        column_colors, n_colors = get_column_coloring(rows, cols, shape)
        cache = (rows, cols, column_colors, n_colors, shape)
    cache = ColoringInfo(cache, dtype)

    by_dtype[dtype] = cache
    try:
        setattr(operator, "_coloring_info", cache)
        setattr(operator, "_coloring_by_dtype", by_dtype)
    except Exception:
        pass

    return cache


def coloring_for(
    operator: Any, shape: tuple[int, int], dtype: Any, traced: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]]:
    """The colouring to materialize ``operator`` at ``dtype``.

    Uses the operator's cached colouring when it was discovered at ``dtype`` (or
    carries no dtype). Otherwise discovers it at ``dtype``, which needs concrete
    execution: inside a trace a missing or mismatched colouring is refused, since
    a dtype-dependent operator can have a different pattern at another dtype.
    """
    dtype = _probe_dtype(dtype)
    by_dtype = getattr(operator, "_coloring_by_dtype", None) or {}
    if dtype in by_dtype:
        return by_dtype[dtype]
    info = getattr(operator, "_coloring_info", None)
    if info is not None and _dtype_matches(info, dtype):
        return info
    if traced:
        if info is None:
            raise ValueError(
                "Callable operators must be pre-scanned before JIT compilation to "
                "determine sparsity.\nCall solve(A, b) once outside of JIT to "
                "compute and cache the sparsity pattern."
            )
        raise ValueError(
            f"the operator's cached colouring was discovered at {info.dtype}, but "
            f"this solve applies it to {dtype} unknowns; call "
            f"jaxamg.cache_coloring(op, shape, dtype={dtype}) outside of JIT first"
        )
    return cache_coloring(operator, shape, dtype=dtype)
