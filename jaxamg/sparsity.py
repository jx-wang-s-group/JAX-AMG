"""Sparsity detection and assembly for matrix-free operators.

Turns a callable operator ``A(x)`` into its sparse matrix: (1) detect the sparsity
pattern, (2) colour the columns, (3) materialise the values with one operator
evaluation per colour. ``cache_coloring`` orchestrates it, tries the two detection
methods in order, and verifies the result -- so it is correct for ANY operator:

- **Tracing** (``trace_sparsity_pattern``, in ``sparsity_tracing``): interpret the
  operator's jaxpr to recover the exact global pattern in a single trace. Returns
  ``None`` for operators that cannot be traced structurally.
- **Probing** (``probe_sparsity_pattern``): exhaustive one-hot basis-vector
  probing; the always-correct fallback when tracing is unavailable.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax.typing import ArrayLike

from .sparsity_tracing import trace_sparsity_pattern


# --- Probing-based detection: exhaustive one-hot basis-vector probing ---
def _probe_columns(
    A_callable: Callable, shape: tuple[int, int], tol: float
) -> tuple[np.ndarray, np.ndarray]:
    """Exhaustive probing with one-hot basis vectors (correct for any operator).

    Probes columns in batches and extracts the non-zeros per block (the full
    (m, n) matrix is never assembled), halving the batch on OOM. O(m) probes.
    """
    n, m = shape

    # Batch the probes with vmap when the operator supports it; fall back to a
    # sequential lax.map for operators that have no vmap rule (e.g. pure_callback
    # / FFI), matching materialize_sparse_matrix. Decide once via a cheap
    # eval_shape, which trips the missing-vmap-rule error without executing.
    try:
        jax.eval_shape(jax.vmap(A_callable), jax.ShapeDtypeStruct((1, m), jnp.float32))
        batched_A = jax.vmap(A_callable)
    except Exception:

        def batched_A(basis):
            return jax.lax.map(A_callable, basis)

    def _eval_batch(start: int, size: int) -> tuple[np.ndarray, np.ndarray]:
        indices = jnp.arange(start, start + size)
        basis = jax.nn.one_hot(indices, m, dtype=jnp.float32)  # (size, m)
        out = batched_A(basis)  # (size, n)
        if out.shape != (size, n):
            raise ValueError(
                f"Operator returned shape {out.shape}, expected ({size}, {n})."
            )
        # out[c, i] = A(e_{start+c})[i] = A[i, start + c].
        col_local, row = np.where(np.abs(np.array(out)) > tol)
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

    batch_size = m
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
    tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Determine the sparsity pattern of a linear operator by one-hot probing.

    Probes the operator with batches of one-hot basis vectors and extracts the
    nonzeros per block (the full (m, n) matrix is never assembled), halving the
    batch on OOM. Correct for any operator; this is the fallback used when
    jaxpr tracing is unavailable (opaque or data-dependent operators).

    Must be run outside of JIT compilation. Returns (rows, cols).
    """
    n, m = shape
    if n == 0 or m == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)
    return _probe_columns(A_callable, shape, tol)


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


def materialize_sparse_matrix(
    A_callable: Callable,
    shape: tuple[int, int],
    rows: ArrayLike,
    cols: ArrayLike,
    column_colors: ArrayLike,
    n_colors: int,
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
    try:
        rows_np = np.asarray(rows).astype(np.int32)
        cols_np = np.asarray(cols).astype(np.int32)
        colors_np = np.asarray(column_colors).astype(np.int32)
        static = True
    except Exception:
        static = False

    column_colors = jnp.array(column_colors, dtype=jnp.int32)

    def evaluate_color(color_id: ArrayLike) -> jax.Array:
        # Create probe vector v_c such that v_c[j] = 1 if color[j] == c, else 0
        mask = column_colors == color_id
        v = mask.astype(jnp.float32)
        w = A_callable(v)
        return w

    # Map over all colors: (n_colors, n)
    # Use lax.map instead of vmap to support primitives without batching rules (e.g. CSR matvec)
    w_matrix = jax.lax.map(evaluate_color, jnp.arange(n_colors))

    if static:
        # Host-side static CSR construction; only `values_sorted` is traced.
        order = np.lexsort((cols_np, rows_np))
        rows_sorted = rows_np[order]
        cols_sorted_np = cols_np[order]
        colors_for_cols_sorted = colors_np[cols_sorted_np]
        indptr_np = np.zeros(int(n) + 1, dtype=np.int32)
        indptr_np[1:] = np.cumsum(np.bincount(rows_sorted, minlength=int(n)))
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
    A_bcsr: jsp.BCSR, tol: float = 1e-9
) -> tuple[jsp.BCSR, np.ndarray, np.ndarray]:
    """Return (BCSR, rows, cols) with near-zero entries removed."""
    data = np.asarray(A_bcsr.data)
    indices = np.asarray(A_bcsr.indices)
    indptr = np.asarray(A_bcsr.indptr)
    n_rows = A_bcsr.shape[0]
    keep = np.abs(data) > tol
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
    operator: Callable, A_bcsr: jsp.BCSR, n_global: int, n_check: int = 5
) -> bool:
    """Check the recovered matrix reproduces the operator on random vectors.

    If A_bcsr != A (entries missing because the operator was not really
    translation-invariant, or boundary couplings were not captured) then
    (A_bcsr - A) v != 0 for almost every v, so a few random probes catch it.

    Probes follow the configured precision -- float64 when x64 is enabled (as in
    the distributed solvers), float32 otherwise -- with the tolerance loosened to
    match, so it neither warns about an unavailable dtype nor false-rejects a
    correct float32 recovery.
    """
    x64 = jax.config.jax_enable_x64
    dtype = jnp.float64 if x64 else jnp.float32
    tol = 1e-6 if x64 else 1e-4
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
    operator: Callable, n_local: int, n_global: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]] | None:
    """Exact sparsity from the operator's jaxpr (no probing), VERIFIED against the
    operator. Works for any JAX-expressed operator; returns None for operators
    that can't be traced structurally (opaque calls, data-dependent indexing),
    so the caller falls back to the probing detector.
    """
    try:
        pattern = trace_sparsity_pattern(operator, (n_local, n_global))
        if pattern is None:
            return None
        rows, cols = pattern
        if rows.size == 0:
            return None
        column_colors, n_colors = get_column_coloring(rows, cols, (n_local, n_global))
        A = materialize_sparse_matrix(
            operator, (n_local, n_global), rows, cols, column_colors, n_colors
        )
        # The pattern is exact; drop_zeros only removes structurally-present but
        # numerically-zero entries (e.g. a vanishing variable coefficient), and
        # verify is the safety net in case a transfer rule is wrong.
        A, final_rows, final_cols = _drop_zeros(A)
        if not _verify_recovery(operator, A, n_global):
            return None
        return (final_rows, final_cols, column_colors, n_colors, (n_local, n_global))
    except Exception:
        return None  # any failure -> probing fallback (correctness preserved)


def cache_coloring(
    operator: Any,
    shape: tuple[int, int] | int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, tuple[int, int]]:
    """
    Compute and cache coloring information for a callable operator.

    Detection uses two methods, so the result is correct for ANY operator:

    1. Tracing: interpret the operator's jaxpr to recover the EXACT sparsity in
       a single trace (no probing), then colour and materialise it. Works for any
       JAX-expressed operator; skipped for operators that can't be traced
       structurally (opaque calls, data-dependent indexing).
    2. Probing (``probe_sparsity_pattern`` + ``get_column_coloring``): exhaustive
       one-hot basis-vector probing, correct for any operator -- the fallback when
       tracing is unavailable.

    Args:
        operator: A callable operator A(x) that returns ``A @ x``.
        shape: Shape of the operator (n, m) or int size (for an n×n matrix). For a
            distributed operator this is the local block ``(n_local, n_global)``.

    Returns:
        Cached coloring information for reattachment with ``with_cache(..., coloring=...)``.
    """
    if isinstance(shape, int):
        shape = (shape, shape)

    existing_cache = getattr(operator, "_coloring_info", None)
    if existing_cache is not None:
        cached_shape = existing_cache[4]
        if cached_shape == shape:
            return existing_cache
        raise ValueError(
            f"Operator already has cached coloring for shape {cached_shape}, "
            f"but requested shape {shape}. Create a new operator instance."
        )

    n_local, n_global = shape

    # 1. Tracing (exact, any JAX operator). 2. Probing (any operator). Tracing
    # verifies before being accepted; probing is exact by construction.
    cache = _try_trace_coloring(operator, n_local, n_global)
    if cache is None:
        rows, cols = probe_sparsity_pattern(operator, shape)
        column_colors, n_colors = get_column_coloring(rows, cols, shape)
        cache = (rows, cols, column_colors, n_colors, shape)

    try:
        setattr(operator, "_coloring_info", cache)
    except Exception:
        pass

    return cache
