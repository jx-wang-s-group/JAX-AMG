"""
Helper functions for BCSR matrix operations.
"""

import jax
from jax import core
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def to_scipy(A: jsp.BCSR, format="csr") -> sp.spmatrix:
    """Convert a JAX BCSR matrix to Scipy sparse matrix format.

    Args:
        A: JAX BCSR matrix
        format: Scipy sparse matrix format (default "csr")
                Supported: "csr", "csc", "coo", "lil", "dok", "bsr"

    Returns:
        Scipy sparse matrix in the specified format
    """
    # First convert to CSR (most efficient from BCSR)
    A_csr = sp.csr_matrix(
        (np.asarray(A.data), np.asarray(A.indices), np.asarray(A.indptr)), shape=A.shape
    )

    # Convert to requested format
    if format == "csr":
        return A_csr
    elif format == "csc":
        return A_csr.tocsc()
    elif format == "coo":
        return A_csr.tocoo()
    elif format == "lil":
        return A_csr.tolil()
    elif format == "dok":
        return A_csr.todok()
    elif format == "bsr":
        return A_csr.tobsr()
    else:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats: csr, csc, coo, lil, dok, bsr"
        )


def get_sparsity_pattern(A_callable, shape, tol=1e-10):
    """
    Determine the sparsity pattern of a linear operator by applying it to basis vectors.
    Returns (rows, cols) of non-zero entries.

    This function must be run outside of JIT compilation.
    """
    n, m = shape
    rows_list = []
    cols_list = []

    for j in range(m):
        e_j = np.zeros(m, dtype=np.float32)
        e_j[j] = 1.0

        # Assume A_callable can handle a JAX array and return one
        col_val = A_callable(jnp.array(e_j))

        if col_val.shape[0] != n:
            raise ValueError(
                f"Operator returned output of shape {col_val.shape}, expected first dimension {n}."
            )

        # Identify non-zeros on host
        col_val_np = np.array(col_val)
        mask = np.abs(col_val_np) > tol
        nz_rows = np.flatnonzero(mask)

        if len(nz_rows) > 0:
            rows_list.append(nz_rows)
            cols_list.append(np.full(len(nz_rows), j, dtype=np.int32))

    if not rows_list:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)
    return rows, cols


def get_column_coloring(rows, cols, shape):
    """
    Compute a coloring of the columns such that no two columns with the same color
    share a non-zero row, enabling simultaneous evaluation.

    Returns:
        colors: array of shape (m,) where colors[j] is the color ID of column j.
        n_colors: total number of colors used.
    """
    n, m = shape

    # Build adjacency (conflict) graph of columns
    # Two columns conflict if they share a non-zero entry in the same row.
    row_to_cols = defaultdict(list)
    for r, c in zip(rows, cols):
        row_to_cols[r].append(c)

    adjacency = defaultdict(set)
    for r, columns_in_row in row_to_cols.items():
        # All columns in this row conflict with each other
        for i in range(len(columns_in_row)):
            u = columns_in_row[i]
            for j in range(i + 1, len(columns_in_row)):
                v = columns_in_row[j]
                adjacency[u].add(v)
                adjacency[v].add(u)

    # Greedy Coloring
    # Sorting columns by generic ID or degree usually sufficient for structured grids.
    column_colors = {}
    n_colors = 0
    all_cols = sorted(list(set(cols)))

    for u in all_cols:
        neighbor_colors = {column_colors[v] for v in adjacency[u] if v in column_colors}

        # Assign lowest available color
        color = 0
        while color in neighbor_colors:
            color += 1
        column_colors[u] = color
        n_colors = max(n_colors, color + 1)

    # Format Output
    final_colors = np.full(m, -1, dtype=np.int32)
    for c, color in column_colors.items():
        final_colors[c] = color

    return final_colors, n_colors


def materialize_sparse_matrix(A_callable, shape, rows, cols, column_colors, n_colors):
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

    # Ensure indices are JAX arrays
    rows = jnp.array(rows, dtype=jnp.int32)
    cols = jnp.array(cols, dtype=jnp.int32)
    column_colors = jnp.array(column_colors, dtype=jnp.int32)

    def evaluate_color(color_id):
        # Create probe vector v_c such that v_c[j] = 1 if color[j] == c, else 0
        mask = column_colors == color_id
        v = mask.astype(jnp.float32)
        w = A_callable(v)
        return w

    # Map over all colors: (n_colors, n)
    # Use lax.map instead of vmap to support primitives without batching rules (e.g. CSR matvec)
    w_matrix = jax.lax.map(evaluate_color, jnp.arange(n_colors))

    # Extract values: w_matrix[color[j], row] corresponds to A[row, j]
    colors_for_cols = column_colors[cols]
    values = w_matrix[colors_for_cols, rows]

    # Reconstruct BCSR matrix with sorted indices
    sort_idx = jnp.lexsort((cols, rows))
    rows_sorted = rows[sort_idx]
    cols_sorted = cols[sort_idx]
    values_sorted = values[sort_idx]

    indptr = jnp.zeros(n + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows_sorted, length=n)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts))

    return jsp.BCSR((values_sorted, cols_sorted, indptr), shape=shape)


def get_preferred_dtype(A, b):
    """
    Determine the preferred precision (float32 or float64) for the solver.

    Logic:
    - Default is float32.
    - If b is float64, use float64.
    - If A provides float64 data, use float64 (promoting b if necessary).
    """
    target_dtype = jnp.float32
    if b is not None:
        target_dtype = b.dtype

    # If A suggests float64, prefer that
    if hasattr(A, "dtype"):
        # e.g. JAX arrays, sparse matrices
        if A.dtype == jnp.float64 or A.dtype == np.float64:
            target_dtype = jnp.float64
    elif hasattr(A, "data") and hasattr(A.data, "dtype"):
        # e.g. BCOO, BCSR
        if A.data.dtype == jnp.float64 or A.data.dtype == np.float64:
            target_dtype = jnp.float64

    return target_dtype


def _ensure_bcsr_properties(A, target_dtype):
    """Ensure BCSR matrix has correct data type and int32 indices."""
    # check data type
    if A.data.dtype != target_dtype:
        A = jsp.BCSR(
            (A.data.astype(target_dtype), A.indices, A.indptr),
            shape=A.shape,
        )

    # check indices type
    if A.indices.dtype != jnp.int32:
        try:
            A = jsp.BCSR(
                (A.data, A.indices.astype(jnp.int32), A.indptr),
                shape=A.shape,
            )
        except Exception:
            raise ValueError(
                f"Matrix column indices must be int32, got {A.indices.dtype}."
            )

    # check indptr type
    if A.indptr.dtype != jnp.int32:
        try:
            A = jsp.BCSR(
                (A.data, A.indices, A.indptr.astype(jnp.int32)),
                shape=A.shape,
            )
        except Exception:
            raise ValueError(
                f"Matrix row pointers must be int32, got {A.indptr.dtype}."
            )

    if A.data.dtype not in [jnp.float32, jnp.float64]:
        raise ValueError(
            f"Matrix values must be float32 or float64, got {A.data.dtype}."
        )

    return A


def to_bcsr_matrix(A, *, b=None):
    """
    Normalize input 'A' to a BCSR matrix.

    Supports multiple input formats, including BCSR, BCOO, SciPy sparse (CSR, CSC, COO), dense arrays (NumPy, JAX), and callable operators.
    """
    # 1. Handle Callables (materialize via graph coloring)
    if callable(A):
        # Check for cached coloring info attached to the callable
        cached_info = getattr(A, "_amgx_coloring_info", None)

        if b is None:
            raise TypeError("Callable A requires RHS b to infer size.")
        if b.ndim != 1:
            raise ValueError(f"RHS b must be 1D, got shape {b.shape}.")
        shape = (int(b.shape[0]), int(b.shape[0]))

        is_jit = isinstance(b, core.Tracer)

        if cached_info is None:
            if is_jit:
                # Inside JIT without cache: Impossible to determine sparsity dynamically.
                raise ValueError(
                    "Callable operators must be pre-scanned before JIT compilation to determine sparsity.\n"
                    "Call amg_solve(A, b) once outside of JIT to compute and cache the sparsity pattern."
                )

            # Outside JIT: Compute sparsity and coloring (expensive O(N))
            rows, cols = get_sparsity_pattern(A, shape)
            column_colors, n_colors = get_column_coloring(rows, cols, shape)

            cached_info = (rows, cols, column_colors, n_colors, shape)
            try:
                setattr(A, "_amgx_coloring_info", cached_info)
            except Exception:
                pass  # Ignore if caching fails (e.g. on partials or immutable objects)

        rows, cols, column_colors, n_colors, cached_shape = cached_info

        if shape != cached_shape:
            raise ValueError(f"Operator shape changed from {cached_shape} to {shape}.")

        # Materialize using graph coloring (works efficienty inside JIT)
        # Note: materialize_sparse_matrix already returns a BCSR
        A = materialize_sparse_matrix(A, shape, rows, cols, column_colors, n_colors)

    # 2. Convert to BCSR from other formats
    target_dtype = get_preferred_dtype(A, b)

    if isinstance(A, jsp.BCSR):
        pass  # Already BCSR
    elif isinstance(A, jsp.BCOO):
        A = jsp.BCSR.from_bcoo(A)
    elif sp.issparse(A):
        # Use numpy dtype for scipy conversion
        np_dtype = np.float64 if target_dtype == jnp.float64 else np.float32
        A = jsp.BCSR.from_scipy_sparse(A.astype(np_dtype))
    elif isinstance(A, (np.ndarray, jnp.ndarray)):
        if A.ndim != 2:
            raise ValueError(f"Dense matrix must be 2D, got shape {A.shape}")
        if isinstance(A, np.ndarray):
            A = jnp.array(
                A
            )  # Ensure JAX array before fromdense if needed, or just let fromdense handle it
        A = jsp.BCSR.fromdense(A)
    else:
        raise TypeError(
            f"Matrix A must be one of: BCSR, BCOO, SciPy sparse, dense array (NumPy/JAX), or callable. "
            f"Got {type(A).__name__}."
        )

    # 3. Standardize properties
    return _ensure_bcsr_properties(A, target_dtype)
