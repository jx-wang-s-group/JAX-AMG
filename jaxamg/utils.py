"""
Helper functions for CSR matrix operations.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np
import scipy.sparse as sp
from collections import defaultdict


def from_scipy(A: sp.csr_matrix) -> jsp.CSR:
    """Convert a scipy CSR matrix to JAX CSR format.

    Args:
        A: Scipy CSR matrix

    Returns:
        JAX CSR matrix with int32 indices and float32 values
    """
    data = jnp.array(A.data, dtype=jnp.float32)
    indices = jnp.array(A.indices, dtype=jnp.int32)
    indptr = jnp.array(A.indptr, dtype=jnp.int32)
    return jsp.CSR((data, indices, indptr), shape=A.shape)


def to_scipy(A: jsp.CSR) -> sp.csr_matrix:
    """Convert a JAX CSR matrix to scipy CSR format.

    Args:
        A: JAX CSR matrix

    Returns:
        Scipy CSR matrix
    """
    return sp.csr_matrix(
        (np.asarray(A.data), np.asarray(A.indices), np.asarray(A.indptr)), shape=A.shape
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
        A_csr: jax.experimental.sparse.CSR matrix containing the values from A_callable.
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

    # Reconstruct CSR matrix with sorted indices
    sort_idx = jnp.lexsort((cols, rows))
    rows_sorted = rows[sort_idx]
    cols_sorted = cols[sort_idx]
    values_sorted = values[sort_idx]

    indptr = jnp.zeros(n + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows_sorted, length=n)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts))

    return jsp.CSR((values_sorted, cols_sorted, indptr), shape=shape)
