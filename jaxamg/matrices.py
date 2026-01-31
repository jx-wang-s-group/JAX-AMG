"""
Standard test matrices and RHS vectors.

This module provides common sparse matrix patterns and right-hand-side
vector generators for testing and demonstration purposes.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np
import scipy.sparse as sp


def tridiagonal_matrix(n: int, diagonal_value: float = 2.0) -> jsp.BCSR:
    """Create a tridiagonal matrix in BCSR format with [-1, diagonal_value, -1] pattern.

    Args:
        n: Size of the matrix (n x n)
        diagonal_value: Value to place on the main diagonal (default 2.0)

    Returns:
        JAX BCSR matrix with [-1, diagonal_value, -1] pattern
    """
    # Total non-zeros: 2 + 3*(n-2) + 2 = 3*n - 2
    nnz = 3 * n - 2

    # Build values array efficiently using vectorized operations
    # Pattern: [diag, -1] + [-1, diag, -1] * (n-2) + [-1, diag]

    if n == 1:
        # Special case: 1x1 matrix
        values = jnp.array([diagonal_value], dtype=jnp.float32)
        indices = jnp.array([0], dtype=jnp.int32)
        indptr = jnp.array([0, 1], dtype=jnp.int32)
    elif n == 2:
        # Special case: 2x2 matrix
        values = jnp.array(
            [diagonal_value, -1.0, -1.0, diagonal_value], dtype=jnp.float32
        )
        indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    else:
        # General case: n >= 3
        # Build middle rows pattern: [-1, diag, -1] repeated (n-2) times
        middle_pattern = jnp.array([-1.0, diagonal_value, -1.0], dtype=jnp.float32)
        middle_values = jnp.tile(middle_pattern, n - 2)

        # Concatenate: first row + middle rows + last row
        values = jnp.concatenate(
            [
                jnp.array([diagonal_value, -1.0], dtype=jnp.float32),  # First row
                middle_values,  # Middle rows
                jnp.array([-1.0, diagonal_value], dtype=jnp.float32),  # Last row
            ]
        )

        # Build indices array
        # First row: [0, 1]
        # Middle rows: for row i: [i-1, i, i+1]
        middle_indices = jnp.stack(
            [
                jnp.arange(n - 2, dtype=jnp.int32),  # i-1
                jnp.arange(1, n - 1, dtype=jnp.int32),  # i
                jnp.arange(2, n, dtype=jnp.int32),  # i+1
            ],
            axis=1,
        ).ravel()

        # Last row: [n-2, n-1]
        indices = jnp.concatenate(
            [
                jnp.array([0, 1], dtype=jnp.int32),  # First row
                middle_indices,  # Middle rows
                jnp.array([n - 2, n - 1], dtype=jnp.int32),  # Last row
            ]
        )

        # Build indptr: [0, 2, 5, 8, ..., 3n-4, 3n-2]
        # First row has 2 entries, middle rows have 3 each, last row has 2
        row_lengths = jnp.concatenate(
            [
                jnp.array([2], dtype=jnp.int32),  # First row
                jnp.full(n - 2, 3, dtype=jnp.int32),  # Middle rows
                jnp.array([2], dtype=jnp.int32),  # Last row
            ]
        )
        indptr = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(row_lengths)]
        )

    return jsp.BCSR((values, indices, indptr), shape=(n, n))


def poisson_matrix(n: int) -> jsp.BCSR:
    """Create a 2D Poisson matrix on an n×n grid in BCSR format.

    Args:
        n: Grid size in each dimension (results in n² × n² matrix)

    Returns:
        JAX BCSR matrix representing the 2D Poisson operator
    """
    n2 = n * n  # Total size

    # Create grid indices using meshgrid
    i_grid, j_grid = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    row_indices = i_flat * n + j_flat

    # Build all entries using vectorized operations
    # For each grid point (i,j), we have up to 5 non-zeros:
    # - Diagonal: 4.0
    # - Left (j-1): -1.0 if j > 0
    # - Right (j+1): -1.0 if j < n-1
    # - Top (i-1): -1.0 if i > 0
    # - Bottom (i+1): -1.0 if i < n-1

    # Diagonal entries (always present)
    diag_rows = row_indices
    diag_cols = row_indices
    diag_vals = jnp.full(n2, 4.0, dtype=jnp.float32)

    # Left neighbors (j > 0)
    left_mask = j_flat > 0
    left_rows = row_indices[left_mask]
    left_cols = row_indices[left_mask] - 1
    left_vals = jnp.full(jnp.sum(left_mask), -1.0, dtype=jnp.float32)

    # Right neighbors (j < n-1)
    right_mask = j_flat < n - 1
    right_rows = row_indices[right_mask]
    right_cols = row_indices[right_mask] + 1
    right_vals = jnp.full(jnp.sum(right_mask), -1.0, dtype=jnp.float32)

    # Top neighbors (i > 0)
    top_mask = i_flat > 0
    top_rows = row_indices[top_mask]
    top_cols = row_indices[top_mask] - n
    top_vals = jnp.full(jnp.sum(top_mask), -1.0, dtype=jnp.float32)

    # Bottom neighbors (i < n-1)
    bottom_mask = i_flat < n - 1
    bottom_rows = row_indices[bottom_mask]
    bottom_cols = row_indices[bottom_mask] + n
    bottom_vals = jnp.full(jnp.sum(bottom_mask), -1.0, dtype=jnp.float32)

    # Concatenate all entries
    rows = jnp.concatenate([diag_rows, left_rows, right_rows, top_rows, bottom_rows])
    cols = jnp.concatenate([diag_cols, left_cols, right_cols, top_cols, bottom_cols])
    vals = jnp.concatenate([diag_vals, left_vals, right_vals, top_vals, bottom_vals])

    # Sort by (row, col)
    sort_idx = jnp.lexsort((cols, rows))
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    vals = vals[sort_idx]

    # Build indptr
    indptr = jnp.zeros(n2 + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows, length=n2)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts))

    return jsp.BCSR((vals, cols, indptr), shape=(n2, n2))


def tridiagonal_operator(diagonal_value: float = 2.0):
    """Create a tridiagonal operator with [-1, diagonal_value, -1] pattern."""
    kernel = jnp.array([-1.0, diagonal_value, -1.0])
    matvec = lambda x: jnp.convolve(x, kernel, mode="same")
    return matvec


def poisson_operator():
    """Create a 2D Poisson operator (flat input)."""
    kernel = jnp.array(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]], dtype=jnp.float32
    )

    def matvec(u_flat):
        size = u_flat.shape[0]
        n = int(size**0.5 + 0.5)

        if n * n != size:
            raise ValueError(f"Input size {size} is not a perfect square (n^2).")

        u = u_flat.reshape((n, n))
        # Boundary='fill', fillvalue=0.0 corresponds to Dirichlet BCs
        Au = jax.scipy.signal.convolve2d(
            u, kernel, mode="same", boundary="fill", fillvalue=0.0
        )
        return Au.ravel()

    return matvec


def rhs_ones(n: int):
    """Create a constant RHS vector of ones.

    Args:
        n: Vector length

    Returns:
        JAX array of ones (float32)
    """
    return jnp.ones(n, dtype=jnp.float32)


def rhs_linear(n: int):
    """Create a linearly increasing RHS vector.

    Args:
        n: Vector length

    Returns:
        JAX array with values linearly spaced from 0 to 1 (float32)
    """
    return jnp.linspace(0, 1, n, dtype=jnp.float32)


def rhs_random(n: int, seed: int = 0):
    """Create a random RHS vector.

    Args:
        n: Vector length
        seed: Random seed for reproducibility

    Returns:
        JAX array with random normal values (float32)
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n,), dtype=jnp.float32)
