"""Standard test matrices and RHS vectors for jaxamg.

This module provides common sparse matrix patterns (1D Laplacian, 2D Poisson)
and right-hand-side vector generators for testing and demonstration purposes.
"""
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp


def laplacian_1d(n: int, diagonal_value: float = 2.0) -> sp.csr_matrix:
    """Create a 1D Laplacian (tridiagonal) matrix in CSR format.

    Args:
        n: Size of the matrix (n x n)
        diagonal_value: Value to place on the main diagonal (default 2.0)
    Returns:
        CSR matrix with [-1, 2, -1] pattern on diagonals
    """
    return sp.diags([-1, diagonal_value, -1], offsets=[-1, 0, 1], shape=(n, n), format='csr', dtype=np.float32)

def csr_components(A: sp.csr_matrix):
    """Extract CSR components from a scipy sparse matrix.

    Args:
        A: Scipy CSR matrix

    Returns:
        Dictionary with keys:
            - 'A': Original CSR matrix
            - 'row_ptrs': JAX array of row pointers (int32)
            - 'col_indices': JAX array of column indices (int32)
            - 'values': JAX array of matrix values (float32)
    """
    return {
        "A": A,
        "row_ptrs": jnp.array(A.indptr, dtype=jnp.int32),
        "col_indices": jnp.array(A.indices, dtype=jnp.int32),
        "values": jnp.array(A.data, dtype=jnp.float32),
    }


def tridiagonal_matrix(n: int, diagonal_value: float = 2.0):
    """Create a 1D tridiagonal test matrix and CSR components.

    Args:
        n: Size of the matrix (n x n)

    Returns:
        Dictionary with matrix and CSR components (see csr_components)
    """
    return csr_components(laplacian_1d(n, diagonal_value=diagonal_value))


def poisson_matrix(n: int):
    """Create a 2D Poisson matrix using Kronecker sum.

    Constructs the discrete 2D Laplacian operator for an n×n grid
    using the Kronecker sum: A = L₁D ⊕ L₁D where L₁D is the 1D Laplacian.
    The resulting matrix has size (n²) × (n²).

    Args:
        n: Grid size in each dimension (results in n² × n² matrix)

    Returns:
        Dictionary with matrix and CSR components (see csr_components)
    """
    L1D = laplacian_1d(n)
    A = sp.kronsum(L1D, L1D, format='csr').astype(np.float32)
    return csr_components(A)


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


def rhs_random(n: int, seed: int=0):
    """Create a random RHS vector.

    Args:
        n: Vector length
        seed: Random seed for reproducibility

    Returns:
        JAX array with random normal values (float32)
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n,), dtype=jnp.float32)
