"""
Standard test matrices and RHS vectors.

This module provides common sparse matrix patterns (1D Laplacian, 2D Poisson)
and right-hand-side vector generators for testing and demonstration purposes.
"""
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np
import scipy.sparse as sp
from .utils import from_scipy


def laplacian_1d(n: int, diagonal_value: float = 2.0) -> sp.csr_matrix:
    """Create a 1D Laplacian (tridiagonal) matrix in CSR format.

    Args:
        n: Size of the matrix (n x n)
        diagonal_value: Value to place on the main diagonal (default 2.0)
    Returns:
        CSR matrix with [-1, 2, -1] pattern on diagonals
    """
    return sp.diags([-1, diagonal_value, -1], offsets=[-1, 0, 1], shape=(n, n), format='csr', dtype=np.float32)


def tridiagonal_matrix(n: int, diagonal_value: float = 2.0) -> jsp.CSR:
    """Create a 1D tridiagonal test matrix.

    Args:
        n: Size of the matrix (n x n)
        diagonal_value: Value to place on the main diagonal (default 2.0)

    Returns:
        JAX CSR matrix with [-1, diagonal_value, -1] pattern
    """
    return from_scipy(laplacian_1d(n, diagonal_value=diagonal_value))


def poisson_matrix(n: int) -> jsp.CSR:
    """Create a 2D Poisson matrix using Kronecker sum.

    Constructs the discrete 2D Laplacian operator for an n×n grid
    using the Kronecker sum: A = L₁D ⊕ L₁D where L₁D is the 1D Laplacian.
    The resulting matrix has size (n²) × (n²).

    Args:
        n: Grid size in each dimension (results in n² × n² matrix)

    Returns:
        JAX CSR matrix representing the 2D Poisson operator
    """
    L1D = laplacian_1d(n)
    A = sp.kronsum(L1D, L1D, format='csr').astype(np.float32)
    return from_scipy(A)


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
