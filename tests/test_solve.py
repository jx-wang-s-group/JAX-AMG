"""Test basic solver functionality."""
import pytest
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import jax.numpy as jnp
from jaxamg import amgx_solve
from jaxamg.matrices import tridiagonal_matrix, poisson_matrix, rhs_ones


class TestSolver:
    """Test basic solver functionality."""

    @pytest.mark.parametrize("n", [32, 64])
    def test_tridiagonal_solve(self, n):
        """Test solving a 1D tridiagonal system against analytical solution."""
        data = tridiagonal_matrix(n)
        b = rhs_ones(n)
        x = amgx_solve(data['row_ptrs'], data['col_indices'], data['values'], b)

        # Verify that Ax = b
        np.testing.assert_allclose(b, data['A'] @ np.asarray(x))

        # Compare with solution from SciPy
        x_sp = spla.spsolve(data['A'], np.asarray(b)).astype(np.float32)
        np.testing.assert_allclose(np.asarray(x), x_sp, rtol=1e-5)

    def test_poisson_manufactured_solution(self):
        """Test 2D Poisson with manufactured solution."""
        grid_size = 8
        data = poisson_matrix(grid_size)
        row_ptrs, col_indices, values = data['row_ptrs'], data['col_indices'], data['values']
        n = grid_size**2

        # Create sparse matrix for verification
        A = sp.csr_matrix(
            (np.array(values, copy=True), np.array(col_indices, copy=True), np.array(row_ptrs, copy=True)),
            shape=(n, n)
        )

        # Manufactured solution: x = sin(πi/n) * cos(πj/n)
        x_true = np.zeros(n, dtype=np.float32)
        for idx in range(n):
            i = idx // grid_size
            j = idx % grid_size
            x_true[idx] = np.sin(np.pi * i / grid_size) * np.cos(np.pi * j / grid_size)

        # Compute b = A * x_true
        b = jnp.array(A @ x_true)

        # Solve
        x_computed = amgx_solve(row_ptrs, col_indices, values, b)

        # Compare with true solution
        np.testing.assert_allclose(x_computed, x_true, atol=1e-6)

    @pytest.mark.parametrize("grid_size", [4, 8, 16])
    def test_poisson_solve(self, grid_size):
        """Test solving 2D Poisson against SciPy solution."""
        data = poisson_matrix(grid_size)
        row_ptrs, col_indices, values = data['row_ptrs'], data['col_indices'], data['values']
        n = grid_size**2
        b = rhs_ones(n)

        # Solve
        x = amgx_solve(row_ptrs, col_indices, values, b)

        # Solve with Scipy
        A = sp.csr_matrix(
            (np.array(values, copy=True), np.array(col_indices, copy=True), np.array(row_ptrs, copy=True)),
            shape=(n, n)
        )
        x_sp = spla.spsolve(A, np.asarray(b))

        # Compare solutions
        np.testing.assert_allclose(x, x_sp, rtol=1e-6)

        # Check residual
        residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
        np.testing.assert_allclose(residual, 0.0, atol=1e-5)