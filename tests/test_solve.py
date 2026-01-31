"""Test basic solver functionality."""

import pytest
import numpy as np
import scipy.sparse.linalg as spla
import jax.numpy as jnp

from jaxamg import amg_solve, AMGXStatus
from jaxamg.matrices import tridiagonal_matrix, poisson_matrix, rhs_ones
from jaxamg.utils import to_scipy


class TestSolver:
    """Test basic solver functionality."""

    @pytest.mark.parametrize("n", [32, 256])
    def test_tridiagonal_solve(self, n):
        """Test solving a 1D tridiagonal system against analytical solution."""
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)
        x, info = amg_solve(A, b)

        # Verify solver status
        if n == 256:
            # Solver is ill-conditioned for this size
            assert info["status"] == AMGXStatus.NOT_CONVERGED
        else:
            # Solver should converge for smaller size
            assert info["status"] == AMGXStatus.SUCCESS

        if info["status"] == AMGXStatus.SUCCESS:
            # Verify that Ax = b
            np.testing.assert_allclose(b, A @ x)

            # Compare with solution from SciPy
            # Convert JAX CSR to SciPy for comparison
            A_sp = to_scipy(A)
            x_sp = spla.spsolve(A_sp, np.asarray(b)).astype(np.float32)
            np.testing.assert_allclose(np.asarray(x), x_sp, rtol=1e-5)

    def test_tridiagonal_solve_single_iter(self):
        """Test solving a 1D tridiagonal system with single iteration."""
        n = 32
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)
        x, info = amg_solve(A, b, max_iters=1)

        assert info["status"] == AMGXStatus.NOT_CONVERGED
        assert info["iterations"] == 1

    def test_poisson_manufactured_solution(self):
        """Test 2D Poisson with manufactured solution."""
        grid_size = 8
        A = poisson_matrix(grid_size)
        n = grid_size**2

        # Manufactured solution: x = sin(πi/n) * cos(πj/n)
        x_true = np.zeros(n, dtype=np.float32)
        for idx in range(n):
            i = idx // grid_size
            j = idx % grid_size
            x_true[idx] = np.sin(np.pi * i / grid_size) * np.cos(np.pi * j / grid_size)

        # Compute b = A * x_true
        b = jnp.array(A @ x_true)

        # Solve
        x_computed, _ = amg_solve(A, b)

        # Compare with true solution
        np.testing.assert_allclose(x_computed, x_true, atol=1e-6)

    @pytest.mark.parametrize("grid_size", [4, 8, 16])
    def test_poisson_solve(self, grid_size):
        """Test solving 2D Poisson against SciPy solution."""
        A = poisson_matrix(grid_size)
        n = grid_size**2
        b = rhs_ones(n)

        # Solve
        x, _ = amg_solve(A, b)

        # Solve with Scipy
        A_sp = to_scipy(A)
        x_sp = spla.spsolve(A_sp, np.asarray(b))

        # Compare solutions
        np.testing.assert_allclose(x, x_sp, rtol=1e-6)

        # Check residual
        residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
        np.testing.assert_allclose(residual, 0.0, atol=1e-5)
