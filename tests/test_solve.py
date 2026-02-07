"""Test basic solver functionality."""

import pytest
import numpy as np
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp

from jaxamg import amg_solve, AMGXStatus
from jaxamg.matrices import (
    tridiagonal_matrix,
    poisson_matrix,
    convection_diffusion_matrix_2d,
    rhs_ones,
)
from jaxamg.utils import to_scipy


class TestSolver:
    """Test basic solver functionality."""

    @pytest.mark.parametrize("n", [32, 256])
    def test_tridiagonal_solve(self, n):
        """Test solving a 1D tridiagonal system against analytical solution."""
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)
        x, info = amg_solve(A, b, solver="CG", max_iters=100)

        # Verify solver status
        if n == 256:
            # CG solver fails to converge for this size within 100 iterations
            assert info["status"] == AMGXStatus.NOT_CONVERGED
        else:
            # CG solver should converge for smaller sizes
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
        x, info = amg_solve(A, b, solver="CG", max_iters=1)

        assert info["status"] == AMGXStatus.NOT_CONVERGED
        assert info["iterations"] == 1

    def test_tridiagonal_solve_jit(self):
        """Test solving a 1D tridiagonal system with JIT compilation."""
        n = 32
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)

        # Create JIT-compiled version
        @jax.jit
        def solve_jit(b):
            x, _ = amg_solve(A, b, solver="CG")
            return x

        # Solve with JIT
        x_jit = solve_jit(b)

        # Solve without JIT
        x_nojit, _ = amg_solve(A, b)

        # Compare results
        np.testing.assert_allclose(x_jit, x_nojit, rtol=1e-6)

    def test_tridiagonal_solve_float64(self):
        """Test solving a 1D tridiagonal system with double precision."""

        # Override the default float32 precision
        jax.config.update("jax_enable_x64", True)

        n = 8
        A = tridiagonal_matrix(n)
        b = rhs_ones(n).astype(jnp.float64)
        x, info = amg_solve(A, b, solver="CG")

        assert info["status"] == AMGXStatus.SUCCESS
        np.testing.assert_allclose(b, A @ x)

        # Check that the solution is float64
        assert x.dtype == jnp.float64

        # Reset to default
        jax.config.update("jax_enable_x64", False)

    def test_tridiagonal_solve_float64_upcasting(self):
        """Test solving a 1D tridiagonal system with float64 matrix and float32 rhs."""

        # Override the default float32 precision
        jax.config.update("jax_enable_x64", True)

        n = 8

        # Explicitly cast matrix to float64
        A = tridiagonal_matrix(n)
        A = jsp.BCSR((A.data.astype(jnp.float64), A.indices, A.indptr), shape=A.shape)

        b = rhs_ones(n).astype(jnp.float32)
        x, info = amg_solve(A, b, solver="CG")

        assert info["status"] == AMGXStatus.SUCCESS
        np.testing.assert_allclose(b, A @ x)

        # Check that the solution is float64
        assert x.dtype == jnp.float64

        # Reset to default
        jax.config.update("jax_enable_x64", False)

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

    @pytest.mark.parametrize("solver", ["CG", "BICGSTAB", "GMRES"])
    def test_nonsymmetric_solve(self, solver):
        """Test solving 2D non-symmetric problem."""
        grid_size = 5
        A = poisson_matrix(grid_size, skew=1.0)
        n = grid_size**2
        b = rhs_ones(n)

        # Solve (with Jacobi preconditioner)
        x, info = amg_solve(A, b, solver=solver, preconditioner="JACOBI_L1")

        # CG should not converge, BICGSTAB and GMRES should converge
        if solver == "CG":
            assert info["status"] == AMGXStatus.NOT_CONVERGED
        else:
            assert info["status"] == AMGXStatus.SUCCESS

            # Check solution
            np.testing.assert_allclose(b, A @ x, rtol=1e-5)

    @pytest.mark.parametrize("solver", ["CG", "PBICGSTAB"])
    def test_convection_diffusion_solve(self, solver):
        """Test solving 2D convection-diffusion equation with analytic solution."""

        n = 64  # Grid size
        h = 1.0 / (n - 1)  # Grid spacing

        # Parameters
        epsilon = 1e-3
        velocity = 1.0
        theta = np.pi / 4.0

        # Velocity components
        vx = velocity * np.cos(theta)
        vy = velocity * np.sin(theta)

        # Grid
        x = jnp.linspace(0, 1, n)
        y = jnp.linspace(0, 1, n)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        # Analytic solution: u = sin(pi*x) * sin(pi*y)
        u_exact = jnp.sin(np.pi * X) * jnp.sin(np.pi * Y)
        laplacian_u = -2 * np.pi**2 * u_exact
        grad_u_x = np.pi * jnp.cos(np.pi * X) * jnp.sin(np.pi * Y)
        grad_u_y = np.pi * jnp.sin(np.pi * X) * jnp.cos(np.pi * Y)

        # Source term f = -epsilon*Delta u + v.Grad u
        f = -epsilon * laplacian_u + vx * grad_u_x + vy * grad_u_y
        b = f.ravel()
        u_exact_flat = u_exact.ravel()

        # Matrix
        A = convection_diffusion_matrix_2d(
            n, epsilon=epsilon, theta=theta, velocity=velocity
        )

        u, info = amg_solve(A, b, solver=solver)

        # Verify solver status
        if solver == "CG":
            # CG should not converge
            assert info["status"] == AMGXStatus.NOT_CONVERGED
        else:
            # PBICGSTAB with AMG preconditioner should converge
            assert info["status"] == AMGXStatus.SUCCESS

            # Verify solution
            np.testing.assert_allclose(A @ u, b, atol=1e-4)
            np.testing.assert_allclose(u, u_exact_flat, atol=0.1)
