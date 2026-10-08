"""Tests for callable linear operator support."""

import jax
import numpy as np
import pytest

import jaxamg
from jaxamg.matrices import (
    poisson_matrix,
    poisson_operator,
    rhs_ones,
    tridiagonal_matrix,
    tridiagonal_operator,
)

# Every test here calls the native AmgX solver (skip logic in conftest.py).
pytestmark = pytest.mark.gpu


class TestOperator:
    def test_tridiagonal_operator(self):
        n = 16

        A_op = tridiagonal_operator()
        A_csr = tridiagonal_matrix(n)
        b = rhs_ones(n)

        # Solve with operator and CSR matrix
        x_op, _ = jaxamg.solve(A_op, b, solver="CG")
        x_csr, _ = jaxamg.solve(A_csr, b, solver="CG")

        # Check that the solutions are the same
        np.testing.assert_allclose(x_op, x_csr)

        # Check that the solution is correct
        np.testing.assert_allclose(A_op(x_op), b)

    def test_tridiagonal_operator_jit(self):
        n = 16

        diagonal_value = 4.0
        A = tridiagonal_operator(diagonal_value=diagonal_value)
        b = rhs_ones(n)

        # Compute coloring cache
        coloring_cache = jaxamg.cache_coloring(A, shape=n)

        # Solve with JIT using cached coloring
        @jax.jit
        def solve(diagonal_value, b):
            A = jaxamg.with_cache(
                tridiagonal_operator(diagonal_value), coloring=coloring_cache
            )
            x, _ = jaxamg.solve(A, b, solver="CG")
            return x

        x_jit = solve(diagonal_value, b)

        # Solve without JIT for comparison
        x_nojit, _ = jaxamg.solve(A, b)

        np.testing.assert_allclose(x_jit, x_nojit, rtol=1e-6)

    def test_poisson_operator(self):
        n = 8

        A_op = poisson_operator()
        A_csr = poisson_matrix(n)
        b = rhs_ones(n * n)

        # Solve with operator and CSR matrix
        x_op, _ = jaxamg.solve(A_op, b, solver="CG")
        x_csr, _ = jaxamg.solve(A_csr, b, solver="CG")

        # Check that the solutions are the same
        np.testing.assert_allclose(x_op, x_csr)

        # Check that the solution is correct
        np.testing.assert_allclose(A_op(x_op), b, rtol=1e-5)

    def test_poisson_operator_nonsymmetric(self):
        """Test non-symmetric poisson_operator with skew parameter."""
        n = 5

        A_op = poisson_operator(skew=1.0)
        b = rhs_ones(n * n)

        # Solve with CG
        x, info = jaxamg.solve(A_op, b, solver="CG")

        # Should not converge
        assert info["status"] == jaxamg.AMGXStatus.NOT_CONVERGED

        # Solve with BiCGSTAB
        x, info = jaxamg.solve(A_op, b, solver="BICGSTAB")

        # Should converge
        assert info["status"] == jaxamg.AMGXStatus.SUCCESS

        # Check that the solution satisfies Ax = b
        np.testing.assert_allclose(A_op(x), b, rtol=1e-5)


class TestOperatorPrecision:
    """A float64 solve through an operator whose arithmetic follows its input
    dtype is as accurate as the same system given as an explicit float64 matrix."""

    @pytest.fixture
    def x64(self):
        jax.config.update("jax_enable_x64", True)
        yield
        jax.config.update("jax_enable_x64", False)

    def test_float64_operator_solve_accuracy(self, x64):
        import jax.experimental.sparse as jsp
        import jax.numpy as jnp

        n = 16
        rng = np.random.default_rng(0)
        k = jnp.asarray(np.exp(0.3 * rng.standard_normal((n, n))))
        kx, ky = 0.5 * (k[1:, :] + k[:-1, :]), 0.5 * (k[:, 1:] + k[:, :-1])

        def op(x):
            u = x.reshape(n, n)
            fx = kx.astype(x.dtype) * (u[1:, :] - u[:-1, :])
            fy = ky.astype(x.dtype) * (u[:, 1:] - u[:, :-1])
            y = jnp.zeros((n, n), x.dtype)
            y = (
                y.at[1:, :]
                .add(-fx)
                .at[:-1, :]
                .add(fx)
                .at[:, 1:]
                .add(-fy)
                .at[:, :-1]
                .add(fy)
            )
            return (0.37 * u - y).reshape(-1)

        dense = np.asarray(jax.vmap(op, in_axes=1, out_axes=1)(jnp.eye(n * n)))
        b = jnp.asarray(rng.standard_normal(n * n))
        reference = np.linalg.solve(dense, np.asarray(b))
        config = {
            "solver": "PCG",
            "preconditioner": {"solver": "AMG", "max_iters": 1},
            "tolerance": 1e-13,
            "max_iters": 400,
        }
        x_op, _ = jaxamg.solve(op, b, config=config)
        x_mat, _ = jaxamg.solve(
            jsp.BCSR.fromdense(jnp.asarray(dense)), b, config=config
        )
        error = lambda x: np.linalg.norm(np.asarray(x) - reference) / np.linalg.norm(
            reference
        )
        assert error(x_mat) < 1e-10
        assert error(x_op) < 1e-10
