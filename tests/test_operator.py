"""Tests for callable linear operator support."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxamg import amg_solve, cache_coloring, with_coloring, AMGXStatus
from jaxamg.matrices import (
    tridiagonal_matrix,
    tridiagonal_operator,
    poisson_matrix,
    poisson_operator,
    rhs_ones,
)


class TestOperator:
    def test_tridiagonal_operator(self):
        n = 16

        A_op = tridiagonal_operator()
        A_csr = tridiagonal_matrix(n)
        b = rhs_ones(n)

        # Solve with operator and CSR matrix
        x_op, _ = amg_solve(A_op, b)
        x_csr, _ = amg_solve(A_csr, b)

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
        coloring_cache = cache_coloring(A, size=n)

        # Solve with JIT using cached coloring
        @jax.jit
        def solve(diagonal_value, b):
            A = with_coloring(tridiagonal_operator(diagonal_value), coloring_cache)
            x, _ = amg_solve(A, b)
            return x

        x_jit = solve(diagonal_value, b)

        # Solve without JIT for comparison
        x_nojit, _ = amg_solve(A, b)

        np.testing.assert_allclose(x_jit, x_nojit)

    def test_poisson_operator(self):
        n = 8

        A_op = poisson_operator()
        A_csr = poisson_matrix(n)
        b = rhs_ones(n * n)

        # Solve with operator and CSR matrix
        x_op, _ = amg_solve(A_op, b)
        x_csr, _ = amg_solve(A_csr, b)

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
        x, info = amg_solve(A_op, b)

        # Should not converge
        assert info["status"] == AMGXStatus.NOT_CONVERGED

        # Solve with BiCGSTAB
        x, info = amg_solve(A_op, b, solver="BICGSTAB")

        # Should converge
        assert info["status"] == AMGXStatus.SUCCESS

        # Check that the solution satisfies Ax = b
        np.testing.assert_allclose(A_op(x), b, rtol=1e-5)
