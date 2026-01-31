"""Tests for callable linear operator support."""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxamg import amg_solve
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
        x_op = amg_solve(A_op, b)
        x_csr = amg_solve(A_csr, b)

        # Check that the solutions are the same
        np.testing.assert_allclose(x_op, x_csr)

        # Check that the solution is correct
        np.testing.assert_allclose(A_op(x_op), b)

    def test_tridiagonal_operator_jit(self):
        n = 16

        diagonal_value = 4.0
        A = tridiagonal_operator(diagonal_value=diagonal_value)
        b = rhs_ones(n)

        # Solve without JIT
        # Also cache the coloring info for JIT compilation
        x_nojit = amg_solve(A, b)
        coloring_cache = A._amgx_coloring_info

        # Solve with JIT
        @jax.jit
        def solve(diagonal_value, b):
            A = tridiagonal_operator(diagonal_value=diagonal_value)
            object.__setattr__(A, "_amgx_coloring_info", coloring_cache)
            return amg_solve(A, b)

        x_jit = solve(diagonal_value, b)

        np.testing.assert_allclose(x_jit, x_nojit)

    def test_poisson_operator(self):
        n = 8

        A_op = poisson_operator()
        A_csr = poisson_matrix(n)
        b = rhs_ones(n * n)

        # Solve with operator and CSR matrix
        x_op = amg_solve(A_op, b)
        x_csr = amg_solve(A_csr, b)

        # Check that the solutions are the same
        np.testing.assert_allclose(x_op, x_csr)

        # Check that the solution is correct
        np.testing.assert_allclose(A_op(x_op), b, rtol=1e-5)
