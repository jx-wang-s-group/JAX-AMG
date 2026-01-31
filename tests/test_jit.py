"""Test JAX JIT compilation with AmgX solver."""
import jax
import numpy as np

from jaxamg import amgx_solve
from jaxamg.matrices import tridiagonal_matrix, rhs_ones


class TestJIT:
    """Test JIT compilation functionality."""

    def test_jit_compilation(self):
        """Test that solver works with JIT compilation."""
        n = 32
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)

        # Create JIT-compiled version
        @jax.jit
        def solve_jit(b):
            return amgx_solve(A, b)

        # Solve with JIT
        x_jit = solve_jit(b)

        # Solve without JIT
        x_nojit = amgx_solve(A, b)

        # Compare results
        np.testing.assert_allclose(x_jit, x_nojit)

    def test_jit_with_different_rhs(self):
        """Test JIT compilation with different RHS values."""
        n = 32
        A = tridiagonal_matrix(n)

        @jax.jit
        def solve_jit(b):
            return amgx_solve(A, b)

        # Solve with two different RHS
        b1 = rhs_ones(n)
        b2 = 2.0 * b1

        x1 = solve_jit(b1)
        x2 = solve_jit(b2)

        # Check that x2 = 2 * x1
        np.testing.assert_allclose(x2, 2.0 * x1)
