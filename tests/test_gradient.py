"""Test automatic differentiation with AmgX solver."""
import pytest
import numpy as np
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp
from jax.test_util import check_grads

from jaxamg import amgx_solve
from jaxamg.matrices import tridiagonal_matrix, rhs_ones, rhs_random
from jaxamg.utils import to_scipy


def l2_loss(A, b):
    """Compute L(b) = ||A^{-1} b||^2."""
    x = amgx_solve(A, b)
    return jnp.sum(x * x)


def vdot_loss(A, v, b):
    """Compute L(b) = v^T A^{-1} b."""
    x = amgx_solve(A, b)
    return jnp.dot(v, x)


class TestGradient:
    """Test automatic differentiation functionality."""

    @pytest.mark.parametrize("n", [32, 64])
    def test_gradient(self, n):
        """
        Test gradient against analytical formula.

        For loss L(b) = ||x||^2 where x = A^{{-1}}b:
        ∂L/∂b = 2 * A^{{-T}} * x
        """
        A = tridiagonal_matrix(n, diagonal_value=4.0) # Better conditioned
        b = rhs_ones(n)

        # Define loss function
        loss = lambda b: l2_loss(A, b)

        # Compute gradient with JAX
        grad_jax = jax.grad(loss)(b)

        # Compute gradient with SciPy (∂L/∂b = 2 * A^(-T) * x)
        x = amgx_solve(A, b)
        # Convert JAX CSR to scipy for comparison
        A_sp = to_scipy(A)
        grad_sp = 2.0 * spla.spsolve(A_sp.T.tocsr(), np.asarray(x))

        # Comparison with SciPy solution
        np.testing.assert_allclose(grad_jax, grad_sp, rtol=1e-6)

        # Comparison with finite differences
        check_grads(loss, (b,), order=1, modes=["rev"])

    @pytest.mark.parametrize("seed", [0, 42, 123])
    def test_gradient_vector_product(self, seed):
        """
        Test gradient on vector-Jacobian product.

        For loss L(b) = v^T x where x = A^(-1) b and v is a vector:
        ∂L/∂b = A^(-T) v
        """
        n = 32
        A = tridiagonal_matrix(n, diagonal_value=4.0) # Better conditioned
        b = rhs_ones(n)

        # Random vector for VJP
        rng = jax.random.PRNGKey(seed)
        v = jax.random.normal(rng, (len(b),), dtype=jnp.float32)

        # Define loss function
        loss = lambda b: vdot_loss(A, v, b)

        # Compute gradient with JAX
        grad_jax = jax.grad(loss)(b)

        # Compute gradient with SciPy
        A_sp = to_scipy(A)
        grad_sp = spla.spsolve(A_sp.T.tocsr(), np.asarray(v))

        # Comparison with SciPy solution
        np.testing.assert_allclose(grad_jax, grad_sp, rtol=1e-5)

        # Comparison with finite differences
        check_grads(loss, (b,), order=1, modes=["rev"])

    def test_gradient_jit(self):
        """Test that gradients work with JIT compilation."""
        n = 32
        A = tridiagonal_matrix(n)
        b = rhs_random(n)

        # Define loss function
        loss = lambda b: l2_loss(A, b)

        # JIT-compiled gradient
        grad_fn_jit = jax.jit(jax.grad(loss))
        grad_fn = jax.grad(loss)

        g_jit = grad_fn_jit(b)
        g_nojit = grad_fn(b)

        # Comparison
        np.testing.assert_allclose(g_jit, g_nojit)
