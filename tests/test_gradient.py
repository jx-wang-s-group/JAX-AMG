"""Test automatic differentiation with AmgX solver."""
import pytest
import numpy as np
import scipy.sparse.linalg as spla
import jax
import jax.numpy as jnp
from jax.test_util import check_grads
from jaxamg import amgx_solve
from jaxamg.matrices import tridiagonal_matrix, rhs_ones, rhs_random


def l2_loss(data, b):
    """Compute L(b) = ||A^{-1} b||^2."""
    x = amgx_solve(data['row_ptrs'], data['col_indices'], data['values'], b)
    return jnp.sum(x * x)


def vdot_loss(data, v, b):
    """Compute L(b) = v^T A^{-1} b."""
    x = amgx_solve(data['row_ptrs'], data['col_indices'], data['values'], b)
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
        data = tridiagonal_matrix(n, diagonal_value=4.0) # Better conditioned
        b = rhs_ones(n)

        # Define loss function
        loss = lambda b: l2_loss(data, b)

        # Compute gradient with JAX
        grad_jax = jax.grad(loss)(b)

        # Compute gradient with SciPy (∂L/∂b = 2 * A^(-T) * x)
        x = amgx_solve(data['row_ptrs'], data['col_indices'], data['values'], b)
        grad_sp = 2.0 * spla.spsolve(data['A'].T.tocsr(), np.asarray(x))

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
        data = tridiagonal_matrix(n, diagonal_value=4.0) # Better conditioned
        b = rhs_ones(n)

        # Random vector for VJP
        rng = jax.random.PRNGKey(seed)
        v = jax.random.normal(rng, (len(b),), dtype=jnp.float32)

        # Define loss function
        loss = lambda b: vdot_loss(data, v, b)

        # Compute gradient with JAX
        grad_jax = jax.grad(loss)(b)

        # Compute gradient with SciPy
        grad_sp = spla.spsolve(data['A'].T.tocsr(), np.asarray(v))

        # Comparison with SciPy solution
        np.testing.assert_allclose(grad_jax, grad_sp, rtol=1e-5)

        # Comparison with finite differences
        check_grads(loss, (b,), order=1, modes=["rev"])

    def test_gradient_jit(self):
        """Test that gradients work with JIT compilation."""
        n = 32
        data = tridiagonal_matrix(n)
        b = rhs_random(n)

        # Define loss function
        loss = lambda b: l2_loss(data, b)

        # JIT-compiled gradient
        grad_fn_jit = jax.jit(jax.grad(loss))
        grad_fn = jax.grad(loss)

        g_jit = grad_fn_jit(b)
        g_nojit = grad_fn(b)

        # Comparison
        np.testing.assert_allclose(g_jit, g_nojit)
