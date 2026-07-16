"""Test automatic differentiation with AmgX solver."""

from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax.test_util import check_grads

import jaxamg
from jaxamg.matrices import (
    poisson_operator,
    rhs_ones,
    rhs_random,
    tridiagonal_matrix,
    tridiagonal_operator,
)
from jaxamg.utils import to_scipy

# Every test here calls the native AmgX solver (skip logic in conftest.py).
pytestmark = pytest.mark.gpu


def l2_loss(A, b, config={"solver": "CG"}):
    """Compute L(b) = ||A^{-1} b||^2."""
    x, _ = jaxamg.solve(A, b, config=config)
    return jnp.sum(x * x)


def vdot_loss(A, v, b, config={"solver": "CG"}):
    """Compute L(b) = v^T A^{-1} b."""
    x, _ = jaxamg.solve(A, b, config=config)
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
        A = tridiagonal_matrix(n, diagonal_value=4.0)  # Better conditioned
        b = rhs_ones(n)

        # Define loss function
        loss = lambda b: l2_loss(A, b)

        # Compute gradient with JAX
        grad_jax = jax.grad(loss)(b)

        # Compute gradient with SciPy (∂L/∂b = 2 * A^(-T) * x)
        x, _ = jaxamg.solve(A, b, solver="CG")
        # Convert JAX CSR to scipy for comparison
        A_sp = cast(sp.csr_matrix, to_scipy(A))
        grad_sp = 2.0 * spla.spsolve(A_sp.T.tocsr(), np.asarray(x))

        # Comparison with SciPy solution
        np.testing.assert_allclose(grad_jax, grad_sp, atol=1e-6)

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
        A = tridiagonal_matrix(n, diagonal_value=4.0)  # Better conditioned
        b = rhs_ones(n)

        # Random vector for VJP
        rng = jax.random.PRNGKey(seed)
        v = jax.random.normal(rng, (len(b),), dtype=jnp.float32)

        # Define loss function
        loss = lambda b: vdot_loss(A, v, b)

        # Compute gradient with JAX
        grad_jax = jax.grad(loss)(b)

        # Compute gradient with SciPy
        A_sp = cast(sp.csr_matrix, to_scipy(A))
        grad_sp = spla.spsolve(A_sp.T.tocsr(), np.asarray(v))

        # Comparison with SciPy solution
        np.testing.assert_allclose(grad_jax, grad_sp, atol=1e-6)

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

    @pytest.mark.parametrize("is_symmetric", [True, False])
    def test_gradient_wrt_diagonal_value_csr(self, is_symmetric):
        """Test gradient of loss function with respect to diagonal value using CSR matrix."""
        n = 16
        b = rhs_ones(n)

        @jax.jit
        def loss(diagonal_value):
            A = tridiagonal_matrix(n, diagonal_value=diagonal_value)
            return l2_loss(jaxamg.with_cache(A, is_symmetric=is_symmetric), b)

        diagonal_value = 4.0

        # Compare automatic differentiation with finite differences
        # Both is_symmetric = True and False should work since the matrix is symmetric
        check_grads(loss, (diagonal_value,), order=1, modes=["rev"])

    def test_gradient_wrt_diagonal_value_operator(self):
        """Test gradient of loss function with respect to diagonal value using operator."""
        n = 16
        b = rhs_ones(n)

        # Compute coloring cache
        coloring_cache = jaxamg.cache_coloring(
            tridiagonal_operator(diagonal_value=1.0), shape=n
        )

        @jax.jit
        def loss(diagonal_value):
            A = jaxamg.with_cache(
                tridiagonal_operator(diagonal_value), coloring=coloring_cache
            )
            return l2_loss(A, b)

        diagonal_value = 4.0

        # Compare automatic differentiation with finite differences
        check_grads(loss, (diagonal_value,), order=1, modes=["rev"])

    @pytest.mark.parametrize("is_symmetric", [True, False])
    def test_gradient_wrt_skew_operator(self, is_symmetric):
        """Test gradient of loss function with respect to skew value in Poisson operator."""
        n = 4
        b = rhs_ones(n)

        # Compute coloring cache
        coloring_cache = jaxamg.cache_coloring(poisson_operator(skew=1.0), shape=n)

        @jax.jit()
        def loss(skew):
            A = jaxamg.with_cache(
                poisson_operator(skew=skew),
                coloring=coloring_cache,
                is_symmetric=is_symmetric,
            )
            config = {
                "solver": "PBICGSTAB",
                "preconditioner": {
                    "solver": "AMG",
                },
            }
            return l2_loss(A, b, config=config)

        skew = 4.0

        # Compare automatic differentiation with finite differences
        if not is_symmetric:
            check_grads(loss, (skew,), order=1, modes=["rev"])
        else:
            # Incorrect symmetry assumption leads to failure
            with pytest.raises(AssertionError):
                check_grads(loss, (skew,), order=1, modes=["rev"])
