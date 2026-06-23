"""Tests for tracing (jaxpr-tracing) sparsity detection."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxamg.matrices import poisson3d_operator, poisson_operator, tridiagonal_operator
from jaxamg.sparsity import (
    _try_trace_coloring,
    cache_coloring,
    materialize_sparse_matrix,
    trace_sparsity_pattern,
)


def _dense_pattern(op, n, tol=1e-5):
    J = np.asarray(jax.jacfwd(op)(jnp.ones(n)))
    r, c = np.nonzero(np.abs(J) > tol)
    return set(zip(r.tolist(), c.tolist()))


def _trace_set(op, shape):
    res = trace_sparsity_pattern(op, shape)
    if res is None:
        return None
    return set(zip(res[0].tolist(), res[1].tolist()))


class TestTracingExact:
    def test_poisson3d_scatter_add(self):
        # 3D 7-point stencil via .at[].add -> exercises the scatter-add rule.
        n = 6**3
        op = poisson3d_operator(robin=2.0)
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    @pytest.mark.parametrize("opf", [tridiagonal_operator(), poisson_operator()])
    def test_convolution_operators(self, opf):
        # jnp.convolve -> conv_general_dilated rule.
        n = 16
        assert _trace_set(opf, (n, n)) == _dense_pattern(opf, n)

    def test_distributed_block(self):
        # Local row block of a distributed operator: op(x_global)[rs:re].
        n, rs, re = 6**3, 70, 150
        base = poisson3d_operator(robin=2.0)
        loc = lambda xg: base(xg)[rs:re]
        J = np.asarray(jax.jacfwd(base)(jnp.ones(n)))[rs:re]
        r, c = np.nonzero(np.abs(J) > 1e-5)
        assert _trace_set(loc, (re - rs, n)) == set(zip(r.tolist(), c.tolist()))

    def test_variable_coefficient(self):
        # Position-dependent (non-translation-invariant) coefficient.
        n = 5**3
        base = poisson3d_operator(robin=1.0)
        c = jax.random.uniform(jax.random.PRNGKey(0), (n,)) + 0.5
        vop = lambda u: c * base(u)
        assert _trace_set(vop, (n, n)) == _dense_pattern(vop, n)


class TestTracingDenseAndStructured:
    def test_dense_matmul(self):
        n = 12
        W = jax.random.normal(jax.random.PRNGKey(0), (n, n))
        op = lambda x: W @ x
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    def test_banded_matmul(self):
        n = 12
        Wb = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if abs(i - j) <= 1:
                    Wb[i, j] = 1.0 + (i + 1)
        Wj = jnp.asarray(Wb)
        op = lambda x: Wj @ x
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    def test_matmul_along_axis(self):
        # einsum-style: apply a small dense matrix along one axis.
        M = jax.random.normal(jax.random.PRNGKey(2), (4, 4))
        op = lambda x: (M @ x.reshape(4, 3)).reshape(-1)
        assert _trace_set(op, (12, 12)) == _dense_pattern(op, 12)

    def test_scan_carry_stencil(self):
        # out[t] = x[t] + x[t-1] via a scan carry -> bidiagonal Jacobian.
        def op(x):
            def body(carry, xt):
                return xt, xt + carry

            _, ys = jax.lax.scan(body, 0.0, x)
            return ys

        n = 10
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    def test_cumsum_falls_back(self):
        # cumsum has prefix dependence; must NOT be mistaken for elementwise.
        n = 12
        assert trace_sparsity_pattern(lambda x: jnp.cumsum(x), (n, n)) is None


class TestTracingFallback:
    def test_opaque_callback_returns_none(self):
        n = 27

        def op(u):
            return jax.pure_callback(
                lambda x: np.asarray(x) * 2.0,
                jax.ShapeDtypeStruct((n,), u.dtype),
                u,
            )

        assert trace_sparsity_pattern(op, (n, n)) is None

    def test_data_dependent_indexing_returns_none(self):
        n = 27
        op = lambda u: u[jnp.argsort(u)]
        assert trace_sparsity_pattern(op, (n, n)) is None


class TestCacheColoringTracing:
    def test_cache_coloring_takes_tracing_path(self):
        n = 6**3
        op = poisson3d_operator(robin=2.0)
        cache = _try_trace_coloring(op, n, n)
        assert cache is not None  # tracing succeeds for a traceable stencil

    def test_cache_coloring_materializes_correctly(self):
        n = 6**3
        op = poisson3d_operator(robin=2.0)
        rows, cols, column_colors, n_colors, shape = cache_coloring(op, (n, n))
        A = materialize_sparse_matrix(op, (n, n), rows, cols, column_colors, n_colors)
        Adense = np.asarray(A.todense())
        J = np.asarray(jax.jacfwd(op)(jnp.ones(n)))
        np.testing.assert_allclose(Adense, J, atol=1e-4)
