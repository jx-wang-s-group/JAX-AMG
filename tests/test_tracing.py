"""Tests for jaxpr-tracing sparsity detection (``trace_sparsity_pattern``).

Covers the per-primitive connectivity rules in ``_interp`` -- elementwise (linear
and nonlinear), every movement primitive (reshape/transpose/flip/slice/
broadcast/concatenate/pad/dynamic_slice/gather), reduce, scatter-add,
convolution, dot_general, scan, and call (jit) -- the conservative-superset
behaviour of cond/where, and the structural fallbacks (opaque calls,
data-dependent indexing, unhandled primitives).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from jaxamg.matrices import poisson3d_operator, poisson_operator, tridiagonal_operator
from jaxamg.sparsity import trace_sparsity_pattern


def _dense_pattern(op, n, tol=1e-5):
    """Ground-truth nonzero pattern from the dense Jacobian."""
    J = np.asarray(jax.jacfwd(op)(jnp.ones(n)))
    r, c = np.nonzero(np.abs(J) > tol)
    return set(zip(r.tolist(), c.tolist()))


def _trace_set(op, shape):
    res = trace_sparsity_pattern(op, shape)
    if res is None:
        return None
    return set(zip(res[0].tolist(), res[1].tolist()))


# Each operator maps a length-12 vector to a length-12 vector and exercises one
# primitive rule (named by the parametrize id); the tracer must recover its
# Jacobian pattern EXACTLY in a single pass.
_N = 12
_EXACT_OPS = {
    "elementwise_diagonal": lambda x: 2.0 * x - 0.5 * x,  # purely diagonal
    "elementwise_nonlinear": lambda x: jnp.sin(x) + jnp.roll(x, 1),
    "reshape_transpose": lambda x: (x.reshape(4, 3).T).reshape(-1),
    "flip_rev": lambda x: jnp.flip(x),
    "concatenate": lambda x: jnp.concatenate([x[1:], x[:1]]),
    "slice_and_pad": lambda x: jnp.pad(x[1:], (0, 1)),
    "dynamic_slice": lambda x: jnp.pad(lax.dynamic_slice(x, (2,), (5,)), (2, 5)),
    "gather_permutation": lambda x: x[
        jnp.array([3, 1, 0, 2, 5, 4, 7, 6, 9, 8, 11, 10])
    ],
    "gather_computed_index": lambda x: x[(jnp.arange(_N) + 3) % _N],  # iota-built idx
    "reduce_then_broadcast": lambda x: jnp.repeat(x.reshape(4, 3).sum(axis=1), 3),
    "reduce_all_axes": lambda x: x + jnp.sum(x),  # full reduction -> scalar -> dense
    "squeeze_expand_dims": lambda x: jnp.squeeze(jnp.expand_dims(x, 1), 1)
    + jnp.roll(x, 1),
    "periodic_wrap": lambda x: 2.0 * x - jnp.roll(x, 1) - jnp.roll(x, -1),
    "jit_wrapped": lambda x: jax.jit(lambda v: 2.0 * v - jnp.roll(v, 1))(x),
    "reverse_scan": lambda x: jax.lax.scan(
        lambda c, xt: (xt, xt + c), 0.0, x, reverse=True
    )[1],
}


class TestTracingExact:
    """Operators whose pattern the tracer recovers exactly in one pass."""

    @pytest.mark.parametrize("op", _EXACT_OPS.values(), ids=_EXACT_OPS.keys())
    def test_primitive_rule(self, op):
        assert _trace_set(op, (_N, _N)) == _dense_pattern(op, _N)

    def test_scatter_add_3d_stencil(self):
        # 3D 7-point stencil via .at[].add -> scatter-add rule, Robin BC diagonal.
        n = 6**3
        op = poisson3d_operator(robin=2.0)
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    @pytest.mark.parametrize("opf", [tridiagonal_operator(), poisson_operator()])
    def test_convolution(self, opf):
        # jnp.convolve -> conv_general_dilated rule.
        n = 16
        assert _trace_set(opf, (n, n)) == _dense_pattern(opf, n)

    def test_dense_matmul(self):
        # dot_general with a dense constant -> fully dense pattern.
        n = 12
        w = jax.random.normal(jax.random.PRNGKey(0), (n, n))
        op = lambda x: w @ x
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    def test_banded_matmul(self):
        n = 12
        wb = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if abs(i - j) <= 1:
                    wb[i, j] = 1.0 + (i + 1)
        wj = jnp.asarray(wb)
        op = lambda x: wj @ x
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    def test_matmul_along_axis(self):
        # einsum-style: apply a small dense matrix along one axis of a reshape.
        m = jax.random.normal(jax.random.PRNGKey(2), (4, 4))
        op = lambda x: (m @ x.reshape(4, 3)).reshape(-1)
        assert _trace_set(op, (12, 12)) == _dense_pattern(op, 12)

    def test_dot_general_data_on_lhs(self):
        # x @ W puts the data operand on the LHS of dot_general (the other
        # dot_general tests put it on the RHS).
        n = 12
        w = jax.random.normal(jax.random.PRNGKey(3), (n, n))
        op = lambda x: x @ w
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

    def test_scan_carry_stencil(self):
        # out[t] = x[t] + x[t-1] via a scan carry -> bidiagonal Jacobian.
        def op(x):
            def body(carry, xt):
                return xt, xt + carry

            _, ys = jax.lax.scan(body, 0.0, x)
            return ys

        n = 10
        assert _trace_set(op, (n, n)) == _dense_pattern(op, n)

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


class TestTracingConservativeSuperset:
    """Branching primitives: the tracer cannot know which branch is taken, so it
    returns a safe SUPERSET (union of branches). cache_coloring later refines it
    to the exact matrix via materialize + drop_zeros (see test_sparsity_cache)."""

    @pytest.mark.parametrize(
        "op",
        [
            lambda x: lax.cond(True, lambda v: jnp.roll(v, 1), lambda v: 2.0 * v, x),
            lambda x: jnp.where(jnp.arange(12) % 2 == 0, x, jnp.roll(x, 1)),
        ],
        ids=["cond", "where_select"],
    )
    def test_trace_is_superset(self, op):
        traced = _trace_set(op, (12, 12))
        truth = _dense_pattern(op, 12)
        assert traced is not None
        assert traced >= truth  # conservative: never misses a true dependency
        assert len(traced) > len(truth)  # and here strictly over-approximates

    def test_data_dependent_cond_predicate_captured(self):
        # The cond predicate depends on x[0], so the branch taken -- and hence
        # every output element -- depends on x[0]. The tracer must fold that in
        # (a safe superset), never under-approximate by ignoring the predicate.
        n = 12
        op = lambda x: lax.cond(
            x[0] > 0, lambda v: jnp.roll(v, 1), lambda v: jnp.roll(v, -1), x
        )
        res = trace_sparsity_pattern(op, (n, n))
        got = set(zip(res[0].tolist(), res[1].tolist()))
        assert all((i, 0) in got for i in range(n))  # predicate input in every row


class TestTracingFallback:
    """Operators the tracer cannot resolve structurally must return None so the
    caller falls back to exhaustive probing."""

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
        # Indices computed from the input (argsort) -> not structurally traceable.
        n = 27
        op = lambda u: u[jnp.argsort(u)]
        assert trace_sparsity_pattern(op, (n, n)) is None

    def test_unhandled_primitive_returns_none(self):
        # cumsum has prefix dependence; it must NOT be mistaken for elementwise,
        # and there is no movement rule for it -> bail to probing.
        n = 12
        assert trace_sparsity_pattern(lambda x: jnp.cumsum(x), (n, n)) is None

    def test_data_dependent_scatter_index_returns_none(self):
        # Scatter target positions computed from the input -> not traceable.
        n = 12
        op = lambda x: jnp.zeros(n).at[jnp.argsort(x)].add(x)
        assert trace_sparsity_pattern(op, (n, n)) is None

    def test_dot_general_both_operands_data_returns_none(self):
        # x @ x is a contraction of two input-dependent operands (nonlinear); the
        # dot_general rule needs exactly one constant operand, so it must bail.
        n = 12
        op = lambda x: jnp.ones(n) * (x @ x)
        assert trace_sparsity_pattern(op, (n, n)) is None
