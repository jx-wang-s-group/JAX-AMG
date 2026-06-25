"""Tests for the detect -> colour -> materialise orchestration in sparsity.py:
``cache_coloring`` (tracing path, probing fallback, caching, shape errors),
``materialize_sparse_matrix`` (incl. drop-zeros of structural-but-numerical
zeros), and ``_verify_recovery``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxamg.matrices import poisson3d_operator
from jaxamg.sparsity import (
    _verify_recovery,
    cache_coloring,
    get_column_coloring,
    materialize_sparse_matrix,
    probe_sparsity_pattern,
    trace_sparsity_pattern,
)


def _dense(op, n):
    return np.asarray(jax.jacfwd(op)(jnp.ones(n)))


def _dense_by_columns(op, n):
    """Ground-truth matrix for a linear op that can't be differentiated (e.g. a
    pure_callback): column j is op(e_j)."""
    return np.stack([np.asarray(op(jnp.eye(n)[j])) for j in range(n)], axis=1)


def _materialize_dense(op, cache):
    rows, cols, colors, n_colors, shape = cache
    A = materialize_sparse_matrix(op, shape, rows, cols, colors, n_colors)
    return np.asarray(A.todense())


def _pat_set(rc):
    """A (rows, cols) pattern as a set of (row, col) pairs, for order-free compares."""
    r, c = rc
    return set(zip(r.tolist(), c.tolist()))


class TestCacheColoringTracingPath:
    def test_takes_tracing_path(self):
        # A traceable stencil should be detected by tracing (not probing).
        n = 6**3
        op = poisson3d_operator(robin=2.0)
        assert trace_sparsity_pattern(op, (n, n)) is not None
        cache = cache_coloring(op, (n, n))
        assert cache[4] == (n, n)

    def test_materializes_correctly(self):
        n = 6**3
        op = poisson3d_operator(robin=2.0)
        cache = cache_coloring(op, (n, n))
        np.testing.assert_allclose(
            _materialize_dense(op, cache), _dense(op, n), atol=1e-4
        )


class TestCacheColoringProbingFallback:
    def test_opaque_operator_falls_back_and_is_correct(self):
        # An opaque-to-tracing but vmap-able linear operator: tracing returns None,
        # so cache_coloring must fall back to one-hot probing and still be exact.
        n = 16

        def opaque(u):
            return jax.pure_callback(
                lambda v: np.asarray(v) * 2.0 - np.roll(np.asarray(v), 1),
                jax.ShapeDtypeStruct((n,), u.dtype),
                u,
                vmap_method="sequential",
            )

        assert trace_sparsity_pattern(opaque, (n, n)) is None  # tracing bails
        cache = cache_coloring(opaque, (n, n))  # -> probing fallback
        np.testing.assert_allclose(
            _materialize_dense(opaque, cache), _dense_by_columns(opaque, n), atol=1e-4
        )

    def test_non_vmappable_operator_uses_sequential_fallback(self):
        # pure_callback with the default vmap_method has NO vmap rule, so probing
        # must fall back to a sequential lax.map (this used to raise outright).
        n = 16

        def opaque(u):
            return jax.pure_callback(
                lambda v: np.asarray(v) * 2.0 - np.roll(np.asarray(v), 1),
                jax.ShapeDtypeStruct((n,), u.dtype),
                u,  # vmap_method=None -> not vmap-able
            )

        cache = cache_coloring(opaque, (n, n))
        np.testing.assert_allclose(
            _materialize_dense(opaque, cache), _dense_by_columns(opaque, n), atol=1e-4
        )


class TestZeroOperator:
    def test_zero_operator_materializes_to_zero(self):
        # An operator with no input dependence traces to an empty pattern
        # (rows.size == 0), so cache_coloring falls through to probing; the
        # materialized matrix must be all zeros.
        n = 12
        op = lambda x: jnp.zeros_like(x)
        cache = cache_coloring(op, (n, n))
        assert len(cache[0]) == 0  # empty pattern
        assert np.allclose(_materialize_dense(op, cache), 0.0)


class TestZeroSkipMul:
    """``mul`` drops rows forced to zero by a structurally-zero constant operand,
    tightening the pattern without changing the matrix."""

    def test_zero_skip_removes_overreporting(self):
        # Mask the dense rows of a lower-triangular op: the old `mul` rule kept the
        # full pattern of L@x (loose superset of the truth); the zero-skip makes it
        # exact -- same matrix from both, the loose one just costs more colours.
        n = 48
        L = jnp.asarray(np.tril(np.ones((n, n), np.float32)))
        keep = jnp.asarray((np.arange(n) < 4).astype(np.float32))
        op = lambda x: keep * (L @ x)

        rt, ct = trace_sparsity_pattern(op, (n, n))  # new, zero-skipped
        rl, cl = trace_sparsity_pattern(lambda x: L @ x, (n, n))  # old, over-reported
        tight = _pat_set((rt, ct))
        loose = _pat_set((rl, cl))
        true_pat = _pat_set(probe_sparsity_pattern(op, (n, n)))  # exact (linear op)

        assert tight == true_pat  # exact now
        assert true_pat < loose  # old pattern over-reported

        truth = _dense(op, n)
        colors_t, nct = get_column_coloring(rt, ct, (n, n))
        colors_l, ncl = get_column_coloring(rl, cl, (n, n))
        np.testing.assert_allclose(
            _materialize_dense(op, (rt, ct, colors_t, nct, (n, n))), truth, atol=1e-5
        )
        np.testing.assert_allclose(
            _materialize_dense(op, (rl, cl, colors_l, ncl, (n, n))), truth, atol=1e-5
        )
        assert nct < ncl  # 4 vs 48


class TestScatterReplace:
    """Plain scatter (``base.at[i:j].set(block)``) drops the operand dependence at
    written positions rather than over-reporting it as combine semantics would."""

    def test_replace_drops_operand_at_written_rows(self):
        # base is diagonal; a window is overwritten by a block of other inputs, so
        # those rows lose their operand (diagonal) dependence. Linear, so probing
        # is a valid oracle.
        n = 16
        op = lambda x: (3.0 * x).at[10:14].set(x[0:4] + x[4:8])

        pat = trace_sparsity_pattern(op, (n, n))
        assert pat is not None
        assert _pat_set(pat) == _pat_set(probe_sparsity_pattern(op, (n, n)))
        assert (10, 10) not in _pat_set(pat)  # operand dependence dropped at write

        cache = cache_coloring(op, (n, n))
        np.testing.assert_allclose(
            _materialize_dense(op, cache), _dense(op, n), atol=1e-5
        )


class TestDynamicUpdateSlice:
    """``lax.dynamic_update_slice`` traces to an exact pattern instead of bailing
    to probing; the update window replaces that slice of the operand."""

    def test_static_window_is_exact(self):
        n = 16

        def op(x):
            return jax.lax.dynamic_update_slice(3.0 * x, x[0:4] + x[4:8], (10,))

        pat = trace_sparsity_pattern(op, (n, n))
        assert pat is not None  # handled now, no longer bails to probing
        assert _pat_set(pat) == _pat_set(probe_sparsity_pattern(op, (n, n)))

        cache = cache_coloring(op, (n, n))
        np.testing.assert_allclose(
            _materialize_dense(op, cache), _dense(op, n), atol=1e-5
        )


class TestOverlappingScatterFallback:
    def test_falls_back_to_probing_and_is_correct(self):
        # Overlapping scatter-add (segment-sum): tracing detects the overlap and
        # bails, so cache_coloring must fall back to probing and stay exact.
        n = 12
        idx = jnp.repeat(jnp.arange(n // 2), 2)  # [0, 0, 1, 1, ...]
        op = lambda x: jnp.zeros(n).at[idx].add(x)
        assert trace_sparsity_pattern(op, (n, n)) is None
        cache = cache_coloring(op, (n, n))
        np.testing.assert_allclose(
            _materialize_dense(op, cache), _dense(op, n), atol=1e-4
        )


class TestCacheColoringOrchestration:
    def test_caches_and_reuses(self):
        op = lambda x: 2.0 * x - jnp.roll(x, 1)
        first = cache_coloring(op, (16, 16))
        second = cache_coloring(op, (16, 16))
        assert first is second  # reused from the attached _coloring_info
        assert getattr(op, "_coloring_info", None) is first

    def test_shape_mismatch_raises(self):
        op = lambda x: 2.0 * x - jnp.roll(x, 1)
        cache_coloring(op, (16, 16))
        with pytest.raises(ValueError):
            cache_coloring(op, (17, 17))

    def test_int_shape_expands_to_square(self):
        cache = cache_coloring(lambda x: 3.0 * x, 16)
        assert cache[4] == (16, 16)


class TestDropZeros:
    def test_structural_but_numerical_zeros_removed(self):
        # A position-dependent coefficient that is exactly zero on half the rows:
        # those rows are structurally present in the trace but numerically zero,
        # so drop_zeros must prune them and the cached pattern must match the
        # dense Jacobian's nonzeros exactly.
        n = 16
        c = jnp.array([1.0, 0.0] * (n // 2))
        op = lambda x: c * (2.0 * x - jnp.roll(x, 1) - jnp.roll(x, -1))
        rows, cols, colors, n_colors, shape = cache_coloring(op, (n, n))
        J = _dense(op, n)
        assert len(rows) == int((np.abs(J) > 1e-9).sum())
        np.testing.assert_allclose(
            _materialize_dense(op, (rows, cols, colors, n_colors, shape)), J, atol=1e-4
        )


class TestVerifyRecovery:
    def test_accepts_correct_rejects_perturbed(self):
        import jax.experimental.sparse as jsp

        n = 16
        op = lambda x: 2.0 * x - jnp.roll(x, 1)
        rows, cols, colors, n_colors, _ = cache_coloring(op, (n, n))
        good = materialize_sparse_matrix(op, (n, n), rows, cols, colors, n_colors)
        assert _verify_recovery(op, good, n) is True
        # Scaling the values breaks reconstruction -> must be rejected.
        bad = jsp.BCSR(
            (np.asarray(good.data) * 1.5, good.indices, good.indptr), shape=(n, n)
        )
        assert _verify_recovery(op, bad, n) is False


class TestMaterializeAutodiff:
    def test_gradient_flows_through_materialize(self):
        # Gradient of a scalar of A(theta) w.r.t. theta, materialized via coloring.
        n = 8
        base = poisson3d_operator(robin=1.0)
        op1 = lambda x: base(x) + 1.0 * x
        rows, cols, colors, n_colors, shape = cache_coloring(op1, (n, n))

        def loss(theta):
            op = lambda x: base(x) + theta * x
            A = materialize_sparse_matrix(op, shape, rows, cols, colors, n_colors)
            return jnp.sum(A.data**2)

        g = jax.grad(loss)(2.0)
        assert np.isfinite(float(g)) and float(g) != 0.0
