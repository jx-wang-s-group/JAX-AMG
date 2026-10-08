"""Pattern reuse across traces, changing values, precisions and devices."""

import pickle
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxamg
from jaxamg.sparsity import materialize_sparse_matrix
from jaxamg.utils import temp_enable_x64, to_bcsr_matrix


def declaration(n=8):
    i = np.arange(n)
    return jaxamg.pattern(
        np.concatenate((i, i[:-1], i[1:])),
        np.concatenate((i, i[1:], i[:-1])),
        (n, n),
    )


def operator(c):
    def apply(x):
        y = (3 + c) * x
        y = y.at[1:].add(-c * x[:-1])
        return y.at[:-1].add(-0.5 * c * x[1:])

    return apply


def matrix(p, c, *, declared=True, mpi=False):
    kw = {"pattern": p} if declared else {"coloring": p._coloring()}
    op = jaxamg.with_cache(operator(c), **kw)
    return to_bcsr_matrix(
        op, jnp.ones(p.shape[0], dtype=c.dtype), use_int64_indices=mpi
    )


def test_first_use_inside_jit_does_not_retain_tracers():
    p = declaration()
    assert p._layout is None
    with jax.checking_leaks():
        result = jax.jit(lambda c: matrix(p, c).todense())(jnp.float32(0.25))
    layout = p._layout
    assert layout is not None
    assert all(
        isinstance(a, jax.Array) and not isinstance(a, jax.core.Tracer) for a in layout
    )
    assert all(a.dtype == jnp.int32 and not a.committed for a in layout)
    eager = matrix(p, jnp.float32(0.25))
    assert eager.indices is layout.cols and eager.indptr is layout.indptr
    np.testing.assert_array_equal(result, eager.todense())
    # Another trace and different coefficients must capture the same structure.
    jax.jit(lambda c: matrix(p, c).data.sum())(jnp.float32(0.5))
    assert p._layout is layout


def test_values_and_gradients_change_while_layout_is_shared():
    p = declaration()

    def loss(c, declared):
        return jnp.sum(matrix(p, c, declared=declared).data ** 2)

    c = jnp.array([0.0, 0.125, 0.5])
    shared = jax.jit(jax.vmap(jax.value_and_grad(lambda c: loss(c, True))))(c)
    reference = jax.jit(jax.vmap(jax.value_and_grad(lambda c: loss(c, False))))(c)
    for actual, expected in zip(shared, reference):
        np.testing.assert_array_equal(actual, expected)
    assert len(np.unique(np.asarray(shared[0]))) == c.size


def test_one_layout_is_captured_regardless_of_site_count():
    p = declaration()

    def sites(c, count):
        return sum(matrix(p, c + i * 0.1).todense().sum() for i in range(count))

    for count in (1, 10):
        traced = jax.make_jaxpr(lambda c: sites(c, count))(jnp.float32(0.2))
        # Check the sharing contract before lowering, independent of XLA's
        # version-specific textual representation or constant folding.
        constants = [a for a in traced.consts if a.shape == (p.rows.size,)]
        assert len(constants) == 3
        assert {id(a) for a in constants} == {
            id(p._layout.rows),
            id(p._layout.cols),
            id(p._layout.entry_colors),
        }


def test_layout_is_independent_of_precision():
    p = declaration()
    layout = None
    with temp_enable_x64():
        for dtype in (jnp.float32, jnp.float64):
            c = dtype(0.123456789123)
            actual = jax.jit(lambda c: matrix(p, c).data)(c)
            expected = matrix(p, c, declared=False).data
            assert actual.dtype == dtype
            np.testing.assert_array_equal(actual, expected)
            layout = p._layout if dtype == jnp.float32 else layout
            assert p._layout is layout


def test_nested_solves_capture_only_forward_and_transpose_layouts():
    forward = declaration()
    transpose = jaxamg.pattern(forward.cols, forward.rows, forward.shape)

    def recover_and_solve(op, b, p):
        dense = to_bcsr_matrix(jaxamg.with_cache(op, pattern=p), b).todense()
        return jnp.linalg.solve(dense, b)

    def program(c, count):
        def step(_, x):
            for site in range(count):

                def body(carry):
                    k, b = carry
                    x = jax.lax.custom_linear_solve(
                        operator(c + 0.1 * site + 0.01 * k),
                        b,
                        solve=lambda op, b: recover_and_solve(op, b, forward),
                        transpose_solve=lambda op, b: recover_and_solve(
                            op, b, transpose
                        ),
                    )
                    return k + 1, x

                _, x = jax.lax.while_loop(lambda carry: carry[0] < 2, body, (0, x))
            return x

        return jax.lax.fori_loop(0, 3, step, jnp.ones(8))

    for count in (1, 10):
        traced = jax.make_jaxpr(lambda c: program(c, count))(jnp.float32(0.2))
        constants = [a for a in traced.consts if a.shape == (forward.rows.size,)]
        assert len(constants) == 6
        assert {id(a) for a in constants} == {
            id(a)
            for p in (forward, transpose)
            for a in (p._layout.rows, p._layout.cols, p._layout.entry_colors)
        }


def test_mpi_indices_bypass_single_device_layout():
    p = declaration()
    c = jnp.float32(0.25)
    with temp_enable_x64():
        actual = matrix(p, c, mpi=True)
        assert p._layout is None
        assert actual.indices.dtype == jnp.int64
        expected = matrix(p, c)
        layout = p._layout
        reused = jax.jit(lambda c: matrix(p, c, mpi=True))(c)
        assert p._layout is layout
        np.testing.assert_array_equal(actual.todense(), expected.todense())
        np.testing.assert_array_equal(reused.todense(), expected.todense())


def test_replacement_uses_the_new_structure_after_layout_was_built():
    p = declaration()
    op = jaxamg.with_cache(operator(jnp.float32(0.5)), pattern=p)
    b = jnp.ones(8)
    to_bcsr_matrix(op, b)
    diagonal = jaxamg.pattern(np.arange(8), np.arange(8), (8, 8))
    jaxamg.with_cache(op, coloring=diagonal._coloring())
    assert op._pattern is None
    assert to_bcsr_matrix(op, b).data.size == 8
    jaxamg.with_cache(op, pattern=diagonal)
    assert to_bcsr_matrix(op, b).indices is diagonal._layout.cols


def test_rectangular_and_empty_patterns():
    for p in (jaxamg.pattern([1, 0], [3, 1], (2, 4)), jaxamg.pattern([], [], (2, 4))):
        op = lambda x: jnp.array([x[1], 2 * x[3]])
        if p.rows.size == 0:
            op = lambda x: jnp.zeros(2, dtype=x.dtype)
        A = jaxamg.with_cache(op, pattern=p)
        actual = jax.jit(lambda b: to_bcsr_matrix(A, b).todense())(jnp.ones(2))
        expected = jax.jacfwd(op)(jnp.ones(4))
        np.testing.assert_array_equal(actual, expected)


def test_traced_indices_still_materialize_without_a_layout():
    p = declaration()

    @jax.jit
    def recover(rows, cols, colors):
        return materialize_sparse_matrix(
            operator(jnp.float32(0.5)), p.shape, rows, cols, colors, p.n_colors
        ).todense()

    np.testing.assert_array_equal(
        recover(p.rows, p.cols, p.colors), matrix(p, jnp.float32(0.5)).todense()
    )


def test_concurrent_first_uses_share_one_layout():
    p = declaration()
    barrier = Barrier(4)

    def build(_):
        barrier.wait(timeout=30)
        return p._materialization_layout()

    with ThreadPoolExecutor(max_workers=4) as pool:
        layouts = list(pool.map(build, range(4)))
    assert all(layout is layouts[0] for layout in layouts)


def test_pickling_a_materialized_pattern_drops_device_state():
    p = declaration()
    expected = matrix(p, jnp.float32(0.5)).todense()
    restored = pickle.loads(pickle.dumps(p))
    assert restored._layout is None
    np.testing.assert_array_equal(
        matrix(restored, jnp.float32(0.5)).todense(), expected
    )


def test_reuse_on_another_device():
    devices = jax.local_devices()
    if len(devices) < 2:
        pytest.skip("requires two visible devices (virtual CPU devices suffice)")
    p = declaration()
    with jax.default_device(devices[0]):
        expected = matrix(p, jnp.float32(0.5)).todense()
    layout = p._layout
    with jax.default_device(devices[1]):
        c = jax.device_put(np.float32(0.5), devices[1])
        actual = jax.jit(lambda c: matrix(p, c).todense())(c)
    assert actual.devices() == {devices[1]}
    assert p._layout is layout
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.gpu
def test_solver_values_and_reverse_derivatives_match_dense():
    p = declaration()
    b = jnp.arange(1, 9, dtype=jnp.float32)

    def loss(c, b):
        op = jaxamg.with_cache(operator(c), pattern=p)
        x, _ = jaxamg.solve(op, b, solver="BICGSTAB", tolerance=1e-7)
        return jnp.sum(x**2)

    def reference(c, b):
        dense = jax.jacfwd(operator(c))(jnp.ones(8))
        return jnp.sum(jnp.linalg.solve(dense, b) ** 2)

    actual = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))
    expected = jax.jit(jax.value_and_grad(reference, argnums=(0, 1)))
    for c in (jnp.float32(0.0), jnp.float32(0.25)):
        for result, target in zip(
            jax.tree.leaves(actual(c, b)), jax.tree.leaves(expected(c, b))
        ):
            np.testing.assert_allclose(result, target, rtol=2e-5, atol=2e-6)
