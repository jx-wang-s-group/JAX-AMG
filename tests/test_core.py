"""The implicit derivative core (``jaxamg.core``) and the derivative policies of
``solve``: forward, reverse and higher-order derivatives against dense float64
references, and the reverse-only adjoint rule."""

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse

import jaxamg
from jaxamg.matrices import poisson_matrix, poisson_operator

pytestmark = pytest.mark.gpu

CONFIG = {
    "solver": "PBICGSTAB",
    "preconditioner": {"solver": "AMG", "max_iters": 1},
    "tolerance": 1e-14,
    "max_iters": 500,
}
RTOL = 1e-8


@pytest.fixture(autouse=True)
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def _csr(A):
    A = scipy.sparse.csr_matrix(A)
    A.sort_indices()
    return jsp.BCSR(
        (
            jnp.asarray(A.data, jnp.float64),
            jnp.asarray(A.indices, jnp.int32),
            jnp.asarray(A.indptr, jnp.int32),
        ),
        shape=A.shape,
    )


def _systems():
    spd = poisson_matrix(8)
    spd = _csr(
        scipy.sparse.csr_matrix(
            (np.asarray(spd.data), np.asarray(spd.indices), np.asarray(spd.indptr)),
            shape=spd.shape,
        )
    )
    nonsym = poisson_matrix(8, skew=3.0)
    nonsym = _csr(
        scipy.sparse.csr_matrix(
            (
                np.asarray(nonsym.data),
                np.asarray(nonsym.indices),
                np.asarray(nonsym.indptr),
            ),
            shape=nonsym.shape,
        )
    )
    node = scipy.sparse.diags(
        (-np.ones(15), 4.0 * np.ones(16), -1.25 * np.ones(15)), offsets=(-1, 0, 1)
    )
    block = _csr(scipy.sparse.kron(node, np.array([[2.0, 0.25], [0.5, 1.5]])))
    return {
        "symmetric": (spd, {}),
        # with_cache marks the object it is given, so the declared system is a
        # separate object from the undeclared one.
        "symmetric_declared": (
            jaxamg.with_cache(
                jsp.BCSR((spd.data, spd.indices, spd.indptr), shape=spd.shape),
                is_symmetric=True,
            ),
            {},
        ),
        "nonsymmetric": (nonsym, {}),
        "block": (block, {"block_dim": 2}),
    }


NAMES = ("symmetric", "symmetric_declared", "nonsymmetric", "block")


def _dense(A):
    return np.asarray(A.todense())


def _reference(values, A, b):
    M = (
        jnp.zeros(A.shape)
        .at[
            jnp.repeat(
                jnp.arange(A.shape[0]),
                jnp.diff(A.indptr),
                total_repeat_length=values.shape[0],
            ),
            A.indices,
        ]
        .add(values)
    )
    return jnp.linalg.solve(M, b)


@pytest.mark.parametrize("name", NAMES)
def test_all_derivative_orders_match_dense(name):
    A, options = _systems()[name]  # built under x64 (the fixture)
    n = A.shape[0]
    b = jnp.asarray(np.random.default_rng(1).standard_normal(n))
    w = jnp.asarray(np.random.default_rng(2).standard_normal(n))
    # A symmetric perturbation, d_k = g[row_k] + g[col_k], keeps a symmetric
    # matrix symmetric (and its declaration true).
    g = np.random.default_rng(3).standard_normal(n)
    rows = np.repeat(np.arange(n), np.diff(np.asarray(A.indptr)))
    direction = jnp.asarray(g[rows] + g[np.asarray(A.indices)])

    def amg(t):
        values = A.data * (1 + 0.1 * t * direction)
        x, _ = jaxamg.solve(_with(A, values), b * (1 + t), config=CONFIG, **options)
        return w @ x

    def dense(t):
        values = A.data * (1 + 0.1 * t * direction)
        return w @ _reference(values, A, b * (1 + t))

    t0 = 0.3
    for label, transform in [
        ("reverse", jax.grad),
        ("forward", jax.jacfwd),
        ("RR", lambda f: jax.grad(jax.grad(f))),
        ("FR", lambda f: jax.jacfwd(jax.grad(f))),
        ("RF", lambda f: jax.grad(jax.jacfwd(f))),
        ("FF", lambda f: jax.jacfwd(jax.jacfwd(f))),
    ]:
        np.testing.assert_allclose(
            transform(amg)(t0), transform(dense)(t0), rtol=RTOL, err_msg=label
        )
    np.testing.assert_allclose(
        jax.jit(jax.grad(amg))(t0), jax.grad(dense)(t0), rtol=RTOL
    )


def _with(A, values):
    matrix = jsp.BCSR((values, A.indices, A.indptr), shape=A.shape)
    if getattr(A, "_is_symmetric", False):
        matrix = jaxamg.with_cache(matrix, is_symmetric=True)
    return matrix


def test_matrix_value_cotangent_and_x0():
    A, _ = _systems()["nonsymmetric"]
    n = A.shape[0]
    b = jnp.asarray(np.random.default_rng(4).standard_normal(n))
    x0 = jnp.asarray(np.random.default_rng(5).standard_normal(n))

    def loss(values, rhs, start):
        x, _ = jaxamg.solve(_with(A, values), rhs, x0=start, config=CONFIG)
        return 0.5 * jnp.sum(x**2)

    g_values, g_b, g_x0 = jax.grad(loss, argnums=(0, 1, 2))(A.data, b, x0)
    D = _dense(A)
    x = np.linalg.solve(D, np.asarray(b))
    lam = np.linalg.solve(D.T, x)
    rows = np.repeat(np.arange(n), np.diff(np.asarray(A.indptr)))
    np.testing.assert_allclose(
        g_values, -lam[rows] * x[np.asarray(A.indices)], rtol=RTOL
    )
    np.testing.assert_allclose(g_b, lam, rtol=RTOL)
    np.testing.assert_array_equal(g_x0, np.zeros(n))
    # A warm start moves the iteration, not the tangent.
    _, tangent = jax.jvp(lambda s: loss(A.data, b, s), (x0,), (jnp.ones(n),))
    assert float(tangent) == 0.0


def test_adjoint_rule_matches_first_reverse_and_refuses_forward():
    A, _ = _systems()["nonsymmetric"]
    b = jnp.asarray(np.random.default_rng(6).standard_normal(A.shape[0]))

    def loss(values, rhs, derivative):
        x, _ = jaxamg.solve(_with(A, values), rhs, config=CONFIG, derivative=derivative)
        return 0.5 * jnp.sum(x**2)

    implicit = jax.grad(loss, argnums=(0, 1))(A.data, b, "implicit")
    adjoint = jax.grad(loss, argnums=(0, 1))(A.data, b, "adjoint")
    for g_i, g_a in zip(implicit, adjoint):
        np.testing.assert_array_equal(g_i, g_a)  # the same native calls and products
    with pytest.raises(TypeError, match="forward-mode"):
        jax.jvp(lambda r: loss(A.data, r, "adjoint"), (b,), (b,))
    with pytest.raises(ValueError, match="derivative must be one of"):
        jaxamg.solve(A, b, derivative="exact")


def test_singular_projected_derivatives():
    n = 32
    main = 2.0 * np.ones(n)
    main[0] = main[-1] = 1.0
    N = _csr(
        scipy.sparse.diags((-np.ones(n - 1), main, -np.ones(n - 1)), offsets=(-1, 0, 1))
    )
    b = jnp.linspace(1.0, 2.0, n)
    P = np.eye(n) - np.ones((n, n)) / n

    def amg(rhs):
        x, _ = jaxamg.solve(
            N, rhs, nullspace="constant", transpose_nullspace="constant", config=CONFIG
        )
        return x

    pinv = np.linalg.pinv(_dense(N))
    np.testing.assert_allclose(amg(b), pinv @ (P @ np.asarray(b)), atol=1e-9)
    J_forward = jax.jacfwd(amg)(b)
    J_reverse = jax.jacrev(amg)(b)
    np.testing.assert_allclose(J_forward, pinv @ P, atol=1e-9)
    np.testing.assert_allclose(J_reverse, pinv @ P, atol=1e-9)


def test_operator_parameters_all_orders():
    n = 64
    coloring = jaxamg.cache_coloring(poisson_operator(0.5), shape=(n, n))
    b = jnp.linspace(-1.0, 1.0, n)

    def amg(s):
        x, _ = jaxamg.solve(
            jaxamg.with_cache(poisson_operator(s), coloring=coloring), b, config=CONFIG
        )
        return jnp.sum(x**2)

    def dense(s):
        M = jax.vmap(poisson_operator(s), in_axes=1, out_axes=1)(jnp.eye(n))
        return jnp.sum(jnp.linalg.solve(M, b) ** 2)

    for transform in (
        jax.grad,
        jax.jacfwd,
        lambda f: jax.jacfwd(jax.grad(f)),
        lambda f: jax.grad(jax.grad(f)),
    ):
        np.testing.assert_allclose(
            transform(amg)(0.5), transform(dense)(0.5), rtol=RTOL
        )


def test_vmap_over_rhs_and_values():
    A, _ = _systems()["symmetric"]
    n = A.shape[0]
    rhs = jnp.asarray(np.random.default_rng(7).standard_normal((3, n)))
    scales = jnp.array([1.0, 1.5, 2.0])

    def solve(scale, r):
        return jaxamg.solve(_with(A, A.data * scale), r, config=CONFIG)[0]

    batched = jax.vmap(solve)(scales, rhs)
    for k in range(3):
        np.testing.assert_allclose(
            batched[k], np.linalg.solve(_dense(A) * scales[k], rhs[k]), rtol=RTOL
        )
    tangents = jax.vmap(lambda s, r: jax.jvp(lambda t: solve(t, r), (s,), (1.0,))[1])(
        scales, rhs
    )
    for k in range(3):
        x = np.linalg.solve(_dense(A) * scales[k], rhs[k])
        np.testing.assert_allclose(tangents[k], -x / scales[k], rtol=RTOL)
