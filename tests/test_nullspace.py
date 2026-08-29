"""Null-space handling of singular systems (`nullspace` / `transpose_nullspace`)."""

import json
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import jaxamg
from jaxamg import NullSpaceWarning
from jaxamg import config as amgx_config
from jaxamg.matrices import poisson_matrix_stretched, tridiagonal_matrix
from jaxamg.nullspace import (
    as_nullspace_basis,
    project_out,
    relative_norm,
    validate_basis,
)
from jaxamg.utils import to_scipy

# PBICGSTAB + classical AMG (the defaults), tightened for float64 gradient checks.
cfg = {"tolerance": 1e-10, "max_iters": 300}


@pytest.fixture
def x64():
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


def dense(A) -> np.ndarray:
    return to_scipy(A).toarray().astype(np.float64)


def stretched(nx=32, ny=24, stretch=1.08, **kwargs):
    A, V = poisson_matrix_stretched(nx, ny, stretch, dtype=jnp.float64, **kwargs)
    return A, np.asarray(V)


def consistent_rhs(V: np.ndarray, seed: int = 0) -> np.ndarray:
    """Random b with Σ V_i b_i = 0."""
    rng = np.random.default_rng(seed)
    b = rng.standard_normal(V.shape[0])
    return b - (V @ b) / V.sum()


def weights(n: int, seed: int = 1) -> np.ndarray:
    """Objective weights with Σ w ≠ 0 (cotangent not in range(Aᵀ))."""
    return np.random.default_rng(seed).standard_normal(n) + 0.5


def remove_component(v: np.ndarray, d: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    return v - d * (d @ v)


class TestPure:
    """No solver needed (runs on CPU)."""

    def test_basis_normalization_and_rejection(self):
        n = 5
        assert as_nullspace_basis(None, n, jnp.float32, "nullspace") is None
        c = as_nullspace_basis("constant", n, jnp.float32, "nullspace")
        assert c.shape == (n, 1) and float(c.sum()) == n
        v = as_nullspace_basis(np.arange(n, dtype=np.float64), n, jnp.float32, "x")
        assert v.shape == (n, 1) and v.dtype == jnp.float32
        for bad in ["bogus", np.ones(3), np.ones((n, n + 1)), np.ones((n, 2, 1))]:
            with pytest.raises(ValueError, match="nullspace"):
                as_nullspace_basis(bad, n, jnp.float32, "nullspace")

    def test_validate_basis(self):
        n = 6
        ok = jnp.asarray(np.random.default_rng(0).standard_normal((n, 2)))
        validate_basis(ok, "nullspace")
        dup = jnp.stack([ok[:, 0], ok[:, 0]], axis=1)
        with pytest.raises(ValueError, match="dependent"):
            validate_basis(dup, "nullspace")
        with pytest.raises(ValueError, match="dependent"):
            validate_basis(jnp.zeros((n, 1)), "nullspace")
        with pytest.raises(ValueError, match="finite"):
            validate_basis(ok.at[0, 0].set(jnp.nan), "nullspace")

    @pytest.mark.parametrize("k", [1, 3])
    def test_project_out(self, k, x64):
        rng = np.random.default_rng(0)
        B = rng.standard_normal((20, k))
        v = rng.standard_normal(20)
        got = np.asarray(project_out(jnp.asarray(v), jnp.asarray(B)))
        ref = v - B @ np.linalg.solve(B.T @ B, B.T @ v)
        np.testing.assert_allclose(got, ref, rtol=1e-5, atol=1e-6)
        assert np.abs(B.T @ got).max() < 1e-5

    def test_relative_norm(self):
        d, v = jnp.array([3.0, 4.0]), jnp.array([0.0, 10.0])
        assert float(relative_norm(d, v)) == pytest.approx(0.5)
        assert float(relative_norm(d, jnp.zeros(2))) == 0.0

    def test_singular_config_defaults(self):
        default = amgx_config.prepare_config({})
        singular = amgx_config.prepare_config({}, singular=True)
        assert amgx_config.uses_dense_lu_coarse_solver(default)
        assert not amgx_config.uses_dense_lu_coarse_solver(singular)
        pc = json.loads(singular)["solver"]["preconditioner"]
        assert (
            pc["min_coarse_rows"] == 8
            and pc["coarse_solver"]["solver"] == "BLOCK_JACOBI"
        )
        # explicit settings win
        custom = amgx_config.prepare_config(
            {"preconditioner": {"coarse_solver": "DENSE_LU_SOLVER"}}, singular=True
        )
        assert amgx_config.uses_dense_lu_coarse_solver(custom)

    def test_stretched_matrix_null_vectors(self, x64):
        A, V = stretched(8, 6, 1.1)
        A_d = dense(A)
        scale = np.abs(A_d).max()
        assert np.abs(A_d @ np.ones(A_d.shape[0])).max() < 1e-12 * scale
        assert np.abs(A_d.T @ V).max() < 1e-12 * scale
        assert np.abs(A_d.T @ np.ones(A_d.shape[0])).max() > 1e-3 * scale
        L, _ = stretched(8, 6, 1.1, normalize=False)
        L_d = dense(L)
        assert np.abs(L_d - L_d.T).max() < 1e-12 * scale
        assert np.abs(L_d @ np.ones(L_d.shape[0])).max() < 1e-12 * scale


class TestSolve:
    """Runs the native AmgX solver (skip logic in conftest.py)."""

    pytestmark = pytest.mark.gpu

    def test_forward_pins_solution_and_projects_rhs(self, x64):
        A, V = stretched()
        b = jnp.asarray(consistent_rhs(V))
        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            x, info = jaxamg.solve(
                A, b, nullspace="constant", transpose_nullspace=V, **cfg
            )
        assert info["status"] == jaxamg.AMGXStatus.SUCCESS
        x = np.asarray(x)
        assert abs(x.mean()) < 1e-10 * np.linalg.norm(x)
        assert np.linalg.norm(dense(A) @ x - np.asarray(b)) < 1e-7 * np.linalg.norm(b)
        assert info["rhs_inconsistency"] < 1e-12

        # Inconsistent RHS: solved in the least-squares sense (b projected onto
        # range(A) = V⊥), removed fraction reported.
        b_bad = np.asarray(b) + 0.5
        x2, info2 = jaxamg.solve(
            A, jnp.asarray(b_bad), nullspace="constant", transpose_nullspace=V, **cfg
        )
        assert info2["status"] == jaxamg.AMGXStatus.SUCCESS
        b_proj = remove_component(b_bad, V)
        np.testing.assert_allclose(
            info2["rhs_inconsistency"],
            np.linalg.norm(b_bad - b_proj) / np.linalg.norm(b_bad),
            rtol=1e-6,
        )
        assert info2["rhs_inconsistency"] > 0.1
        assert np.linalg.norm(
            dense(A) @ np.asarray(x2) - b_proj
        ) < 1e-7 * np.linalg.norm(b_proj)

    def test_gradient_matches_pseudoinverse(self, x64):
        """jax.grad returns (Aᵀ)⁺g on the stretched grid."""
        A, V = stretched()
        b = jnp.asarray(consistent_rhs(V))
        w = weights(A.shape[0])

        def loss(b_):
            x, _ = jaxamg.solve(
                A, b_, nullspace="constant", transpose_nullspace=V, **cfg
            )
            return jnp.dot(jnp.asarray(w), x)

        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            g = jax.grad(loss)(b)
        g_ref = np.linalg.pinv(dense(A)).T @ w
        np.testing.assert_allclose(
            np.asarray(g), g_ref, rtol=1e-6, atol=1e-8 * np.linalg.norm(g_ref)
        )

    def test_gradient_wrt_matrix(self, x64):
        """d/dt of w·x with A(t) = (1+t)·A (null spaces preserved):
        x = A⁺b/(1+t), so the derivative is -w·A⁺b/(1+t)²."""
        A, V = stretched()
        b = jnp.asarray(consistent_rhs(V))
        w = jnp.asarray(weights(A.shape[0]))

        def loss(t):
            A_t = jax.experimental.sparse.BCSR(
                (A.data * (1.0 + t), A.indices, A.indptr), shape=A.shape
            )
            x, _ = jaxamg.solve(
                A_t, b, nullspace="constant", transpose_nullspace=V, **cfg
            )
            return jnp.dot(w, x)

        t0 = 0.2
        g = float(jax.grad(loss)(t0))
        ref = -float(w @ (np.linalg.pinv(dense(A)) @ np.asarray(b))) / (1 + t0) ** 2
        assert g == pytest.approx(ref, rel=1e-6)

    def test_nullspace_only_on_nonsymmetric_matrix(self, x64):
        """Only `nullspace`: warns, the adjoint still converges, gradient is
        right up to a null(Aᵀ) component. Only `transpose_nullspace`: warns."""
        A, V = stretched()
        b = jnp.asarray(consistent_rhs(V))
        w = weights(A.shape[0])

        def loss(b_):
            x, _ = jaxamg.solve(A, b_, nullspace="constant", **cfg)
            return jnp.dot(jnp.asarray(w), x)

        with pytest.warns(NullSpaceWarning, match="without transpose_nullspace"):
            g = jax.grad(loss)(b)
        g_ref = np.linalg.pinv(dense(A)).T @ w
        np.testing.assert_allclose(
            remove_component(np.asarray(g), V),
            remove_component(g_ref, V),
            rtol=1e-6,
            atol=1e-8 * np.linalg.norm(g_ref),
        )
        with pytest.warns(NullSpaceWarning, match="without nullspace"):
            jaxamg.solve(A, b, transpose_nullspace=V, max_iters=5)

    def test_symmetric_matrix_defaults_transpose_nullspace(self, x64):
        """Symmetric flux form marked symmetric: `transpose_nullspace` defaults
        to `nullspace`, no warnings, exact gradient."""
        L, _ = stretched(normalize=False)
        L = jaxamg.with_cache(L, is_symmetric=True)
        n = L.shape[0]
        b = np.random.default_rng(0).standard_normal(n)
        b = jnp.asarray(b - b.mean())
        w = weights(n)

        def loss(b_):
            x, _ = jaxamg.solve(L, b_, nullspace="constant", **cfg)
            return jnp.dot(jnp.asarray(w), x)

        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            x, info = jaxamg.solve(L, b, nullspace="constant", **cfg)
            g = jax.grad(loss)(b)
        assert info["status"] == jaxamg.AMGXStatus.SUCCESS
        assert "rhs_inconsistency" in info
        assert abs(float(jnp.mean(x))) < 1e-10 * float(jnp.linalg.norm(x))
        g_ref = np.linalg.pinv(dense(L)).T @ w
        np.testing.assert_allclose(
            np.asarray(g), g_ref, rtol=1e-6, atol=1e-8 * np.linalg.norm(g_ref)
        )

    def test_singular_matrix_warning(self):
        """A·1 = 0 without a declared null space warns (also at trace time for
        a closed-over matrix); a nonsingular matrix does not."""
        A, V = poisson_matrix_stretched(12, 10, 1.05)  # float32
        b = jnp.asarray(consistent_rhs(np.asarray(V)), dtype=jnp.float32)
        with pytest.warns(NullSpaceWarning, match="singular"):
            jaxamg.solve(A, b, max_iters=5)
        with pytest.warns(NullSpaceWarning, match="singular"):  # memoized verdict
            jaxamg.solve(A, b, max_iters=5)
        with pytest.warns(NullSpaceWarning, match="singular"):
            jax.jit(lambda b_: jaxamg.solve(A, b_, max_iters=5)[0])(b)

        A2 = tridiagonal_matrix(32)
        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            jaxamg.solve(A2, jnp.ones(32), solver="CG")
            jax.jit(lambda b_: jaxamg.solve(A2, b_, solver="CG")[0])(jnp.ones(32))

    def test_nullspace_verification_warning(self, x64):
        A, V = stretched()
        b = jnp.asarray(consistent_rhs(V))
        # The constant vector spans null(A) but not null(Aᵀ) (that is V)...
        with pytest.warns(
            NullSpaceWarning, match="transpose_nullspace does not appear"
        ):
            jaxamg.solve(
                A, b, nullspace="constant", transpose_nullspace="constant", max_iters=5
            )
        # ...and V spans null(Aᵀ) but not null(A).
        with pytest.warns(
            NullSpaceWarning,
            match="nullspace does not appear to span the null space of A:",
        ):
            jaxamg.solve(A, b, nullspace=V, transpose_nullspace=V, max_iters=5)
        with pytest.raises(ValueError, match="dependent"):
            jaxamg.solve(A, b, nullspace=np.zeros(A.shape[0]), max_iters=5)
        # explicit DENSE_LU coarse solve with a null space warns
        with pytest.warns(NullSpaceWarning, match="DENSE_LU"):
            jaxamg.solve(
                A,
                b,
                nullspace="constant",
                transpose_nullspace=V,
                config={"preconditioner": {"coarse_solver": "DENSE_LU_SOLVER"}},
                max_iters=5,
            )

    def test_jit_matches_eager(self, x64):
        A, V = stretched()
        b = jnp.asarray(consistent_rhs(V))
        w = jnp.asarray(weights(A.shape[0]))

        def loss(b_, V_):
            x, _ = jaxamg.solve(
                A, b_, nullspace="constant", transpose_nullspace=V_, **cfg
            )
            return jnp.dot(w, x)

        v_eager, g_eager = jax.value_and_grad(loss)(b, jnp.asarray(V))
        v_jit, g_jit = jax.jit(jax.value_and_grad(loss))(b, jnp.asarray(V))
        np.testing.assert_allclose(v_jit, v_eager, rtol=1e-10)
        np.testing.assert_allclose(g_jit, g_eager, rtol=1e-8)
        info = jax.jit(
            lambda b_: jaxamg.solve(
                A, b_, nullspace="constant", transpose_nullspace=V, **cfg
            )[1]
        )(b)
        assert float(info["rhs_inconsistency"]) < 1e-12

    def test_multidimensional_nullspace(self, x64):
        """Two disconnected grids: a 2-D null space."""
        A1, V1 = stretched(12, 10, 1.06)
        A2, V2 = stretched(10, 8, 1.10)
        n1, n2 = A1.shape[0], A2.shape[0]
        n = n1 + n2
        A = sp.block_diag([to_scipy(A1), to_scipy(A2)], format="csr")
        N = np.zeros((n, 2))
        N[:n1, 0] = 1.0
        N[n1:, 1] = 1.0
        M = np.zeros((n, 2))
        M[:n1, 0] = V1
        M[n1:, 1] = V2
        b = jnp.asarray(np.concatenate([consistent_rhs(V1, 0), consistent_rhs(V2, 1)]))
        w = weights(n)

        def loss(b_):
            x, _ = jaxamg.solve(A, b_, nullspace=N, transpose_nullspace=M, **cfg)
            return jnp.dot(jnp.asarray(w), x)

        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            x, info = jaxamg.solve(A, b, nullspace=N, transpose_nullspace=M, **cfg)
            g = jax.grad(loss)(b)
        assert info["status"] == jaxamg.AMGXStatus.SUCCESS
        x = np.asarray(x)
        assert abs(x[:n1].mean()) < 1e-10 * np.linalg.norm(x)
        assert abs(x[n1:].mean()) < 1e-10 * np.linalg.norm(x)
        g_ref = np.linalg.pinv(A.toarray()).T @ w
        np.testing.assert_allclose(
            np.asarray(g), g_ref, rtol=1e-6, atol=1e-8 * np.linalg.norm(g_ref)
        )

    def test_with_cache_attaches_nullspaces(self, x64):
        A, V = stretched()
        b = jnp.asarray(consistent_rhs(V))
        w = jnp.asarray(weights(A.shape[0]))
        A_c = jaxamg.with_cache(A, nullspace="constant", transpose_nullspace=V)

        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            x, info = jaxamg.solve(A_c, b, **cfg)
            g = jax.grad(lambda b_: jnp.dot(w, jaxamg.solve(A_c, b_, **cfg)[0]))(b)
        assert "rhs_inconsistency" in info
        assert abs(float(jnp.mean(x))) < 1e-10 * float(jnp.linalg.norm(x))
        g_ref = np.linalg.pinv(dense(A)).T @ np.asarray(w)
        np.testing.assert_allclose(
            np.asarray(g), g_ref, rtol=1e-6, atol=1e-8 * np.linalg.norm(g_ref)
        )
