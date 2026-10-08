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
    as_labels,
    as_nullspace_basis,
    project_out,
    relative_norm,
    validate_basis,
    validate_label_values,
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


def per_label_columns(B: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """The same null space as dense columns: each column on each label's rows,
    labels whose columns vanish left out."""
    columns = [
        np.where(labels == label, B[:, j], 0.0)
        for label in range(count)
        for j in range(B.shape[1])
    ]
    return np.stack([c for c in columns if np.any(c)], axis=1)


def disconnected(sizes=((12, 10, 1.06), (10, 8, 1.10), (6, 5, 1.0)), normalize=True):
    """Disconnected stretched grids (unequal volumes) as one matrix: its
    labels, the constant vectors and the volume vectors V per part."""
    parts = [stretched(nx, ny, r, normalize=normalize) for nx, ny, r in sizes]
    A = sp.block_diag([to_scipy(a) for a, _ in parts], format="csr")
    labels = np.concatenate([np.full(len(v), i) for i, (_, v) in enumerate(parts)])
    V = np.concatenate([v for _, v in parts])
    return A, labels, V


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

    def test_labels_normalization(self):
        n = 6
        count, rows = as_labels((3, np.array([0, 0, 2, -1, 2, 1])), n)
        assert count == 3 and rows.dtype == jnp.int32
        assert as_labels((5, np.zeros(n, int)), n)[0] == 5
        for bad in [np.zeros(n - 1, int), np.zeros(n), np.array([0, 3, 0, 0, 0, -2])]:
            with pytest.raises(ValueError, match="labels"):
                as_labels((3, bad), n)
        with pytest.raises(ValueError, match="pair"):
            as_labels(np.zeros(n, int), n)  # the count is part of the labels
        with pytest.raises(ValueError, match=r"\[-1, 2\)"):
            as_labels((2, np.array([0, 2, 0, 0, 0, 0])), n)
        traced = jax.jit(lambda r: as_labels((4, r), n)[1])(jnp.arange(n) % 4)
        np.testing.assert_array_equal(traced, np.arange(n) % 4)

    @pytest.mark.parametrize("k", [1, 3])
    def test_project_out_with_labels_is_the_dense_per_label_basis(self, k, x64):
        rng = np.random.default_rng(0)
        n, count = 40, 6
        labels = rng.integers(-1, count - 1, n)  # label count - 1 has no rows: empty
        labels[labels == 2] = -1  # label 2 too
        B = rng.standard_normal((n, k))
        v = rng.standard_normal(n)
        got = np.asarray(
            project_out(
                jnp.asarray(v), jnp.asarray(B), labels=as_labels((count, labels), n)
            )
        )
        dense = per_label_columns(B, labels, count)
        ref = v - dense @ np.linalg.solve(dense.T @ dense, dense.T @ v)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(got[labels == -1], v[labels == -1])
        # One label (plain sums), some rows in none.
        one = np.where(labels >= 0, 0, -1)
        got = np.asarray(
            project_out(jnp.asarray(v), jnp.asarray(B), labels=as_labels((1, one), n))
        )
        dense = per_label_columns(B, one, 1)
        ref = v - dense @ np.linalg.solve(dense.T @ dense, dense.T @ v)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_project_out_with_labels_is_transposed_and_batched_by_jax(self, x64):
        """An orthogonal projector: its reverse-mode derivative (JAX's
        transpose of its linearization) is itself, under vmap as well."""
        rng = np.random.default_rng(1)
        n = 30
        labels = as_labels((4, rng.integers(-1, 4, n)), n)
        B = jnp.asarray(rng.standard_normal((n, 2)))
        project = lambda v: project_out(v, B, labels=labels)
        v, w = (jnp.asarray(rng.standard_normal((3, n))) for _ in range(2))
        (transposed,) = jax.vjp(project, v[0])[1](w[0])
        np.testing.assert_allclose(transposed, project(w[0]), rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(
            jax.vmap(project)(v),
            np.stack([project(x) for x in v]),
            rtol=1e-13,
            atol=1e-14,
        )

    def test_validate_basis_with_labels(self):
        rng = np.random.default_rng(0)
        n = 12
        B = jnp.asarray(rng.standard_normal((n, 2)))
        labels = np.repeat([0, 1, 2], 4)
        validate_basis(B, "nullspace", labels=as_labels((3, labels), n))
        # Columns vanishing on a label's rows (here all of label 1 at -1): skipped.
        validate_basis(
            B, "nullspace", labels=as_labels((3, np.where(labels == 1, -1, labels)), n)
        )
        validate_basis(
            B.at[4:8].set(0.0), "nullspace", labels=as_labels((3, labels), n)
        )
        # Dependent on one label's rows, though independent over all rows.
        dependent = B.at[8:, 1].set(2 * B[8:, 0])
        validate_basis(dependent, "nullspace")
        with pytest.raises(ValueError, match="dependent columns on label 2"):
            validate_basis(dependent, "nullspace", labels=as_labels((3, labels), n))
        with pytest.raises(ValueError, match="finite"):
            validate_basis(
                B.at[0, 0].set(jnp.nan), "nullspace", labels=as_labels((3, labels), n)
            )

    @pytest.mark.parametrize("k", [1, 2])
    @pytest.mark.parametrize(
        "dtype, magnitude", [(jnp.float32, 1e30), (jnp.float64, 1e250)]
    )
    def test_label_projection_preserves_extremely_scaled_components(
        self, k, dtype, magnitude, x64
    ):
        rng = np.random.default_rng(41)
        rows = np.repeat([0, 1, 2, 3, -1], 6)
        labels = as_labels((5, rows), len(rows))  # label 4 has no rows
        B = rng.standard_normal((len(rows), k))
        B[rows == 3] = 0  # an exactly vanishing component
        scales = np.array([1, magnitude, 1 / magnitude, 1, 1])
        scaled = B * np.repeat(scales, 6)[:, None]
        if k == 2:
            scaled[:, 1] = B[:, 1] * np.repeat(1 / scales, 6)
        basis = jnp.asarray(scaled, dtype)
        validate_basis(basis, "nullspace", labels=labels)

        dense = per_label_columns(B, rows, 5)
        q, _ = np.linalg.qr(dense)
        v = rng.standard_normal(len(rows))
        expected = v - q @ (q.T @ v)
        project = lambda v, B, r: project_out(v, B, labels=(5, r))
        args = (jnp.asarray(v, dtype), basis, jnp.asarray(rows))
        atol = 2e-6 if dtype == jnp.float32 else 1e-12
        np.testing.assert_allclose(project(*args), expected, atol=atol, rtol=atol)
        np.testing.assert_allclose(
            jax.jit(project)(*args), expected, atol=atol, rtol=atol
        )
        gradient = jax.jit(
            jax.grad(lambda x, B, r: jnp.vdot(args[0], project(x, B, r)))
        )(*args)
        np.testing.assert_allclose(gradient, expected, atol=atol, rtol=atol)

        if k == 2:
            dependent = basis.at[rows == 2, 1].set(basis[rows == 2, 0])
            with pytest.raises(ValueError, match="dependent columns on label 2"):
                validate_basis(dependent, "nullspace", labels=labels)

    def test_label_scaling_with_rank_local_numbering(self):
        # Simulate collective ranks with vmap, including a label absent from
        # one rank. The callback only supplies sums, never maxima.
        global_rows = np.array([[0, 0, 1, 1], [1, 1, 2, 2]])
        slots = (global_rows + np.arange(2)[:, None]) % 3
        scale = np.array([1e-30, 1, 1e30], np.float32)
        B = np.array([[1, 2, 3, 4], [2, 1, 2, 3]], np.float32)
        basis = jnp.asarray((B * scale[global_rows])[..., None])
        v = jnp.arange(8, dtype=jnp.float32).reshape(2, 4)

        def local(v, B, rows):
            rank = jax.lax.axis_index("rank")
            to_global = (jnp.arange(3) + rank) % 3
            to_local = (jnp.arange(3) - rank) % 3
            reduce = lambda s: jax.lax.psum(s[to_global], "rank")[to_local]
            return project_out(v, B, reduce, (3, rows))

        project = jax.jit(jax.vmap(local, axis_name="rank"))
        dense = per_label_columns(B.reshape(-1, 1), global_rows.ravel(), 3)
        q, _ = np.linalg.qr(dense)
        expected = np.asarray(v).ravel() - q @ (q.T @ np.asarray(v).ravel())
        np.testing.assert_allclose(
            project(v, basis, jnp.asarray(slots)).ravel(), expected, atol=3e-6
        )
        gradient = jax.grad(
            lambda rhs: jnp.vdot(v, project(rhs, basis, jnp.asarray(slots)))
        )(v)
        np.testing.assert_allclose(gradient.ravel(), expected, atol=3e-6)

    def test_label_range_rejection_is_collective(self, mock_mpi):
        from types import SimpleNamespace

        # This rank's labels are valid; the remote rank reports an invalid
        # value after both ranks agree that their labels are concrete.
        answers = iter([True, False])
        comm = SimpleNamespace(allreduce=lambda value, op: next(answers))
        with pytest.raises(ValueError, match=r"\[-1, 2\)"):
            validate_label_values((2, np.array([0, 1])), comm)

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


class TestLabels:
    """Labels on a disconnected domain, against the same null space as dense
    columns and the pseudoinverse (runs the native AmgX solver)."""

    pytestmark = pytest.mark.gpu

    def test_nonsymmetric_constants_and_volumes_per_component(self, x64):
        """A = D⁻¹L on three stretched grids of unequal size: constants per
        component for null(A), V per component for null(Aᵀ); the forward and
        the gradient against the pseudoinverse and against dense columns."""
        A, labels, V = disconnected()
        n = A.shape[0]
        b = jnp.asarray(np.random.default_rng(0).standard_normal(n))
        w = jnp.asarray(weights(n))
        N = per_label_columns(np.ones((n, 1)), labels, 3)
        M = per_label_columns(V[:, None], labels, 3)

        def solve(b_, **declared):
            return jaxamg.solve(A, b_, **declared, **cfg)

        labelled = dict(nullspace="constant", transpose_nullspace=V, labels=(3, labels))
        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            x, info = solve(b, **labelled)
            g = jax.grad(lambda b_: jnp.dot(w, solve(b_, **labelled)[0]))(b)
        x_dense, info_dense = solve(b, nullspace=N, transpose_nullspace=M)
        pinv = np.linalg.pinv(A.toarray())
        np.testing.assert_allclose(
            x,
            pinv @ remove_label_volume(np.asarray(b), labels, V),
            rtol=1e-6,
            atol=1e-8 * np.linalg.norm(x),
        )
        np.testing.assert_allclose(x, x_dense, rtol=1e-6, atol=1e-8 * np.linalg.norm(x))
        np.testing.assert_allclose(
            info["rhs_inconsistency"], info_dense["rhs_inconsistency"], rtol=1e-10
        )
        g_ref = pinv.T @ np.asarray(w)
        np.testing.assert_allclose(
            g, g_ref, rtol=1e-6, atol=1e-8 * np.linalg.norm(g_ref)
        )

    def test_symmetric_labels_in_every_derivative_mode(self, x64):
        """The symmetric flux form: jit, grad, jacfwd and vmap with traced
        (count, labels), against the pseudoinverse."""
        L, labels, _ = disconnected(normalize=False)
        L = jaxamg.with_cache(L, is_symmetric=True)
        n = L.shape[0]
        pinv = np.linalg.pinv(dense(L))
        rng = np.random.default_rng(2)
        b = jnp.asarray(rng.standard_normal(n))
        w = jnp.asarray(weights(n))

        def solve(b_, rows):
            return jaxamg.solve(L, b_, nullspace="constant", labels=(3, rows), **cfg)[0]

        rows = jnp.asarray(labels, jnp.int32)
        x = jax.jit(solve)(b, rows)
        consistent = np.asarray(b) - per_label_mean(np.asarray(b), labels)
        np.testing.assert_allclose(
            x, pinv @ consistent, rtol=1e-6, atol=1e-8 * np.linalg.norm(x)
        )
        g = jax.jit(jax.grad(lambda b_, r: jnp.dot(w, solve(b_, r))))(b, rows)
        np.testing.assert_allclose(
            g, pinv.T @ np.asarray(w), rtol=1e-6, atol=1e-8 * np.linalg.norm(g)
        )
        direction = jnp.asarray(rng.standard_normal(n))
        tangent = jax.jvp(lambda b_: solve(b_, rows), (b,), (direction,))[1]
        np.testing.assert_allclose(
            tangent,
            pinv
            @ (np.asarray(direction) - per_label_mean(np.asarray(direction), labels)),
            rtol=1e-6,
            atol=1e-8 * np.linalg.norm(tangent),
        )
        batch = jnp.stack([b, direction])
        np.testing.assert_allclose(
            jax.vmap(solve, in_axes=(0, None))(batch, rows),
            np.stack([solve(v, rows) for v in batch]),
            rtol=1e-8,
            atol=1e-12,
        )

    def test_per_solve_minus_one_labels_and_empty_labels(self, x64):
        """A component anchored in one solve is labelled -1 there (its rows
        keep their own solution); a label with no rows is skipped exactly."""
        L, labels, _ = disconnected(normalize=False)
        anchored = labels == 0
        L = (L + sp.diags(np.where(anchored, 0.5, 0.0))).tocsr()
        L = jaxamg.with_cache(L, is_symmetric=True)
        n = L.shape[0]
        b = np.random.default_rng(3).standard_normal(n)
        b = b - np.where(anchored, 0.0, per_label_mean(b, labels))
        rows = jnp.asarray(np.where(anchored, -1, labels), jnp.int32)
        solve = jax.jit(
            lambda b_, r: jaxamg.solve(
                L, b_, nullspace="constant", labels=(5, r), **cfg
            )[0]
        )  # labels 3 and 4 have no rows
        x = solve(jnp.asarray(b), rows)
        np.testing.assert_allclose(
            x, np.linalg.pinv(dense(L)) @ b, rtol=1e-6, atol=1e-8 * np.linalg.norm(x)
        )

    def test_several_vectors_per_component_in_a_block_system(self, x64):
        """Two unknowns per node (block_dim=2), uncoupled: the constant of each
        unknown on each component, two columns per label."""
        L, node_labels, _ = disconnected(normalize=False)
        A = jaxamg.with_cache(sp.kron(L, sp.eye(2), format="csr"), is_symmetric=True)
        n = A.shape[0]
        B = np.zeros((n, 2))
        B[0::2, 0] = B[1::2, 1] = 1.0
        labels = np.repeat(node_labels, 2)
        b = np.random.default_rng(4).standard_normal(n)
        consistent = (
            b
            - per_label_columns(B, labels, 3)
            @ np.linalg.lstsq(per_label_columns(B, labels, 3), b, rcond=None)[0]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", NullSpaceWarning)
            x, info = jaxamg.solve(
                A, jnp.asarray(b), block_dim=2, nullspace=B, labels=(3, labels), **cfg
            )
        assert info["status"] == jaxamg.AMGXStatus.SUCCESS
        np.testing.assert_allclose(
            x,
            np.linalg.pinv(dense(A)) @ consistent,
            rtol=1e-6,
            atol=1e-8 * np.linalg.norm(x),
        )

    def test_labels_splitting_a_connected_matrix_warn(self, x64):
        L, _ = stretched(8, 6, 1.1, normalize=False)
        n = L.shape[0]
        split = np.arange(n) % 2  # every coupled pair across two labels
        with pytest.warns(NullSpaceWarning, match="labels split"):
            jaxamg.solve(
                jaxamg.with_cache(L, is_symmetric=True),
                jnp.zeros(n),
                nullspace="constant",
                labels=(2, split),
                max_iters=5,
            )
        with pytest.raises(ValueError, match="declare them"):
            jaxamg.solve(L, jnp.zeros(n), labels=(2, split), max_iters=5)


def per_label_mean(v: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Each labelled row's label mean (zero on rows labelled -1)."""
    out = np.zeros_like(v)
    for label in np.unique(labels[labels >= 0]):
        out[labels == label] = v[labels == label].mean()
    return out


def remove_label_volume(b: np.ndarray, labels: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``b`` projected per label orthogonally to that label's V."""
    out = b.copy()
    for label in np.unique(labels):
        on = labels == label
        out[on] -= V[on] * (V[on] @ b[on]) / (V[on] @ V[on])
    return out


@pytest.mark.parametrize("dtype,magnitude", [(jnp.float32, 1e30), (jnp.float64, 1e250)])
def test_unlabelled_basis_validation_is_scale_safe(dtype, magnitude, x64):
    from jaxamg.nullspace import unit_bases

    basis = np.random.default_rng(73).normal(size=(12, 2))
    scaled = jnp.asarray(basis * [magnitude, 1 / magnitude], dtype)
    validate_basis(scaled, "nullspace")
    v = jnp.arange(12, dtype=dtype)
    expected = project_out(v, jnp.asarray(basis, dtype))
    actual = project_out(v, unit_bases((scaled,))[0])
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
    duplicate = jnp.stack((scaled[:, 0], scaled[:, 0]), axis=1)
    with pytest.raises(ValueError, match="dependent"):
        validate_basis(duplicate, "nullspace")
