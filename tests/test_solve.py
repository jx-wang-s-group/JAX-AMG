"""Test basic solver functionality."""

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.io
import scipy.sparse.linalg as spla

import jaxamg
import jaxamg.jaxamg as jaxamg_module
from jaxamg import config as amgx_config
from jaxamg.jaxamg import _amgx_solve_impl
from jaxamg.matrices import (
    convection_diffusion_matrix_2d,
    download_suitesparse_matrix,
    poisson_matrix,
    rhs_ones,
    tridiagonal_matrix,
)
from jaxamg.utils import to_scipy

# Every test here calls the native AmgX solver (skip logic in conftest.py).
pytestmark = pytest.mark.gpu


class TestSolver:
    """Test basic solver functionality."""

    @pytest.mark.parametrize("n", [32, 256])
    def test_tridiagonal_solve(self, n):
        """Test solving a 1D tridiagonal system against analytical solution."""
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)
        x, info = jaxamg.solve(A, b, solver="CG", max_iters=100)

        # Verify solver status
        if n == 256:
            # CG solver fails to converge for this size within 100 iterations
            assert info["status"] == jaxamg.AMGXStatus.NOT_CONVERGED
        else:
            # CG solver should converge for smaller sizes
            assert info["status"] == jaxamg.AMGXStatus.SUCCESS

        if info["status"] == jaxamg.AMGXStatus.SUCCESS:
            # Verify that Ax = b
            np.testing.assert_allclose(b, A @ x)

            # Compare with solution from SciPy
            # Convert JAX CSR to SciPy for comparison
            A_sp = to_scipy(A)
            x_sp = spla.spsolve(A_sp, np.asarray(b)).astype(np.float32)
            np.testing.assert_allclose(np.asarray(x), x_sp, rtol=1e-5)

    def test_tridiagonal_solve_single_iter(self):
        """Test solving a 1D tridiagonal system with single iteration."""
        n = 32
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)
        x, info = jaxamg.solve(A, b, solver="CG", max_iters=1)

        assert info["status"] == jaxamg.AMGXStatus.NOT_CONVERGED
        assert info["iterations"] == 1

    def test_tridiagonal_solve_jit(self):
        """Test solving a 1D tridiagonal system with JIT compilation."""
        n = 32
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)

        # Create JIT-compiled version
        @jax.jit
        def solve_jit(b):
            x, _ = jaxamg.solve(A, b, solver="CG")
            return x

        # Solve with JIT
        x_jit = solve_jit(b)

        # Solve without JIT
        x_nojit, _ = jaxamg.solve(A, b)

        # Compare results
        np.testing.assert_allclose(x_jit, x_nojit, rtol=1e-6)

    def test_tridiagonal_solve_float64(self):
        """Test solving a 1D tridiagonal system with double precision."""

        # Override the default float32 precision
        jax.config.update("jax_enable_x64", True)

        n = 8
        A = tridiagonal_matrix(n)
        b = rhs_ones(n).astype(jnp.float64)
        x, info = jaxamg.solve(A, b, solver="CG")

        assert info["status"] == jaxamg.AMGXStatus.SUCCESS
        np.testing.assert_allclose(b, A @ x)

        # Check that the solution is float64
        assert x.dtype == jnp.float64

        # Reset to default
        jax.config.update("jax_enable_x64", False)

    def test_tridiagonal_solve_float64_upcasting(self):
        """Test solving a 1D tridiagonal system with float64 matrix and float32 rhs."""

        # Override the default float32 precision
        jax.config.update("jax_enable_x64", True)

        n = 8

        # Explicitly cast matrix to float64
        A = tridiagonal_matrix(n)
        A = jsp.BCSR((A.data.astype(jnp.float64), A.indices, A.indptr), shape=A.shape)

        b = rhs_ones(n).astype(jnp.float32)
        x, info = jaxamg.solve(A, b, solver="CG")

        assert info["status"] == jaxamg.AMGXStatus.SUCCESS
        np.testing.assert_allclose(b, A @ x)

        # Check that the solution is float64
        assert x.dtype == jnp.float64

        # Reset to default
        jax.config.update("jax_enable_x64", False)

    def test_poisson_manufactured_solution(self):
        """Test 2D Poisson with manufactured solution."""
        grid_size = 8
        A = poisson_matrix(grid_size)
        n = grid_size**2

        # Manufactured solution: x = sin(πi/n) * cos(πj/n)
        x_true = np.zeros(n, dtype=np.float32)
        for idx in range(n):
            i = idx // grid_size
            j = idx % grid_size
            x_true[idx] = np.sin(np.pi * i / grid_size) * np.cos(np.pi * j / grid_size)

        # Compute b = A * x_true
        b = jnp.array(A @ x_true)

        # Solve
        x_computed, _ = jaxamg.solve(A, b)

        # Compare with true solution
        np.testing.assert_allclose(x_computed, x_true, atol=1e-6)

    @pytest.mark.parametrize("grid_size", [4, 8, 16])
    def test_poisson_solve(self, grid_size):
        """Test solving 2D Poisson against SciPy solution."""
        A = poisson_matrix(grid_size)
        n = grid_size**2
        b = rhs_ones(n)

        # Solve
        x, _ = jaxamg.solve(A, b)

        # Solve with Scipy
        A_sp = to_scipy(A)
        x_sp = spla.spsolve(A_sp, np.asarray(b))

        # Compare solutions
        np.testing.assert_allclose(x, x_sp, rtol=1e-6)

        # Check residual
        residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
        np.testing.assert_allclose(residual, 0.0, atol=1e-5)

    @pytest.mark.parametrize("solver", ["CG", "BICGSTAB", "GMRES"])
    def test_nonsymmetric_solve(self, solver):
        """Test solving 2D non-symmetric problem."""
        grid_size = 5
        A = poisson_matrix(grid_size, skew=1.0)
        n = grid_size**2
        b = rhs_ones(n)

        # Solve (with Jacobi preconditioner)
        x, info = jaxamg.solve(A, b, solver=solver, preconditioner="JACOBI_L1")

        # CG should not converge, BICGSTAB and GMRES should converge
        if solver == "CG":
            assert info["status"] == jaxamg.AMGXStatus.NOT_CONVERGED
        else:
            assert info["status"] == jaxamg.AMGXStatus.SUCCESS

            # Check solution
            np.testing.assert_allclose(b, A @ x, rtol=1e-5)

    @pytest.mark.parametrize("solver", ["CG", "PBICGSTAB"])
    def test_convection_diffusion_solve(self, solver):
        """Test solving 2D convection-diffusion equation with analytic solution."""

        n = 64  # Grid size
        h = 1.0 / (n - 1)  # Grid spacing

        # Parameters
        epsilon = 1e-3
        velocity = 1.0
        theta = np.pi / 4.0

        # Velocity components
        vx = velocity * np.cos(theta)
        vy = velocity * np.sin(theta)

        # Grid
        x = jnp.linspace(0, 1, n)
        y = jnp.linspace(0, 1, n)
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        # Analytic solution: u = sin(pi*x) * sin(pi*y)
        u_exact = jnp.sin(np.pi * X) * jnp.sin(np.pi * Y)
        laplacian_u = -2 * np.pi**2 * u_exact
        grad_u_x = np.pi * jnp.cos(np.pi * X) * jnp.sin(np.pi * Y)
        grad_u_y = np.pi * jnp.sin(np.pi * X) * jnp.cos(np.pi * Y)

        # Source term f = -epsilon*Delta u + v.Grad u
        f = -epsilon * laplacian_u + vx * grad_u_x + vy * grad_u_y
        b = f.ravel()
        u_exact_flat = u_exact.ravel()

        # Matrix
        A = convection_diffusion_matrix_2d(
            n, epsilon=epsilon, theta=theta, velocity=velocity
        )

        u, info = jaxamg.solve(A, b, solver=solver)

        # Verify solver status
        if solver == "CG":
            # CG should not converge
            assert info["status"] == jaxamg.AMGXStatus.NOT_CONVERGED
        else:
            # PBICGSTAB with AMG preconditioner should converge
            assert info["status"] == jaxamg.AMGXStatus.SUCCESS

            # Verify solution
            np.testing.assert_allclose(A @ u, b, atol=1e-4)
            np.testing.assert_allclose(u, u_exact_flat, atol=0.1)

    def test_transpose_solve(self):
        """Test backend transpose solve against explicit A^T construction."""
        grid_size = 8
        A = poisson_matrix(grid_size, skew=0.7)
        n = grid_size**2

        # Use a deterministic RHS for A^T lambda = g_x
        key = jax.random.PRNGKey(0)
        g_x = jax.random.normal(key, (n,), dtype=jnp.float32)

        # Build config string exactly as production solve path does
        config_str = amgx_config.prepare_config(
            {"solver": "PBICGSTAB", "preconditioner": "JACOBI_L1", "tolerance": 1e-6}
        )

        # Solve transpose directly in backend
        lam_backend, info_backend = _amgx_solve_impl(
            A.indptr,
            A.indices,
            A.data,
            g_x,
            config_str=config_str,
            transpose_solve=True,
        )

        # Explicitly materialize A^T then solve
        A_T = jsp.BCSR.from_bcoo(A.to_bcoo().transpose())
        lam_explicit, info_explicit = _amgx_solve_impl(
            A_T.indptr,
            A_T.indices,
            A_T.data,
            g_x,
            config_str=config_str,
            transpose_solve=False,
        )

        assert info_backend[2] == info_explicit[2] == jaxamg.AMGXStatus.SUCCESS
        np.testing.assert_allclose(lam_backend, lam_explicit)

    def test_reuse_setup_forwards_to_adjoint_solve(self, monkeypatch):
        """The VJP must use the same reuse_setup mode as the forward solve."""
        calls = []

        def fake_amgx_solve(
            row_ptrs,
            col_indices,
            values,
            b,
            config_str="",
            transpose_solve=False,
            return_stats=False,
            reuse_setup=False,
        ):
            calls.append(
                {
                    "transpose_solve": bool(transpose_solve),
                    "reuse_setup": bool(reuse_setup),
                }
            )
            return b, jnp.array([0, 0, 0], dtype=b.dtype)

        monkeypatch.setattr(jaxamg_module, "_amgx_solve_impl", fake_amgx_solve)
        jaxamg_module._get_solver_primitive.cache_clear()

        try:
            A = poisson_matrix(2, skew=0.7)
            b = rhs_ones(A.shape[0])
            solver = jaxamg_module._get_solver_primitive(
                "test-config", reuse_setup=True
            )

            def loss(rhs):
                x, _ = solver(A, rhs)
                return jnp.sum(x)

            grad_b = jax.grad(loss)(b)

            assert grad_b.shape == b.shape
            assert calls[0] == {"transpose_solve": False, "reuse_setup": True}
            assert any(
                call == {"transpose_solve": True, "reuse_setup": True}
                for call in calls[1:]
            )
        finally:
            jaxamg_module._get_solver_primitive.cache_clear()

    def test_save_stats_file(self, tmp_path):
        """Test that jaxamg.solve generates a stats file correctly."""
        n = 32
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)

        stats_file = tmp_path / "test_stats.txt"

        # Solve with AMG to generate both solver iterations and grid stats
        x, info = jaxamg.solve(A, b, save_stats_file=stats_file)

        assert stats_file.exists()

        content = stats_file.read_text()
        # Verify both AMG grid stats and solver iterations are present
        assert "AMG GRID STATISTICS" in content
        assert "SOLVER ITERATIONS" in content

    def test_vmap_solve(self):
        """Test that jax.vmap works out-of-the-box for jaxamg.solve via sequential FFI."""
        n = 32
        batch_size = 5
        A = tridiagonal_matrix(n)

        # Create a batch of RHS vectors
        # Shape: (batch_size, n)
        b_batched = jax.random.normal(jax.random.PRNGKey(0), (batch_size, n))

        # Vmap over the 0th axis of b_batched: we pass A as unscanned (None)
        # solve signature: solve(A, b, ...)
        vmap_solve = jax.vmap(jaxamg.solve, in_axes=(None, 0))

        x_batched, info_batched = vmap_solve(A, b_batched)

        assert x_batched.shape == (batch_size, n)

        # Verify mathematically by comparing against numpy/scipy sequential loops
        import jax.numpy.linalg as jla

        A_dense = A.todense()

        for i in range(batch_size):
            x_expected = jla.solve(A_dense, b_batched[i])
            np.testing.assert_allclose(x_batched[i], x_expected, rtol=1e-4, atol=1e-5)
            # Verify status array is broadcasted and successful
            assert info_batched["status"][i] == jaxamg.AMGXStatus.SUCCESS

    def test_vmap_batched_solve(self):
        """Test that jax.vmap works with batched matrices and routes each batch
        member's data to the right lane (distinct matrices and RHS per lane)."""
        n = 8
        A1 = tridiagonal_matrix(n, diagonal_value=2.0)
        A2 = tridiagonal_matrix(n, diagonal_value=4.0)
        b1 = rhs_ones(n)
        b2 = 2.0 * rhs_ones(n)

        A_batched = jsp.BCSR(
            (
                jnp.stack([A1.data, A2.data]),
                jnp.stack([A1.indices, A2.indices]),
                jnp.stack([A1.indptr, A2.indptr]),
            ),
            shape=(2, n, n),
        )
        b_batched = jnp.stack([b1, b2])

        # Vmap over the 0th axis of A_batched and b_batched
        vmap_solve = jax.vmap(jaxamg.solve, in_axes=(0, 0))

        # This should NOT raise ValueError because inside vmap A is not batched
        x_batched, info_batched = vmap_solve(A_batched, b_batched)

        assert x_batched.shape == (2, n)
        for i, (A, b) in enumerate(((A1, b1), (A2, b2))):
            x_expected = jnp.linalg.solve(A.todense(), b)
            np.testing.assert_allclose(x_batched[i], x_expected, rtol=1e-4, atol=1e-5)
            assert info_batched["status"][i] == jaxamg.AMGXStatus.SUCCESS

    def test_get_solver_cache_info(self):
        """get_solver_cache_info reports the cached solver with a parsed config."""
        jaxamg.clear_solver_cache()
        n = 16
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)
        jaxamg.solve(A, b)

        info = jaxamg.get_solver_cache_info()
        assert set(info) >= {"single_gpu", "mpi", "isolated_mode"}
        if info["isolated_mode"]:
            pytest.skip("JAXAMG_CACHE_SIZE=0: nothing is cached in isolated mode")

        single = info["single_gpu"]
        assert single["size"] == len(single["entries"]) == 1
        assert single["capacity"] >= 1
        entry = single["entries"][0]
        assert entry["n_rows"] == n
        assert entry["nnz"] == 3 * n - 2
        assert entry["mode"] == "float32"
        # The config string must round-trip to the prepared default config.
        assert isinstance(entry["config"], dict)
        assert entry["config"]["solver"]["solver"] == "PBICGSTAB"

    def test_jitted_iterative_solve_matches_eager(self):
        """Regression test for the missing device sync after AmgX solves.

        An FFI solve inside a jitted loop feeds XLA kernels immediately; before
        the trailing ``cudaDeviceSynchronize`` in the native handler, those
        consumers could read the solution buffer before AmgX's download landed
        (stale zeros/garbage under GPU contention). The jitted Richardson
        iteration must match the identical iteration run eagerly, where
        ``block_until_ready`` guarantees every download has completed.
        """
        n = 8
        steps = 20
        A = poisson_matrix(n)
        b = rhs_ones(n * n)
        apply = jaxamg.make_preconditioner(A)

        @jax.jit
        def richardson(v):
            def body(_, x):
                return x + apply(v - A @ x)

            return jax.lax.fori_loop(0, steps, body, jnp.zeros_like(v))

        x_ref = jnp.zeros_like(b)
        for _ in range(steps):
            z = apply(b - A @ x_ref)
            z.block_until_ready()
            x_ref = x_ref + z

        x_jit = richardson(b)
        np.testing.assert_allclose(
            np.asarray(x_jit), np.asarray(x_ref), rtol=1e-4, atol=1e-4
        )

    def test_batched_matrix_error(self):
        """Test that passing a batched matrix directly raises ValueError."""
        n = 8
        A = tridiagonal_matrix(n)
        b = rhs_ones(n)

        # Create a batched version by stacking
        A_batched = jsp.BCSR(
            (
                jnp.stack([A.data, A.data]),
                jnp.stack([A.indices, A.indices]),
                jnp.stack([A.indptr, A.indptr]),
            ),
            shape=(2, n, n),
        )

        with pytest.raises(ValueError, match="does not support batched BCSR matrices"):
            jaxamg.solve(A_batched, b)

    def test_cache_distinguishes_sparsity_pattern(self):
        """Reusing the solver cache must respect col_indices, not just row_ptrs.

        Two SPD matrices with identical row pointers and nnz but different column
        indices are solved back-to-back in one process. The process-global C++
        solver cache must treat them as distinct structures; if it keys only on
        the row pointers, the second solve reuses the first matrix's graph (a
        values-only ``AMGX_matrix_replace_coefficients``) and returns a wrong
        solution.
        """
        import scipy.sparse as sp

        def cycle_laplacian(n, step, alpha):
            # alpha*I + graph Laplacian of the 2-regular graph i <-> (i +/- step).
            # Exactly 3 nonzeros per row, so row_ptrs are independent of `step`;
            # only the column indices (the pattern) change with `step`.
            rows, cols, vals = [], [], []
            for i in range(n):
                nbrs = sorted({(i - step) % n, (i + step) % n})
                assert len(nbrs) == 2 and i not in nbrs
                for j in (i, *nbrs):
                    rows.append(i)
                    cols.append(j)
                    vals.append((alpha + 2.0) if j == i else -1.0)
            A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
            A.sort_indices()
            return A

        n = 64
        A1 = cycle_laplacian(n, step=1, alpha=1.0)
        A2 = cycle_laplacian(n, step=2, alpha=3.0)

        # Same row pointers and nnz, different columns: only the pattern differs.
        np.testing.assert_array_equal(A1.indptr, A2.indptr)
        assert A1.nnz == A2.nnz
        assert not np.array_equal(A1.indices, A2.indices)

        rng = np.random.default_rng(0)
        b = jnp.asarray(rng.standard_normal(n).astype(np.float32))
        cfg = {"solver": "CG", "tolerance": 1e-10, "max_iters": 1000}

        # Clean cache so A1 populates it and A2 must not silently reuse it.
        jaxamg.clear_solver_cache()
        x1, _ = jaxamg.solve(A1, b, config=cfg)
        x2, _ = jaxamg.solve(A2, b, config=cfg)

        # Each solution must satisfy its own system.
        np.testing.assert_allclose(A1 @ np.asarray(x1), np.asarray(b), atol=1e-4)
        np.testing.assert_allclose(A2 @ np.asarray(x2), np.asarray(b), atol=1e-4)

    def test_download_suitesparse_matrix_from_cache(self, tmp_path):
        """SuiteSparse helper should load a cached Matrix Market tarball."""
        import tarfile

        group = "TestGroup"
        name = "toy"
        matrix_dir = tmp_path / group / name
        matrix_dir.mkdir(parents=True)

        A = scipy.sparse.csr_matrix(
            np.array([[2.0, 0.0, -1.0], [0.0, 3.0, 4.0]], dtype=np.float64)
        )
        source_dir = tmp_path / "source" / name
        source_dir.mkdir(parents=True)
        source_mtx = source_dir / f"{name}.mtx"
        scipy.io.mmwrite(source_mtx, A)

        archive_path = matrix_dir / f"{name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as archive:
            archive.add(source_mtx, arcname=f"{name}/{name}.mtx")

        loaded = download_suitesparse_matrix(
            f"{group}/{name}", cache_dir=tmp_path, dtype=np.float32
        )

        assert scipy.sparse.isspmatrix_csr(loaded)
        assert loaded.dtype == np.float32
        np.testing.assert_allclose(loaded.toarray(), A.astype(np.float32).toarray())
