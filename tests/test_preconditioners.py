import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.sparse.linalg import bicgstab

import jaxamg
from jaxamg import preconditioners
from jaxamg.matrices import poisson_matrix, rhs_ones


def test_make_preconditioner_uses_amg_defaults(monkeypatch):
    """Test that `make_preconditioner` calls `solve` with the expected default config for AMG preconditioners."""
    calls = []

    def fake_solve(A, b, config=None, **kwargs):
        calls.append({"A": A, "b": b, "config": config, "kwargs": kwargs})
        return b + 1, {"iterations": 1, "residual": 0.0, "status": 0}

    monkeypatch.setattr(preconditioners, "solve", fake_solve)

    M = jaxamg.make_preconditioner("A")
    x = M(jnp.array([1.0, 2.0], dtype=jnp.float32))

    assert jnp.allclose(x, jnp.array([2.0, 3.0], dtype=jnp.float32))
    assert calls[0]["config"]["config_version"] == 2
    assert calls[0]["config"]["solver"]["solver"] == "AMG"
    assert calls[0]["config"]["solver"]["max_iters"] == 1


def test_make_preconditioner_returns_info(monkeypatch):
    """Test that `make_preconditioner(..., return_info=True)` returns the info dictionary from `solve`."""

    def fake_solve(A, b, config=None, **kwargs):
        return b, {"iterations": 3, "residual": 1e-3, "status": 0}

    monkeypatch.setattr(preconditioners, "solve", fake_solve)

    M = jaxamg.make_preconditioner("A", return_info=True, max_iters=2)
    x, info = M(jnp.array([1.0], dtype=jnp.float32))

    assert jnp.allclose(x, jnp.array([1.0], dtype=jnp.float32))
    assert info["iterations"] == 3


def test_make_preconditioner_forwards_mpi_args(monkeypatch):
    """Test that `make_preconditioner` forwards MPI-related arguments to `solve` when provided."""

    calls = []
    fake_comm = object()

    def fake_solve(
        A,
        b,
        config=None,
        comm=None,
        nglobal=None,
        partition_info=None,
        save_stats_file=None,
    ):
        calls.append(
            {
                "A": A,
                "b": b,
                "config": config,
                "comm": comm,
                "nglobal": nglobal,
                "partition_info": partition_info,
                "save_stats_file": save_stats_file,
            }
        )
        return b, {"iterations": 1, "residual": 0.0, "status": 0}

    monkeypatch.setattr(preconditioners, "solve", fake_solve)

    M = jaxamg.make_preconditioner(
        "A_local",
        comm=fake_comm,
        nglobal=16,
        partition_info=(4, 8),
        save_stats_file="stats.txt",
    )
    M(jnp.array([1.0, 2.0], dtype=jnp.float32))

    assert calls[0]["comm"] is fake_comm
    assert calls[0]["nglobal"] == 16
    assert calls[0]["partition_info"] == (4, 8)
    assert calls[0]["save_stats_file"] == "stats.txt"


def test_make_preconditioner_nested_config_merge(monkeypatch):
    """Test that `make_preconditioner` correctly merges nested config dictionaries and kwargs into the final config passed to `solve`."""

    calls = []

    def fake_solve(A, b, config=None, **kwargs):
        calls.append(config)
        return b, {"iterations": 1, "residual": 0.0, "status": 0}

    monkeypatch.setattr(preconditioners, "solve", fake_solve)

    M = jaxamg.make_preconditioner(
        "A",
        config={
            "config_version": 2,
            "solver": {"solver": "AMG", "smoother": {"solver": "JACOBI_L1"}},
        },
        cycle="W",
    )
    M(jnp.array([1.0], dtype=jnp.float32))

    solver_cfg = calls[0]["solver"]
    assert solver_cfg["solver"] == "AMG"
    assert solver_cfg["smoother"]["solver"] == "JACOBI_L1"
    assert solver_cfg["cycle"] == "W"
    assert solver_cfg["coarse_solver"] == "DENSE_LU_SOLVER"


@pytest.mark.parametrize("skew", [0.0, 1.0, 2.0])
def test_make_preconditioner_with_native_jax_bicgstab(skew):
    """Test that a preconditioner created by `make_preconditioner` can be used with `jax.scipy.sparse.linalg.bicgstab` to solve a skewed Poisson problem."""

    n = 8
    A = poisson_matrix(n, skew=skew)
    b = rhs_ones(n * n)

    M = jaxamg.make_preconditioner(A)
    x, _ = bicgstab(A, b, M=M, tol=1e-6, maxiter=50)

    np.testing.assert_allclose(b, A @ x, rtol=1e-5)
