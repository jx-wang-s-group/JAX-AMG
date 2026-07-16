import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.sparse.linalg import bicgstab

import jaxamg
from jaxamg import preconditioners
from jaxamg.matrices import poisson_matrix, rhs_ones

lx = pytest.importorskip("lineax")


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
@pytest.mark.gpu
def test_make_preconditioner_with_native_jax_bicgstab(skew):
    """Test that a preconditioner created by `make_preconditioner` can be used with `jax.scipy.sparse.linalg.bicgstab` to solve a skewed Poisson problem."""

    n = 8
    A = poisson_matrix(n, skew=skew)
    b = rhs_ones(n * n)

    M = jaxamg.make_preconditioner(A)
    x, _ = bicgstab(A, b, M=M, tol=1e-6, maxiter=50)

    np.testing.assert_allclose(b, A @ x, rtol=1e-5)


def test_make_lineax_preconditioner_returns_tagged_operator(monkeypatch):
    """`make_lineax_preconditioner` wraps a Lineax operator into a preconditioner
    operator: it inherits the operator's tags, exposes the same input structure,
    and `.mv(r)` applies the AMG approximate inverse to the operator's action."""

    calls = []

    def fake_solve(A, b, config=None, **kwargs):
        calls.append({"A": A, "b": b})
        return b + 1.0, {"iterations": 1, "residual": 0.0, "status": 0}

    monkeypatch.setattr(preconditioners, "solve", fake_solve)

    structure = jax.ShapeDtypeStruct((3,), jnp.float32)
    mat = jnp.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])
    operator = lx.FunctionLinearOperator(
        lambda x: mat @ x,
        structure,
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )

    M = jaxamg.make_lineax_preconditioner(operator)

    assert isinstance(M, lx.FunctionLinearOperator)
    assert M.in_structure() == structure
    # A⁻¹ shares A's symmetry/definiteness, so the tags are carried over (CG needs this).
    assert M.tags == operator.tags

    r = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    out = M.mv(r)
    # fake_solve hands the operator's mv straight through and returns rhs + 1.
    assert jnp.allclose(out, r + 1.0)
    assert callable(calls[0]["A"])  # the operator's matrix-free action was forwarded


def test_make_lineax_preconditioner_tag_override_and_validation(monkeypatch):
    """Explicit `tags` override the inherited ones, and non-1D / non-operator
    inputs are rejected with clear errors."""

    monkeypatch.setattr(
        preconditioners,
        "solve",
        lambda A, b, config=None, **kwargs: (b, {"status": 0}),
    )

    structure = jax.ShapeDtypeStruct((4,), jnp.float32)
    operator = lx.FunctionLinearOperator(
        lambda x: x, structure, tags=(lx.symmetric_tag,)
    )

    M = jaxamg.make_lineax_preconditioner(operator, tags=())
    assert M.tags == frozenset()

    with pytest.raises(TypeError):
        jaxamg.make_lineax_preconditioner(lambda x: x)  # not a Lineax operator

    matrix_2d = lx.FunctionLinearOperator(
        lambda x: x, jax.ShapeDtypeStruct((2, 2), jnp.float32)
    )
    with pytest.raises(ValueError):
        jaxamg.make_lineax_preconditioner(matrix_2d)  # acts on a 2D structure


@pytest.mark.parametrize("skew", [0.0, 2.0])
@pytest.mark.gpu
def test_make_lineax_preconditioner_with_lineax_solver(skew):
    """End-to-end: an AMG preconditioner built from a Lineax operator accelerates a
    Lineax Krylov solve of a (possibly skewed) Poisson problem."""

    n = 8
    A = poisson_matrix(n, skew=skew)
    b = rhs_ones(n * n)
    structure = jax.ShapeDtypeStruct(b.shape, b.dtype)

    symmetric = skew == 0.0
    op_tags = (lx.symmetric_tag, lx.positive_semidefinite_tag) if symmetric else ()
    operator = lx.FunctionLinearOperator(lambda x: A @ x, structure, tags=op_tags)

    M = jaxamg.make_lineax_preconditioner(operator)
    # CG for the SPD case; GMRES for the skewed (nonsymmetric) one.
    solver = (
        lx.CG(rtol=1e-6, atol=1e-6, max_steps=200)
        if symmetric
        else lx.GMRES(rtol=1e-6, atol=1e-6, max_steps=200)
    )
    sol = lx.linear_solve(operator, b, solver=solver, options={"preconditioner": M})

    np.testing.assert_allclose(b, A @ sol.value, rtol=1e-5)
