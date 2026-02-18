import json

import numpy as np
import pytest

import jaxamg
from jaxamg.config import prepare_config
from jaxamg.matrices import rhs_ones, tridiagonal_matrix


@pytest.fixture
def linear_system():
    """Fixture providing a simple tridiagonal system."""
    n = 8
    A = tridiagonal_matrix(n)
    b = rhs_ones(n)
    return A, b


def test_config_defaults(linear_system):
    """Test that default configuration works."""
    A, b = linear_system

    x, info = jaxamg.solve(A, b)
    assert info["status"] == jaxamg.AMGXStatus.SUCCESS


def test_config_flat_dict(linear_system):
    """Test simple flat dictionary configuration."""
    A, b = linear_system

    config = {"solver": "CG", "max_iters": 100, "tolerance": 1e-6}
    x, info = jaxamg.solve(A, b, config=config)
    assert info["status"] == jaxamg.AMGXStatus.SUCCESS


def test_config_kwargs_override(linear_system):
    """Test that kwargs override default and config values."""
    A, b = linear_system

    # Force 1 iteration via kwargs
    x, info = jaxamg.solve(A, b, max_iters=1)
    assert info["iterations"] == 1

    # Verify override of passed config
    # Use CG with no preconditioner to ensure it takes more than 1 iteration
    config = {"max_iters": 100}
    # We pass explicit solver/preconditioner to ensure slow convergence
    x, info = jaxamg.solve(
        A,
        b,
        config=config,
        max_iters=2,
        solver="CG",
        preconditioner="NOSOLVER",
        tolerance=1e-16,
    )
    # CG without preconditioner should not converge in 2 iterations
    assert info["status"] == jaxamg.AMGXStatus.NOT_CONVERGED
    assert info["iterations"] == 2


def test_config_nested(linear_system):
    """Test nested dictionary configuration."""
    A, b = linear_system

    # Explicit nested config structure
    config = {
        "config_version": 2,
        "solver": {
            "solver": "PBICGSTAB",
            "preconditioner": {"solver": "AMG", "smoother": "JACOBI_L1"},
            "max_iters": 200,
            "tolerance": 1e-6,
        },
    }
    _, info = jaxamg.solve(A, b, config=config)
    assert info["status"] == jaxamg.AMGXStatus.SUCCESS

    # Check residual tracking injection is successful even if not explicitly requested
    assert np.isfinite(info["residual"])


def test_prepare_config_amg_defaults_preserved():
    """Test that AMG preconditioner with partial override keeps AMG defaults."""
    cfg = prepare_config(
        {"preconditioner": {"solver": "AMG", "smoother": "MULTICOLOR_GS"}}
    )
    p = json.loads(cfg)["solver"]["preconditioner"]

    assert p["solver"] == "AMG"
    assert p["smoother"] == "MULTICOLOR_GS"
    assert p["algorithm"] == "CLASSICAL"
    assert p["selector"] == "PMIS"
    assert p["coarse_solver"] == "DENSE_LU_SOLVER"


def test_prepare_config_non_amg_preconditioner_clean():
    """Test that non-AMG preconditioner should not carry AMG-specific keys."""
    cfg = prepare_config({"preconditioner": {"solver": "JACOBI_L1"}})
    p = json.loads(cfg)["solver"]["preconditioner"]

    assert p["solver"] == "JACOBI_L1"
    assert "algorithm" not in p
    assert "selector" not in p
    assert "coarse_solver" not in p


def test_prepare_config_string_preconditioner():
    """Test that preconditioner passed as a plain string."""
    cfg = prepare_config({"preconditioner": "NOSOLVER"})
    p = json.loads(cfg)["solver"]["preconditioner"]
    assert p == "NOSOLVER"


def test_prepare_config_no_user_config():
    """Test that default config should have full AMG preconditioner."""
    cfg = prepare_config()
    p = json.loads(cfg)["solver"]["preconditioner"]

    assert p["solver"] == "AMG"
    assert p["algorithm"] == "CLASSICAL"


def test_prepare_config_nested_merges_defaults():
    """Test that nested config merges defaults."""
    cfg = prepare_config(
        {
            "config_version": 2,
            "solver": {
                "solver": "PCG",
                "preconditioner": {"solver": "JACOBI_L1"},
            },
        }
    )
    solver_cfg = json.loads(cfg)["solver"]

    assert solver_cfg["solver"] == "PCG"
    assert solver_cfg["preconditioner"]["solver"] == "JACOBI_L1"
    assert solver_cfg["convergence"] == "RELATIVE_INI"
    assert solver_cfg["max_iters"] == 1000
    assert solver_cfg["tolerance"] == 1e-6
    assert "algorithm" not in solver_cfg["preconditioner"]


def test_prepare_config_nested_kwargs_override():
    """Test that kwargs override nested config."""
    cfg = prepare_config(
        {
            "config_version": 2,
            "solver": {
                "solver": "PCG",
                "max_iters": 200,
            },
        },
        max_iters=7,
    )
    solver_cfg = json.loads(cfg)["solver"]
    assert solver_cfg["max_iters"] == 7
