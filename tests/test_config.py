import numpy as np
import pytest

import jaxamg
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
