import pytest
import numpy as np
import jax.numpy as jnp
from jaxamg import amg_solve, AMGXStatus
from jaxamg.matrices import tridiagonal_matrix, rhs_ones


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

    x, info = amg_solve(A, b)
    assert info["status"] == AMGXStatus.SUCCESS


def test_config_flat_dict(linear_system):
    """Test simple flat dictionary configuration."""
    A, b = linear_system

    config = {"solver": "CG", "max_iters": 100, "tolerance": 1e-6}
    x, info = amg_solve(A, b, config=config)
    assert info["status"] == AMGXStatus.SUCCESS


def test_config_kwargs_override(linear_system):
    """Test that kwargs override default and config values."""
    A, b = linear_system

    # Force 1 iteration via kwargs
    x, info = amg_solve(A, b, max_iters=1)
    assert info["iterations"] == 1

    # Verify override of passed config
    # Use CG with no preconditioner to ensure it takes more than 1 iteration
    config = {"max_iters": 100}
    # We pass explicit solver/preconditioner to ensure slow convergence
    x, info = amg_solve(
        A,
        b,
        config=config,
        max_iters=2,
        solver="CG",
        preconditioner="NOSOLVER",
        tolerance=1e-16,
    )
    # CG without preconditioner should not converge in 2 iterations
    assert info["status"] == AMGXStatus.NOT_CONVERGED
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
    x, info = amg_solve(A, b, config=config)
    assert info["status"] == AMGXStatus.SUCCESS

    # Check residual tracking injection is successful even if not explicitly requested
    assert np.isfinite(info["residual"])


def test_config_invalid_solver(linear_system):
    """Test invalid solver name raises an exception."""
    A, b = linear_system
    config = {"solver": "INVALID_SOLVER_NAME"}

    with pytest.raises(Exception) as excinfo:
        amg_solve(A, b, config=config)

    # Check that the exception message comes from AmgX
    assert "AMGX Error" in str(excinfo.value) or "Incorrect parameters" in str(
        excinfo.value
    )
