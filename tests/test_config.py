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


@pytest.mark.gpu
def test_config_defaults(linear_system):
    """Test that default configuration works."""
    A, b = linear_system

    x, info = jaxamg.solve(A, b)
    assert info["status"] == jaxamg.AMGXStatus.SUCCESS


@pytest.mark.gpu
def test_config_flat_dict(linear_system):
    """Test simple flat dictionary configuration."""
    A, b = linear_system

    config = {"solver": "CG", "max_iters": 100, "tolerance": 1e-6}
    x, info = jaxamg.solve(A, b, config=config)
    assert info["status"] == jaxamg.AMGXStatus.SUCCESS


@pytest.mark.gpu
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


@pytest.mark.gpu
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


def test_prepare_config_rejects_mpi_dilu_tiny_coarse_grid():
    """Catch the distributed DILU/deep-coarsening config before AmgX crashes."""
    config = {
        "solver": "FGMRES",
        "preconditioner": {
            "solver": "AMG",
            "smoother": {"solver": "MULTICOLOR_DILU"},
        },
    }

    with pytest.raises(ValueError, match="MULTICOLOR_DILU"):
        prepare_config(config, mpi=True)

    # The same config remains allowed for single-GPU mode.
    prepare_config(config, mpi=False)


def test_prepare_config_allows_mpi_dilu_with_coarse_floor():
    """A non-degenerate coarse floor avoids the known distributed DILU failure."""
    cfg = prepare_config(
        {
            "solver": "FGMRES",
            "preconditioner": {
                "solver": "AMG",
                "smoother": {"solver": "MULTICOLOR_DILU"},
                "dense_lu_num_rows": 64,
                "min_coarse_rows": 64,
            },
        },
        mpi=True,
    )
    p = json.loads(cfg)["solver"]["preconditioner"]

    assert p["smoother"]["solver"] == "MULTICOLOR_DILU"
    assert p["dense_lu_num_rows"] == 64


@pytest.mark.parametrize(
    "config, message",
    [
        ({"communicator": "BAD"}, "communicator"),
        ({"max_iters": 0}, "max_iters"),
        ({"tolerance": 0.0}, "tolerance"),
        (
            {"solver": "GMRES", "gmres_n_restart": 0},
            "gmres_n_restart",
        ),
        (
            {"preconditioner": {"solver": "AMG", "max_levels": 0}},
            "max_levels",
        ),
        (
            {"preconditioner": {"solver": "AMG", "presweeps": -1}},
            "presweeps",
        ),
        (
            {"preconditioner": {"solver": "AMG", "smoother": ["JACOBI_L1"]}},
            "smoother",
        ),
    ],
)
def test_prepare_config_rejects_invalid_config_values(config, message):
    """Config validation should catch common malformed values before AmgX."""
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_config(config)


def test_prepare_config_rejects_mpi_dilu_tiny_min_coarse_rows():
    """DILU MPI validation also catches an explicit degenerate min_coarse_rows."""
    config = {
        "solver": "FGMRES",
        "preconditioner": {
            "solver": "AMG",
            "smoother": {"solver": "MULTICOLOR_DILU"},
            "dense_lu_num_rows": 64,
            "min_coarse_rows": 1,
        },
    }

    with pytest.raises(ValueError, match="MULTICOLOR_DILU"):
        prepare_config(config, mpi=True)


def test_outer_max_iters():
    """outer_max_iters reads the outer solver scope of a prepared config."""
    from jaxamg.config import outer_max_iters

    # Default config: jaxamg's default outer max_iters.
    assert outer_max_iters(prepare_config()) == 1000

    # Flat overrides land in the outer scope.
    assert outer_max_iters(prepare_config({"max_iters": 25})) == 25
    assert outer_max_iters(prepare_config(max_iters=7)) == 7

    # Nested config: the outer solver block's max_iters wins; the
    # preconditioner's own max_iters is ignored.
    nested = prepare_config(
        {
            "solver": {
                "solver": "FGMRES",
                "max_iters": 40,
                "preconditioner": {"solver": "AMG", "max_iters": 3},
            }
        }
    )
    assert outer_max_iters(nested) == 40

    # Unparseable input falls back to AmgX's registered default.
    assert outer_max_iters("") == 100
    assert outer_max_iters("not json") == 100


def test_prepare_config_block_dim():
    """block_dim > 1 switches AMG defaults to aggregation; CLASSICAL rejected."""
    cfg = json.loads(prepare_config(block_dim=2))
    precond = cfg["solver"]["preconditioner"]
    assert precond["algorithm"] == "AGGREGATION"
    assert precond["selector"] == "SIZE_2"

    # Scalar default is unchanged.
    cfg1 = json.loads(prepare_config())
    assert cfg1["solver"]["preconditioner"]["algorithm"] == "CLASSICAL"

    # Explicit classical AMG under block_dim > 1 is rejected.
    with pytest.raises(ValueError, match="CLASSICAL"):
        prepare_config(
            {"preconditioner": {"solver": "AMG", "algorithm": "CLASSICAL"}},
            block_dim=2,
        )

    # Non-AMG configs pass through untouched.
    cfg2 = json.loads(
        prepare_config(
            {"solver": "FGMRES", "preconditioner": {"solver": "BLOCK_JACOBI"}},
            block_dim=2,
        )
    )
    assert cfg2["solver"]["solver"] == "FGMRES"


def test_prepare_config_mpi_block_coarse_solver():
    """MPI block defaults avoid AmgX's broken distributed DENSE_LU coarse solve."""
    # MPI + block_dim > 1: coarse solver switches to block-Jacobi sweeps.
    cfg = json.loads(prepare_config(mpi=True, block_dim=2))
    precond = cfg["solver"]["preconditioner"]
    assert precond["coarse_solver"] == {"solver": "BLOCK_JACOBI", "max_iters": 50}
    assert "dense_lu_num_rows" not in precond

    # Single-GPU block and MPI scalar defaults keep DENSE_LU.
    cfg1 = json.loads(prepare_config(block_dim=2))
    assert cfg1["solver"]["preconditioner"]["coarse_solver"] == "DENSE_LU_SOLVER"
    cfg2 = json.loads(prepare_config(mpi=True))
    assert cfg2["solver"]["preconditioner"]["coarse_solver"] == "DENSE_LU_SOLVER"

    # Explicit DENSE_LU coarse solver under MPI + block_dim > 1 is rejected,
    # both as a plain string and as a nested solver scope.
    with pytest.raises(ValueError, match="DENSE_LU_SOLVER"):
        prepare_config(
            {"preconditioner": {"solver": "AMG", "coarse_solver": "DENSE_LU_SOLVER"}},
            mpi=True,
            block_dim=2,
        )
    with pytest.raises(ValueError, match="DENSE_LU_SOLVER"):
        prepare_config(
            {
                "preconditioner": {
                    "solver": "AMG",
                    "coarse_solver": {"solver": "DENSE_LU_SOLVER"},
                }
            },
            mpi=True,
            block_dim=2,
        )

    # DENSE_LU as the outer solver is rejected too.
    with pytest.raises(ValueError, match="DENSE_LU_SOLVER"):
        prepare_config({"solver": "DENSE_LU_SOLVER"}, mpi=True, block_dim=2)

    # ...but stays allowed for single-GPU block and MPI scalar configs.
    prepare_config(
        {"preconditioner": {"solver": "AMG", "coarse_solver": "DENSE_LU_SOLVER"}},
        block_dim=2,
    )
    prepare_config(
        {"preconditioner": {"solver": "AMG", "coarse_solver": "DENSE_LU_SOLVER"}},
        mpi=True,
    )
