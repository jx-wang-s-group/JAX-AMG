import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from jaxamg import amg_solve, AMGXStatus, with_cache, cache_mpi_metadata
from jaxamg.mpi_utils import (
    gather_solution,
    partition_vector,
    partition_csr_matrix,
    validate_partition,
)
from jaxamg.matrices import (
    poisson_matrix,
    poisson_matrix_distributed,
    rhs_linear,
    tridiagonal_matrix_distributed,
)


@pytest.fixture
def mpi_context():
    """Fixture providing MPI context (comm, rank, nranks)."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    return comm, rank, nranks


@pytest.mark.mpi(min_size=2)
def test_mpi_poisson(mpi_context):
    comm, rank, nranks = mpi_context

    grid_size = 16
    n = grid_size**2

    # Create local matrix for each process
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )

    b_global = rhs_linear(n)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    # Solve the system on each process
    x_local, info = amg_solve(
        A_local,
        b_local,
        comm=comm,
        nglobal=n,
        partition_info=(row_start, row_end),
        solver="PCG",
        preconditioner={"solver": "MULTICOLOR_DILU"},
    )

    # Gather the solution to the root process
    x = gather_solution(x_local, comm, root=0)

    # Check if the solve was successful
    assert info["status"] == AMGXStatus.SUCCESS

    # Check if the solution is correct
    if rank == 0:
        A_global = poisson_matrix(grid_size)
        b_global = rhs_linear(n)
        np.testing.assert_allclose(A_global @ x, b_global, atol=1e-5)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("enable_x64", [False, True])
def test_mpi_autodiff_jit(mpi_context, enable_x64):
    comm, rank, nranks = mpi_context

    # Test with both 32-bit and 64-bit precision
    jax.config.update("jax_enable_x64", enable_x64)

    n_global = 16

    b_global = jnp.ones(n_global)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    # Pre-cache MPI metadata
    config = {"solver": "CG"}
    dummy_A, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, 4.0
    )
    mpi_cache = cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), dummy_A
    )

    def loss_fn(diag_val):
        # Create matrix
        A, _, _ = tridiagonal_matrix_distributed(
            n_global, rank, nranks, diagonal_value=diag_val
        )

        # Attach MPI cache
        A = with_cache(A, mpi=mpi_cache, is_symmetric=True)

        # Solve
        x_local, _ = amg_solve(A, b_local)

        return jnp.sum(x_local**2)

    diag_val = 5.0

    # Compute gradient with JIT and make sure no warnings
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grad = jax.jit(jax.grad(loss_fn))(diag_val)

        # Filter out mpi4jax warnings
        non_mpi4jax_warnings = [
            warning for warning in w if "mpi4jax" not in str(warning.filename)
        ]

        # Fail if there are any warnings
        if non_mpi4jax_warnings:
            raise AssertionError(
                f"Found {len(non_mpi4jax_warnings)} warning(s):\n"
                + "\n".join(
                    f"{w.filename}:{w.lineno}: {w.message}"
                    for w in non_mpi4jax_warnings
                )
            )

    # Compare with finite difference
    check_grads(loss_fn, (diag_val,), order=1, modes=["rev"])

    # Compare with non-JIT execution
    def loss_nojit(diag_val):
        A, _, _ = tridiagonal_matrix_distributed(
            n_global, rank, nranks, diagonal_value=diag_val
        )
        x_local, _ = amg_solve(
            A,
            b_local,
            comm=comm,
            nglobal=n_global,
            partition_info=(row_start, row_end),
            config=config,
        )
        return jnp.sum(x_local**2)

    grad_nojit = jax.grad(loss_nojit)(diag_val)

    # Gradients should match
    np.testing.assert_allclose(grad, grad_nojit)

    # Reset to default precision
    jax.config.update("jax_enable_x64", False)


@pytest.mark.mpi(min_size=2)
def test_mpi_partition(mpi_context):
    comm, rank, nranks = mpi_context

    grid_size = 4
    n = grid_size**2

    # Create local matrix based on predefined partition function
    # for 2D Poisson matrix
    A, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )
    validate_partition(A, n, row_start, row_end)

    # Create local matrix based on auto partition from global matrix
    A_global = poisson_matrix(grid_size)
    A_local, row_start_auto, row_end_auto = partition_csr_matrix(A_global, rank, nranks)
    validate_partition(A_local, n, row_start_auto, row_end_auto)

    # Check if the two partitions are the same
    np.testing.assert_array_equal(A.todense(), A_local.todense())
    np.testing.assert_array_equal(row_start, row_start_auto)
    np.testing.assert_array_equal(row_end, row_end_auto)
