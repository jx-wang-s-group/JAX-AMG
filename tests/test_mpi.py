import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import pytest
import jax.numpy as jnp
import numpy as np

from jaxamg import amg_solve, AMGXStatus
from jaxamg.mpi_utils import (
    gather_solution,
    partition_vector,
)
from jaxamg.matrices import poisson_matrix, poisson_matrix_distributed, rhs_linear


@pytest.mark.mpi(min_size=2)
def test_mpi_poisson():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    grid_size = 16
    n = grid_size**2

    # Create local matrix for each process
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks, dtype=jnp.float32
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
