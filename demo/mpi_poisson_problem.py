"""
Demo: MPI-distributed 2D Poisson solver.

Usage:
    mpirun -n 4 python demo/mpi_poisson_problem.py

Demonstrates distributed solving across multiple GPUs with MPI.
Validates results against single-GPU reference solution.
"""

import os
from mpi4py import MPI
import jax.numpy as jnp
import time

from jaxamg import amg_solve
from jaxamg.mpi_utils import (
    validate_partition,
    gather_solution,
    partition_vector,
)
from jaxamg.matrices import poisson_matrix, poisson_matrix_distributed, rhs_linear


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    gpu_id = rank
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    grid_size = 32
    n = grid_size**2

    if rank == 0:
        print(f"Setting up MPI 2D Poisson problem on {grid_size}x{grid_size} grid...")
        print(f"MPI ranks: {nranks}")
        print(f"\nPartitioning matrix across {nranks} ranks...")

    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks, dtype=jnp.float32
    )
    n_local = row_end - row_start

    comm.Barrier()

    print(
        f"  Rank {rank}: {n_local} rows [{row_start}:{row_end}), {len(A_local.data)} non-zeros"
    )
    comm.Barrier()

    validate_partition(A_local, n, row_start, row_end)

    b_global_np = rhs_linear(n)
    b_local, _, _ = partition_vector(b_global_np, rank, nranks)

    if rank == 0:
        print(f"\nSolving distributed system...")

    # Solver configuration
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "MULTICOLOR_DILU"},
        "tolerance": 1e-6,
        "monitor_residual": 1,
    }

    t_start = time.time()
    x_local, info = amg_solve(
        A_local,
        b_local,
        config=config,
        comm=comm,
        nglobal=n,
        partition_info=(row_start, row_end),
    )
    solve_time = time.time() - t_start

    comm.Barrier()
    for r in range(nranks):
        comm.Barrier()

    if rank == 0:
        print(f"  Info: {info}")
        print(f"  Solve time: {solve_time:.3f}s\n")

    x_mpi = gather_solution(x_local, comm, root=0)

    if rank == 0:
        print(f"Validating against single-GPU result...")
        x_ref, info_ref = amg_solve(
            poisson_matrix(grid_size),
            rhs_linear(n),
            config=config,
        )

        diff = jnp.linalg.norm(x_mpi - x_ref) / jnp.linalg.norm(x_ref)
        print(f"  Residual (MPI): {info['residual']:.2e}")
        print(f"  Residual (single-GPU): {info_ref['residual']:.2e}")
        print(f"  First 5 entries (MPI solution): {x_mpi[:5]}")
        print(f"  First 5 entries (single-GPU solution): {x_ref[:5]}")
        print(f"  Relative error: {diff:.2e}")


if __name__ == "__main__":
    main()
