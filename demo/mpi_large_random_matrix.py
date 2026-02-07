"""
Demo: Solve a large random sparse linear system with MPI.

Usage:
    mpirun -n 4 python demo/mpi_large_random_matrix.py
"""

import os
from mpi4py import MPI
import jax.numpy as jnp
import time

from jaxamg import amg_solve
from jaxamg.matrices import random_matrix_distributed, rhs_random


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    gpu_ids = [0, 1, 3]
    gpu_id = gpu_ids[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    n = 100000

    if rank == 0:
        print(f"Setting up a large random sparse matrix of size {n} x {n}...")
        print(f"MPI ranks: {nranks}")
        print(f"\nPartitioning matrix across {nranks} ranks...")

    A_local, row_start, row_end = random_matrix_distributed(
        n, rank, nranks, density=0.01, seed=42, dtype=jnp.float32
    )
    n_local = row_end - row_start

    comm.Barrier()
    print(
        f"  Rank {rank}: {n_local} rows [{row_start}:{row_end}), {len(A_local.data)} non-zeros"
    )
    comm.Barrier()

    b_local = rhs_random(n_local, seed=42 + rank)

    if rank == 0:
        print(f"\nSolving distributed system...")

    # Solver configuration
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "MULTICOLOR_DILU"},
        "communicator": "MPI_DIRECT",
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

    if rank == 0:
        print(f"  Info: {info}")
        print(f"  Solve time: {solve_time:.3f}s\n")


if __name__ == "__main__":
    main()
