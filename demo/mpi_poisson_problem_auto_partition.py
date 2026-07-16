"""
Demo: MPI-distributed 2D Poisson solver with automatic partitioning.

Usage:
    mpirun -n 4 python demo/mpi_poisson_problem_auto_partition.py

Demonstrates distributed solving across multiple GPUs with MPI. Local
matrices are created via automatic partitioning of the global matrix.
"""

import time

import jax
import jax.numpy as jnp
from mpi4py import MPI

import jaxamg
from jaxamg.matrices import poisson_matrix, rhs_linear
from jaxamg.mpi_utils import (
    gather_vector,
    partition_csr_matrix,
    validate_partition,
)


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    _gpus = jax.devices()
    jax.config.update("jax_default_device", _gpus[rank % len(_gpus)])

    grid_size = 32
    n = grid_size**2

    if rank == 0:
        print(f"Setting up MPI 2D Poisson problem on {grid_size}x{grid_size} grid...")
        print(f"MPI ranks: {nranks}")
        print("Creating global matrix on rank 0...")

        # Create global matrix
        A_global = poisson_matrix(grid_size)
        b_global = rhs_linear(n)

        print("Broadcasting matrix structure to all ranks...")
    else:
        A_global = None
        b_global = None

    # Broadcast matrix from rank 0 to all ranks
    A_global = comm.bcast(A_global, root=0)
    b_global = comm.bcast(b_global, root=0)

    if rank == 0:
        print(f"Partitioning matrix across {nranks} ranks...")

    # Each rank extracts its local partition
    A_local, row_start, row_end = partition_csr_matrix(A_global, rank, nranks)
    b_local = b_global[row_start:row_end]
    n_local = row_end - row_start

    comm.Barrier()
    print(
        f"Rank {rank}: {n_local} rows [{row_start}:{row_end}), {len(A_local.data)} non-zeros"
    )

    validate_partition(A_local, n, row_start, row_end)

    comm.Barrier()
    if rank == 0:
        print("\nSolving distributed system...")

    # Solver configuration
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "MULTICOLOR_DILU"},
        "tolerance": 1e-6,
        "monitor_residual": 1,
    }

    t_start = time.time()
    x_local, info = jaxamg.solve(
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

    x_mpi = gather_vector(x_local, comm, root=0)

    if rank == 0:
        print("Validating against single-GPU result...")
        x_ref, info_ref = jaxamg.solve(
            poisson_matrix(grid_size),
            rhs_linear(n),
            config=config,
        )

        x_mpi = jnp.asarray(x_mpi)
        diff = jnp.linalg.norm(x_mpi - x_ref) / jnp.linalg.norm(x_ref)

        print(f"  Residual (MPI): {info['residual']:.2e}")
        print(f"  Residual (single-GPU): {info_ref['residual']:.2e}")
        print(f"  First 5 entries (MPI solution): {x_mpi[:5]}")
        print(f"  First 5 entries (single-GPU solution): {x_ref[:5]}")
        print(f"  Relative error: {diff:.2e}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
