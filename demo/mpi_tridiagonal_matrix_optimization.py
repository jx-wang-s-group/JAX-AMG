"""
Demo: MPI-distributed optimization of a tridiagonal matrix diagonal parameter.

This example demonstrates parameter optimization in an MPI-distributed setting,
by optimizing the diagonal value of a tridiagonal matrix to match a known solution.

Usage:
    mpirun -n 4 python demo/mpi_tridiagonal_matrix_optimization.py
"""

import os
from mpi4py import MPI

import jax
import jax.numpy as jnp
import numpy as np

from jaxamg import amg_solve, cache_mpi_metadata, with_cache
from jaxamg.matrices import (
    tridiagonal_operator,
    tridiagonal_matrix_distributed,
    rhs_ones,
)
from jaxamg.mpi_utils import get_partition_info

jax.config.update("jax_enable_x64", True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    gpu_ids = [0, 1, 2, 3]
    gpu_id = gpu_ids[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    n_global = 512

    if rank == 0:
        print(
            f"Setting up {n_global}x{n_global} diagonal system with true diagonal = 4.0..."
        )
        print(f"MPI ranks: {nranks}")
        print()

    # Ground truth
    true_diag = 4.0

    # Generate ground truth solution using single-GPU on rank 0
    if rank == 0:
        b_global = rhs_ones(n_global).astype(jnp.float64)
        A_true = tridiagonal_operator(true_diag)
        x_target_global, _ = amg_solve(A_true, b_global)
    else:
        x_target_global = None

    x_target_global = comm.bcast(x_target_global, root=0)

    # Partition the target solution and RHS
    diag_init = 4.5  # Initial guess
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)

    x_target_local = jnp.array(x_target_global[row_start:row_end])
    b_local = rhs_ones(n_local, dtype=jnp.float64)

    comm.Barrier()
    print(f"  Rank {rank}: {n_local} rows [{row_start}:{row_end})")
    comm.Barrier()

    config = {
        "solver": "PCG",
        "preconditioner": {"solver": "MULTICOLOR_DILU"},
    }
    mpi_cache = cache_mpi_metadata(config, comm, n_global, (row_start, row_end))

    # Define loss function
    def loss_local(diag, b_loc, x_true_loc):
        A_loc, _, _ = tridiagonal_matrix_distributed(
            n_global,
            rank,
            nranks,
            diagonal_value=diag,
            dtype=jnp.float64,
        )

        A = with_cache(A_loc, mpi=mpi_cache, is_symmetric=True)
        x_pred_loc, _ = amg_solve(A, b_loc)

        loss_loc = jnp.sum((x_pred_loc - x_true_loc) ** 2)
        return loss_loc

    # JIT-compile the gradient function
    grad_fn = jax.jit(jax.grad(loss_local))

    # Gradient Descent
    lr = 0.01  # Learning rate
    if rank == 0:
        print("\nStarting optimization...")
        print(f"  Learning rate: {lr}")
        print(f"  Initial diagonal: {diag_init:.4f}")
        print()
        print(f"{'Epoch':<6} {'Diagonal':<12} {'Global Loss':<15} {'Gradient':<12}")
        print("-" * 50)

    for epoch in range(100):
        # Compute local loss and gradient
        l_local = loss_local(diag_init, b_local, x_target_local)
        g_local = grad_fn(diag_init, b_local, x_target_local)

        # Reduce across all ranks to get global loss and gradient
        l_global = comm.allreduce(float(l_local), op=MPI.SUM)
        g_global = comm.allreduce(float(g_local), op=MPI.SUM)

        if rank == 0:
            print(f"{epoch:<6} {diag_init:<12.4f} {l_global:<15.6f} {g_global:<12.6f}")

        # Update diagonal (all ranks)
        diag_init = diag_init - lr * g_global

        if l_global < 1e-6:
            if rank == 0:
                print("\nConverged!")
            break

    if rank == 0:
        print()
        print(f"Final diag: {diag_init:.4f}, True diag: {true_diag:.4f}")


if __name__ == "__main__":
    main()
