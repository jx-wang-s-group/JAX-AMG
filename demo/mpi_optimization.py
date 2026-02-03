"""
Demo: MPI-distributed optimization via automatic differentiation.

This example demonstrates using gradient-based optimization with
JIT-compiled functions in an MPI-distributed setting.

Usage:
    mpirun -n 3 python demo/mpi_optimization.py
"""

import os
from mpi4py import MPI

import jax
import jax.numpy as jnp
import numpy as np

from jaxamg import amg_solve, cache_mpi_metadata, with_cache
from jaxamg.matrices import tridiagonal_matrix_distributed, tridiagonal_matrix

jax.config.update("jax_enable_x64", True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    gpu_ids = [0, 1, 2]
    gpu_id = gpu_ids[rank]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    n_global = 64

    if rank == 0:
        print(f"Setting up MPI-distributed optimization")
        print(f"MPI ranks: {nranks}")
        print(f"Global system size: {n_global}")
        print()

    # Create distributed matrix with diagonal_value=4.0 (same as optimization.py)
    A_local, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0, dtype=jnp.float64
    )
    n_local = row_end - row_start

    if rank == 0:
        print(f"Matrix partitioned across {nranks} ranks")

    # Initial local RHS
    b_init_local = jnp.ones(n_local, dtype=jnp.float64)

    # Configuration
    config = {"solver": "CG"}

    # Cache MPI metadata for JIT-compatible solver usage
    if rank == 0:
        print(f"Caching MPI metadata for JIT...")
    mpi_cache = cache_mpi_metadata(config, comm, n_global, (row_start, row_end))

    # Define loss function
    def loss_local(b_loc):
        A = with_cache(A_local, mpi=mpi_cache)
        x_loc, _ = amg_solve(A, b_loc)
        return jnp.sum(x_loc * x_loc)

    # JIT-compiled loss and gradient function
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_local))

    # Warm up JIT compilation
    _, _ = loss_and_grad_fn(b_init_local)

    # Gradient descent optimization
    learning_rate = 0.01
    num_iterations = 10

    if rank == 0:
        print(f"Gradient Descent Optimization:")
        print(f"Learning rate: {learning_rate}")
        print(f"Iterations: {num_iterations}")
        print()
        print(
            f"{'Iter':<6} {'Global Loss':<15} {'Global Grad Norm':<18} {'Loss Change':<15}"
        )
        print("-" * 70)

    b_current = b_init_local
    loss_prev_global = None

    for i in range(num_iterations):
        # Compute local loss and gradient
        loss_current_local, gradient_local = loss_and_grad_fn(b_current)
        loss_current_local = float(loss_current_local)
        grad_norm_local = float(jnp.sum(gradient_local * gradient_local))

        # Reduce across all ranks to get global loss and gradient norm
        loss_current_global = comm.allreduce(loss_current_local, op=MPI.SUM)
        grad_norm_squared_global = comm.allreduce(grad_norm_local, op=MPI.SUM)
        grad_norm_global = np.sqrt(grad_norm_squared_global)

        # Compute loss change
        if loss_prev_global is not None:
            loss_change = loss_current_global - loss_prev_global
        else:
            loss_change = 0.0

        # Display iteration info (rank 0 only)
        if rank == 0:
            print(
                f"{i:<6} {loss_current_global:<15.6e} {grad_norm_global:<18.6e} {loss_change:<15.6e}"
            )

        # Update b using gradient descent (all ranks)
        b_current = b_current - learning_rate * gradient_local

        loss_prev_global = loss_current_global

    if rank == 0:
        print()
        print("Optimization complete!")

    # Verify against single-GPU (rank 0 only)
    if rank == 0:
        print()
        print("Verifying final loss against single-GPU reference...")

        # Gather final b from all ranks
        b_final_list = comm.gather(np.array(b_current), root=0)

        if b_final_list is not None:
            b_final_global = jnp.concatenate(b_final_list)

            # Compute reference loss on the same final b
            A_global = tridiagonal_matrix(
                n_global, diagonal_value=4.0, dtype=jnp.float64
            )
            x_ref, _ = amg_solve(A_global, b_final_global, solver="CG")
            loss_ref = float(jnp.sum(x_ref * x_ref))

            print(f"  MPI loss:        {loss_current_global:.6e}")
            print(f"  Single-GPU loss: {loss_ref:.6e}")
            print(
                f"  Relative error:  {abs(loss_current_global - loss_ref) / abs(loss_ref):.2e}"
            )
    else:
        comm.gather(np.array(b_current), root=0)


if __name__ == "__main__":
    main()
