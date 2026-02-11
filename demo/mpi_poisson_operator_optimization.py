"""
Demo: MPI-distributed optimization of a Poisson operator skew parameter.

Demonstrates end-to-end JIT compilation and differentiation of a custom JAX operator
in an MPI setting.

Usage:
    mpirun -n 4 python demo/mpi_poisson_operator_optimization.py
"""

import jax
import jax.numpy as jnp
from mpi4py import MPI

import jaxamg
from jaxamg.matrices import (
    poisson_operator,
    poisson_operator_distributed,
    rhs_ones,
)
from jaxamg.mpi_utils import get_partition_info

jax.config.update("jax_enable_x64", True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    # Problem size
    grid_size = 16
    n_global = grid_size * grid_size

    if rank == 0:
        print(f"Setting up MPI Poisson Optimization on {grid_size}x{grid_size} grid...")
        print(f"MPI ranks: {nranks}")
        print()

    # Ground truth
    true_skew = 5.0

    # Generate ground truth solution using single-GPU on rank 0
    if rank == 0:
        b_global = rhs_ones(n_global)
        A_true = poisson_operator(true_skew)

        x_target_global, info = jaxamg.solve(
            A_true,
            b_global,
            solver="PBICGSTAB",
            preconditioner={"solver": "JACOBI_L1"},
            tolerance=1e-8,
        )
    else:
        x_target_global = None

    # Broadcast ground truth
    x_target_global = comm.bcast(x_target_global, root=0)

    # Partition
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)
    x_target_local = jnp.array(x_target_global[row_start:row_end])
    b_local = rhs_ones(n_local)

    comm.Barrier()
    print(f"  Rank {rank}: {n_local} rows [{row_start}:{row_end})")
    comm.Barrier()

    # Configuration for solver
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "max_iters": 50,
        "tolerance": 1e-6,
    }

    # Create dummy operator for caching
    dummy_op = poisson_operator_distributed(grid_size, row_start, row_end, skew=1.0)

    # Cache MPI metadata
    mpi_cache = jaxamg.cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), dummy_op
    )

    # Cache coloring
    coloring_cache = jaxamg.cache_coloring(dummy_op, shape=(n_local, n_global))

    if rank == 0:
        print("\nStarting optimization...")

    # Define loss function
    # @jax.jit
    def loss_local(skew, b_loc, x_true_loc):
        op = poisson_operator_distributed(grid_size, row_start, row_end, skew=skew)
        A = jaxamg.with_cache(op, coloring=coloring_cache, mpi=mpi_cache)

        x_pred_loc, info = jaxamg.solve(A, b_loc)

        loss = jnp.sum((x_pred_loc - x_true_loc) ** 2) / n_global
        return loss

    # JIT gradients
    grad_fn = jax.jit(jax.grad(loss_local))

    # Optimization Loop
    skew_init = 0.0
    lr = 0.1

    if rank == 0:
        print(f"{'Epoch':<6} {'Skew':<12} {'Global Loss':<15} {'Gradient':<12}")
        print("-" * 50)

    for epoch in range(200):
        # Compute local loss and gradient
        l_loc = loss_local(skew_init, b_local, x_target_local)
        g_loc = grad_fn(skew_init, b_local, x_target_local)

        # Sync
        l_global = comm.allreduce(float(l_loc), op=MPI.SUM)
        g_global = comm.allreduce(float(g_loc), op=MPI.SUM)

        if rank == 0:
            print(f"{epoch:<6} {skew_init:<12.4f} {l_global:<15.6f} {g_global:<12.6f}")

        # Update
        skew_init -= lr * g_global

        if l_global < 1e-6:
            if rank == 0:
                print("\nConverged!")
            break

    if rank == 0:
        print(f"\nFinal skew: {skew_init:.4f}, True skew: {true_skew:.4f}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
