"""
Demo: JAX-sharded optimization of a Poisson operator skew parameter.

Demonstrates end-to-end JIT compilation and differentiation of a custom JAX operator
with the sharding interface. The counterpart of demo/mpi_poisson_operator_optimization.py:
the RHS and solution are JAX global arrays, so the loss is a global reduction and
no MPI call is needed.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 python demo/sharded_poisson_operator_optimization.py
"""

import jax
import jax.numpy as jnp

# One JAX process per MPI rank; must precede any JAX array creation.
jax.distributed.initialize(cluster_detection_method="mpi4py")

from jax.experimental import multihost_utils

import jaxamg
from jaxamg.matrices import poisson_operator, rhs_ones
from jaxamg.mpi_utils import get_partition_info, partition_operator

jax.config.update("jax_enable_x64", True)


def main():
    rank = jax.process_index()
    nranks = jax.process_count()

    # Problem size
    grid_size = 16
    n_global = grid_size * grid_size

    if rank == 0:
        print(
            f"Setting up sharded Poisson Optimization on {grid_size}x{grid_size} grid..."
        )
        print(f"Processes: {nranks}")
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
        x_target_global = jnp.zeros(n_global)

    # Broadcast ground truth
    x_target_global = multihost_utils.broadcast_one_to_all(x_target_global)

    # Partition
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)
    x_target_local = jnp.array(x_target_global[row_start:row_end])
    b_local = rhs_ones(n_local)

    multihost_utils.sync_global_devices("partition")
    print(f"  Rank {rank}: {n_local} rows [{row_start}:{row_end})")
    multihost_utils.sync_global_devices("partition")

    # Configuration for solver
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "max_iters": 50,
        "tolerance": 1e-6,
    }

    # Create local dummy operator for caching
    dummy_op, _, _ = partition_operator(
        poisson_operator(skew=1.0), n_global, rank, nranks
    )

    # Cache coloring
    coloring_cache = jaxamg.cache_coloring(dummy_op, shape=(n_local, n_global))

    # Global arrays and the sharded solver (fixes the sparsity pattern)
    b = jaxamg.make_sharded_vector(b_local, global_size=n_global)
    x_target = jaxamg.make_sharded_vector(x_target_local, global_size=n_global)
    A = jaxamg.make_sharded_matrix(
        jaxamg.with_cache(dummy_op, coloring=coloring_cache), b
    )
    solver = jaxamg.make_sharded_solver(A, b, config=config)

    if rank == 0:
        print("\nStarting optimization...")

    # Define loss function
    def loss(skew, b, x_true):
        op, _, _ = partition_operator(
            poisson_operator(skew=skew), n_global, rank, nranks
        )

        x_pred, info = solver(b, A=op)

        loss = jnp.sum((x_pred - x_true) ** 2) / n_global
        return loss

    # JIT loss and gradient
    value_and_grad_fn = jax.jit(jax.value_and_grad(loss))

    # Optimization Loop
    skew_init = 0.0
    lr = 0.1

    if rank == 0:
        print(f"{'Epoch':<6} {'Skew':<12} {'Global Loss':<15} {'Gradient':<12}")
        print("-" * 50)

    # Outer transforms over sharded arrays need the mesh in context
    with jax.set_mesh(A.mesh):
        for epoch in range(200):
            # Force solver rebuild
            if epoch % 20 == 0:
                jaxamg.clear_solver_cache()

            # Compute global loss and gradient
            l_global, g_global = value_and_grad_fn(skew_init, b, x_target)
            l_global = float(l_global)
            g_global = float(g_global)

            if rank == 0:
                print(
                    f"{epoch:<6} {skew_init:<12.4f} {l_global:<15.6f} {g_global:<12.6f}"
                )

            # Update
            skew_init -= lr * g_global

            if l_global < 1e-6:
                if rank == 0:
                    print("\nConverged!")
                break

    if rank == 0:
        print(f"\nFinal skew: {skew_init:.4f}, True skew: {true_skew:.4f}")

    multihost_utils.sync_global_devices("finalize")
    jaxamg.finalize()
    multihost_utils.sync_global_devices("shutdown")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
