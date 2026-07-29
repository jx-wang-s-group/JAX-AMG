"""Hybrid JAX-sharded and MPI-distributed Poisson solve with autodiff.

All GPUs on a node must remain visible to every local MPI process. JAX selects
one distinct GPU per process through ``local_device_ids``.

Single-node usage:
    CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 python demo/sharded_poisson_problem.py
"""

import jax
import jax.numpy as jnp
import numpy as np

# Initialize before importing JAX-AMG modules that may initialize XLA.
jax.distributed.initialize(cluster_detection_method="mpi4py")

from jax.experimental import multihost_utils

import jaxamg
from jaxamg.matrices import poisson_matrix_distributed


def main() -> None:

    jax.config.update("jax_logging_level", "ERROR")

    rank = jax.process_index()
    nranks = jax.process_count()
    grid_size = 32
    n_global = grid_size**2

    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )
    b_local = np.ones(row_end - row_start)

    # Create a mesh and sharding for the distributed vector
    mesh = jax.make_mesh((nranks,), ("rank",))
    b = jaxamg.make_sharded_vector(
        b_local,
        mesh=mesh,
        global_size=n_global,
    )

    # Create a sharded solver
    solver = jaxamg.make_sharded_solver(
        A_local,
        b,
        mesh=mesh,
        config={
            "solver": "CG",
            "preconditioner": {"solver": "JACOBI_L1"},
            "communicator": "MPI_DIRECT",
        },
    )

    x, info = solver(b)
    x.block_until_ready()

    def loss(A_data, rhs):
        solution, _ = solver(rhs, A_data=A_data)
        return jnp.sum(solution**2)

    with jax.set_mesh(mesh):
        grad_A_data, grad_b = jax.grad(loss, argnums=(0, 1))(solver.A_data, b)
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()
    grad_A_local = solver.local_matrix_gradient(grad_A_data)

    # Extract the unpadded portion owned by this process.
    x_local = np.asarray(solver.local_vector(x))
    grad_b_local = np.asarray(solver.local_vector(grad_b))
    local_status = np.asarray(info["status"].addressable_shards[0].data)
    print(
        f"rank {rank}: solution norm={np.linalg.norm(x_local):.6e}, "
        f"RHS gradient norm={np.linalg.norm(grad_b_local):.6e}, "
        f"matrix gradient norm={np.linalg.norm(grad_A_local.data):.6e}, "
        f"status={local_status.item()}",
        flush=True,
    )

    multihost_utils.sync_global_devices("before jaxamg finalize")
    jaxamg.finalize()
    multihost_utils.sync_global_devices("before jax distributed shutdown")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
