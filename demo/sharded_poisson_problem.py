"""Hybrid JAX-sharded and MPI-distributed Poisson solve with autodiff.

All GPUs on a node must remain visible to every local MPI process. JAX selects
one distinct GPU per process through ``local_device_ids``.

Single-node usage:
    CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 python demo/sharded_poisson_problem.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

# Initialize before importing JAX-AMG modules that may initialize XLA.
jax.distributed.initialize(cluster_detection_method="mpi4py")

import jaxamg
from jaxamg.matrices import poisson_matrix_distributed


def main() -> None:

    jax.config.update("jax_logging_level", "ERROR")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    grid_size = 32
    n_global = grid_size**2

    # NamedSharding requires equal shard sizes
    if n_global % nranks:
        raise ValueError(
            "JAX row sharding requires grid_size**2 to be divisible by the "
            "number of ranks"
        )

    A_local, _, _ = poisson_matrix_distributed(grid_size, grid_size, rank, nranks)
    b_local = np.ones(n_global // nranks)

    # Create a mesh and sharding for the distributed vector
    mesh = jax.make_mesh((nranks,), ("rank",))
    sharding = jax.NamedSharding(mesh, jax.P("rank"))
    b = jax.make_array_from_process_local_data(
        sharding, b_local, global_shape=(n_global,)
    )

    # Create a sharded solver
    solver = jaxamg.make_sharded_solver(
        A_local,
        b,
        comm=comm,
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

    # Multi-host global arrays cannot be converted directly to NumPy. Inspect
    # the one addressable shard owned by this process instead.
    x_local = np.asarray(x.addressable_shards[0].data)
    grad_b_local = np.asarray(grad_b.addressable_shards[0].data)
    local_status = np.asarray(info["status"].addressable_shards[0].data)
    print(
        f"rank {rank}: solution norm={np.linalg.norm(x_local):.6e}, "
        f"RHS gradient norm={np.linalg.norm(grad_b_local):.6e}, "
        f"matrix gradient norm={np.linalg.norm(grad_A_local.data):.6e}, "
        f"status={local_status.item()}",
        flush=True,
    )

    comm.Barrier()
    jaxamg.finalize()
    comm.Barrier()
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
