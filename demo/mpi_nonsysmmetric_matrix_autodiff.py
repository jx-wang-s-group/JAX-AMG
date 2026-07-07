"""
Demo: MPI-distributed automatic differentiation of torso3 (a structurally non-symmetric system).

Usage:
    mpirun -n 2 python demo/mpi_nonsysmmetric_matrix_autodiff.py
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

import jaxamg
from jaxamg.matrices import download_suitesparse_matrix
from jaxamg.mpi_utils import gather_vector, partition_csr_matrix


def make_rhs(row_start, row_end, n_global):
    local_idx = jnp.arange(row_start, row_end, dtype=np.float64)
    full_idx = jnp.arange(n_global, dtype=np.float64)
    rhs_func = lambda x: jnp.sin(0.001 * (x + 1.0)) + 0.25 * jnp.cos(
        0.00013 * (x + 1.0)
    )
    local = rhs_func(local_idx)
    full = rhs_func(full_idx)
    return local / jnp.linalg.norm(full) * jnp.sqrt(n_global)


def main():
    jax.config.update("jax_enable_x64", True)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    gpus = jax.devices()
    dev = gpus[rank % len(gpus)]
    jax.config.update("jax_default_device", dev)

    if rank == 0:
        print("Setting up MPI automatic differentiation for torso3 matrix...")
        print(f"MPI ranks: {nranks}")
        print("Downloading/loading torso3 from SuiteSparse...")
        download_suitesparse_matrix("Norris/torso3")

    # Solver configuration
    config = {
        "config_version": 2,
        "solver": {
            "solver": "FGMRES",
            "gmres_n_restart": 100,
            "max_iters": 3000,
            "convergence": "RELATIVE_INI",
            "tolerance": 1e-14,
            "norm": "L2",
            "monitor_residual": 1,
            "communicator": "MPI",
            "preconditioner": {
                "solver": "AMG",
                "algorithm": "CLASSICAL",
                "selector": "PMIS",
                "interpolator": "D2",
                "strength_threshold": 0.5,
                "max_levels": 100,
                "cycle": "V",
                "presweeps": 2,
                "postsweeps": 3,
                "max_iters": 1,
                "coarse_solver": "DENSE_LU_SOLVER",
                "dense_lu_num_rows": 64,
                "min_coarse_rows": 64,
                "smoother": {"solver": "MULTICOLOR_DILU"},
            },
            "store_res_history": 1,
        },
    }

    A_global = download_suitesparse_matrix("Norris/torso3")
    n = A_global.shape[0]

    A_local, row_start, row_end = partition_csr_matrix(A_global, rank, nranks)
    b = make_rhs(row_start, row_end, n)

    # Pre-cache MPI metadata
    mpi_cache = jaxamg.cache_mpi_metadata(
        config,
        comm,
        n,
        (row_start, row_end),
        A_local,
        is_symmetric=False,
    )

    # Attach MPI cache to matrix
    A_cached = jaxamg.with_cache(A_local, mpi=mpi_cache, is_symmetric=False)

    def loss_fn(rhs):
        x, _ = jaxamg.solve(A_cached, rhs)
        return 0.5 * jnp.sum(x**2)

    # JIT-compile the gradient computation
    grad_fn = jax.jit(jax.grad(loss_fn))

    if rank == 0:
        print("\nComputing RHS gradient...")

    comm.Barrier()
    t0 = time.time()
    grad_local = grad_fn(b)
    grad_local.block_until_ready()
    gradient_time = comm.allreduce(time.time() - t0, op=MPI.MAX)

    comm.Barrier()
    t1 = time.time()
    x_local, info = jaxamg.solve(A_cached, b)
    x_local.block_until_ready()
    solve_time = comm.allreduce(time.time() - t1, op=MPI.MAX)

    x_global = gather_vector(x_local, comm, root=0)
    g_global = gather_vector(grad_local, comm, root=0)

    if rank == 0:
        print("\nValidating forward and adjoint residuals...")
        b_global = make_rhs(0, n, n)
        forward_residual = np.linalg.norm(
            b_global - A_global @ x_global
        ) / np.linalg.norm(b_global)
        adjoint_residual = np.linalg.norm(
            A_global.T @ g_global - x_global
        ) / np.linalg.norm(x_global)

        print(f"Forward time: {solve_time:.3f}s")
        print(f"Forward iterations: {int(info['iterations'])}")
        print(f"Forward residual: {forward_residual:.3e}")
        print(f"Gradient time: {gradient_time:.3f}s")
        print(f"Adjoint residual: {adjoint_residual:.3e}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
