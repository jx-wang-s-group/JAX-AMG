"""
Deo: MPI-distributed automatic differentiation.

Demonstrates that gradients computed via MPI-distributed solver match
the single-GPU reference implementation.

Usage:
    mpirun -n 3 python demo/mpi_autodiff.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

import jaxamg
from jaxamg.matrices import (
    rhs_random,
    tridiagonal_matrix,
    tridiagonal_matrix_distributed,
)
from jaxamg.mpi_utils import gather_vector

jax.config.update("jax_enable_x64", True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    _gpus = jax.devices()
    jax.config.update("jax_default_device", _gpus[rank % len(_gpus)])

    n_global = 1024

    if rank == 0:
        print(
            f"Setting up MPI automatic differentiation for a tridiagonal system of size {n_global}..."
        )
        print(f"MPI ranks: {nranks}")
        print(f"\nPartitioning matrix across {nranks} ranks...")

    A_local, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, dtype=jnp.float64
    )
    n_local = row_end - row_start

    comm.Barrier()

    print(
        f"  Rank {rank}: {n_local} rows [{row_start}:{row_end}), {len(A_local.data)} non-zeros"
    )
    comm.Barrier()

    b_local = rhs_random(n_local, seed=rank)

    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "MULTICOLOR_DILU"},
        "max_iters": 100,
        "monitor_residual": 1,
        "tolerance": 1e-6,
    }

    # Pre-cache MPI metadata
    mpi_cache = jaxamg.cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), A_local
    )

    # Attach MPI cache to matrix
    A_local = jaxamg.with_cache(A_local, mpi=mpi_cache, is_symmetric=True)

    def loss_fn(b_loc):
        """Loss function using cached MPI metadata - JIT will be applied to grad."""
        x_loc, _ = jaxamg.solve(A_local, b_loc)
        return jnp.sum(x_loc**2)

    # JIT-compile the gradient computation
    loss_and_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    if rank == 0:
        print("\nComputing loss and gradient...")

    loss_mpi, grad_mpi = loss_and_grad_fn(b_local)

    print(
        f"  Rank {rank}: Loss = {loss_mpi:.4e}, Grad norm = {jnp.linalg.norm(grad_mpi):.4e}"
    )

    b_global = gather_vector(b_local, comm)
    grad_global = gather_vector(grad_mpi, comm)
    loss_total = comm.reduce(float(loss_mpi), op=MPI.SUM, root=0)

    if rank == 0:
        print("\nValidating against single-GPU result...")

        A_global = tridiagonal_matrix(n_global, dtype=jnp.float64)

        def loss_fn_ref(b_vec):
            x_ref, _ = jaxamg.solve(A_global, b_vec, config=config)
            return jnp.sum(x_ref**2)

        loss_ref, grad_ref = jax.value_and_grad(loss_fn_ref)(b_global)
        loss_error = abs(loss_total - loss_ref) / (abs(loss_ref))
        grad_error = np.linalg.norm(grad_global - grad_ref) / (np.linalg.norm(grad_ref))

        print(f"  Loss (MPI): {loss_total:.4e}")
        print(f"  Loss (single-GPU): {loss_ref:.4e}")
        print(f"  Loss relative error: {loss_error:.2e}")
        print(f"  Gradient relative error: {grad_error:.2e}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
