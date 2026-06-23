"""
Demo: MPI-distributed optimization of a large matrix-free 3D Poisson operator.

A 3D Poisson operator with a learnable diagonal-shift `theta`, run as a large
200^3 (8M-unknown) problem across 4 MPI ranks: the library detects the operator's
sparsity and coloring once, then we differentiate through the distributed solve
in an optimization loop.

Usage:
    mpirun -n 4 python demo/mpi_large_3d_poisson_operator_optimization.py
"""

import jax
import jax.numpy as jnp
from mpi4py import MPI

import jaxamg
from jaxamg.matrices import poisson3d_operator, rhs_ones
from jaxamg.mpi_utils import (
    get_partition_info,
    partition_operator,
)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    gpus = jax.devices()
    jax.config.update("jax_default_device", gpus[rank % len(gpus)])

    grid = 200
    n_global = grid**3
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)

    b_local = rhs_ones(n_local)
    target_local = jnp.full(n_local, 0.2)  # synthetic per-rank target

    if rank == 0:
        print("Matrix-free 3D Poisson operator optimization")
        print(f"Grid {grid}^3 = {n_global / 1e6:.1f}M unknowns, MPI ranks: {nranks}\n")
    comm.Barrier()
    print(f"  rank {rank} -> {gpus[rank % len(gpus)]}: {n_local / 1e6:.1f}M local rows")
    comm.Barrier()

    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "AMG"},
        "communicator": "MPI_DIRECT",
        "tolerance": 1e-6,
    }

    # 3D Poisson operator with a learnable diagonal-shift parameter theta.
    def make_operator(theta):
        base = poisson3d_operator(robin=2.0)
        return lambda u: base(u) + theta * u

    # Detect coloring + MPI metadata once; the pattern is the same for every theta
    # (a diagonal shift), so it is reused across the loop.
    dummy_op, _, _ = partition_operator(make_operator(1.0), n_global, rank, nranks)
    coloring_cache = jaxamg.cache_coloring(dummy_op, shape=(n_local, n_global))
    mpi_cache = jaxamg.cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), dummy_op
    )
    if rank == 0:
        print(f"  detected operator: {coloring_cache[3]} colors\n")

    # Per-rank loss over this rank's rows; the scalar loss and gradient are
    # reduced across ranks below.
    def loss_local(theta, b_loc):
        op, _, _ = partition_operator(make_operator(theta), n_global, rank, nranks)
        A = jaxamg.with_cache(op, coloring=coloring_cache, mpi=mpi_cache)
        x_loc, _ = jaxamg.solve(A, b_loc)
        return jnp.sum((x_loc - target_local) ** 2) / n_global

    loss_and_grad = jax.jit(jax.value_and_grad(loss_local))

    theta = 2.0
    lr = 5.0
    if rank == 0:
        print(f"{'epoch':<6}{'theta':<12}{'loss':<18}{'gradient':<14}")
        print("-" * 50)

    for epoch in range(100):
        loss_loc, grad_loc = loss_and_grad(theta, b_local)
        loss = comm.allreduce(float(loss_loc), op=MPI.SUM)
        grad_global = comm.allreduce(float(grad_loc), op=MPI.SUM)
        if rank == 0:
            print(f"{epoch:<6}{theta:<12.4f}{loss:<18.10f}{grad_global:<14.6f}")
        theta -= lr * grad_global

        if loss < 1e-3:
            if rank == 0:
                print(f"\nConverged at epoch {epoch} with loss {loss:.6e}")
            break

    if rank == 0:
        print(f"\nDone. Final theta: {theta:.4f}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
