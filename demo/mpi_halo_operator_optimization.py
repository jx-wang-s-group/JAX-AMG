"""
Demo: MPI optimization of a Poisson operator skew parameter with a halo-form
operator (`jaxamg.halo_operator`): each rank describes only its own rows, so no
rank forms a global-length vector.

Usage:
    mpirun -n 4 python demo/mpi_halo_operator_optimization.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

import jaxamg
from jaxamg.mpi_utils import get_partition_info

jax.config.update("jax_enable_x64", True)


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    _gpus = jax.devices()
    jax.config.update("jax_default_device", _gpus[rank % len(_gpus)])

    # A g x g grid, rows split into contiguous blocks of cells (C order).
    g = 16
    n_global = g * g
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)
    cells = np.arange(row_start, row_end)

    # The 5-point stencil reaches one grid row up and down: the ghosts are the
    # cells within g of this rank's block, outside it.
    lo, hi = max(0, row_start - g), min(n_global, row_end + g)
    ghost_ids = np.array(
        [c for c in range(lo, hi) if c < row_start or c >= row_end], dtype=np.int64
    )
    n_below = int(np.sum(ghost_ids < row_start))

    if rank == 0:
        print(f"Halo-form Poisson optimization on a {g}x{g} grid, {nranks} ranks")
    print(f"  Rank {rank}: rows [{row_start}:{row_end}), {len(ghost_ids)} ghosts")

    def poisson(skew):
        """This rank's rows of -Δu + skew·(∂u/∂x + ∂u/∂y) (Dirichlet), as
        fn(x_local, x_ghost); the same stencil as jaxamg.matrices.poisson_operator."""
        ahead, behind = -1.0 - skew / 2.0, -1.0 + skew / 2.0
        col = cells % g

        def apply(x_local, x_ghost):
            # Values of cells lo .. hi-1, in order.
            ext = jnp.concatenate([x_ghost[:n_below], x_local, x_ghost[n_below:]])

            def at(c, inside):
                return jnp.where(inside, ext[np.clip(c, lo, hi - 1) - lo], 0.0)

            return (
                4.0 * x_local
                + ahead
                * (at(cells + 1, col < g - 1) + at(cells + g, cells < n_global - g))
                + behind * (at(cells - 1, col > 0) + at(cells - g, cells >= g))
            )

        return apply

    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "max_iters": 200,
        "tolerance": 1e-10,
    }

    # The target: the solution at the true skew. This eager solve also
    # discovers the operator's colouring.
    true_skew = 5.0
    b_local = jnp.ones(n_local)
    base = jaxamg.halo_operator(
        poisson(true_skew), n_local=n_local, ghost_ids=ghost_ids
    )
    mpi_cache = jaxamg.cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), base
    )
    x_target, _ = jaxamg.solve(jaxamg.with_cache(base, mpi=mpi_cache), b_local)

    def operator(skew):
        # Per-step operators reuse the base operator's colouring inside jit.
        return jaxamg.with_cache(base.with_fn(poisson(skew)), mpi=mpi_cache)

    def loss_local(skew):
        x, _ = jaxamg.solve(operator(skew), b_local)
        return jnp.sum((x - x_target) ** 2) / n_global

    value_and_grad = jax.jit(jax.value_and_grad(loss_local))

    skew, lr = 0.0, 0.1
    if rank == 0:
        print(f"\n{'Epoch':<6} {'Skew':<12} {'Global Loss':<15} {'Gradient':<12}")
        print("-" * 50)
    for epoch in range(200):
        loss, grad = value_and_grad(skew)
        # Each rank holds its rows' share of the loss and of its gradient.
        loss = comm.allreduce(float(loss), op=MPI.SUM)
        grad = comm.allreduce(float(grad), op=MPI.SUM)
        if rank == 0 and epoch % 10 == 0:
            print(f"{epoch:<6} {skew:<12.4f} {loss:<15.3e} {grad:<12.3e}")
        if loss < 1e-8:
            break
        skew -= lr * grad

    if rank == 0:
        print(f"\nFinal skew: {skew:.4f}, true skew: {true_skew:.4f}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
