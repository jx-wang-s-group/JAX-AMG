"""
Demo: sharded optimization of a Poisson operator skew parameter with a
global-view operator (`jaxamg.global_operator`): a `shard_map` function in which
each rank exchanges only its boundary grid rows, so no rank forms a
global-length vector.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 python demo/sharded_global_operator_optimization.py
"""

import jax
import jax.numpy as jnp

# One JAX process per MPI rank; must precede any JAX array creation.
jax.distributed.initialize(cluster_detection_method="mpi4py")

import numpy as np
from jax.experimental import multihost_utils
from mpi4py import MPI

import jaxamg

jax.config.update("jax_enable_x64", True)


def main():
    comm = MPI.COMM_WORLD
    rank, nranks = comm.Get_rank(), comm.Get_size()
    mesh = jax.make_mesh((nranks,), ("rank",))

    # A g x g grid (C order); each rank owns g // nranks consecutive grid rows.
    g = 16
    if g % nranks:
        raise SystemExit(f"the grid rows ({g}) must divide evenly over the ranks")
    n_global = g * g
    rows_per_rank = g // nranks
    n_local = rows_per_rank * g
    row_start = rank * n_local

    if rank == 0:
        print(f"Global-view Poisson optimization on a {g}x{g} grid, {nranks} ranks")

    def poisson(skew):
        """-Δu + skew·(∂u/∂x + ∂u/∂y) (Dirichlet) on the global sharded vector;
        the same stencil as jaxamg.matrices.poisson_operator."""
        # Coefficients are shard_map arguments: under a derivative, JAX refuses
        # closures over values computed from a traced parameter here.
        coefficients = jnp.asarray([-1.0 - skew / 2.0, -1.0 + skew / 2.0])
        down = [(i, i + 1) for i in range(nranks - 1)]  # to the next rank
        up = [(i + 1, i) for i in range(nranks - 1)]  # to the previous rank

        def local(x, coefficients):
            ahead, behind = coefficients
            u = x.reshape(rows_per_rank, g)
            # The neighbours' boundary grid rows (zero at the domain boundary).
            above = jax.lax.ppermute(u[-1:], "rank", perm=down)
            below = jax.lax.ppermute(u[:1], "rank", perm=up)
            ext = jnp.pad(jnp.concatenate([above, u, below]), ((0, 0), (1, 1)))
            centre = ext[1:-1, 1:-1]
            y = (
                4.0 * centre
                + ahead * (ext[1:-1, 2:] + ext[2:, 1:-1])
                + behind * (ext[1:-1, :-2] + ext[:-2, 1:-1])
            )
            return y.reshape(-1)

        stencil = jax.shard_map(
            local,
            mesh=mesh,
            in_specs=(jax.P("rank"), jax.P()),
            out_specs=jax.P("rank"),
        )
        return lambda x: stencil(x, coefficients)

    # This rank's row pattern (global columns): the 5-point stencil.
    cells = row_start + np.arange(n_local)
    i, j = np.divmod(cells, g)
    neighbours = [
        (cells, np.ones(n_local, bool)),
        (cells - 1, j > 0),
        (cells + 1, j < g - 1),
        (cells - g, i > 0),
        (cells + g, i < g - 1),
    ]
    columns = [
        np.sort(np.array([c[k] for c, ok in neighbours if ok[k]]))
        for k in range(n_local)
    ]
    indices = np.concatenate(columns)
    indptr = np.concatenate(([0], np.cumsum([len(c) for c in columns])))

    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "max_iters": 50,
        "tolerance": 1e-6,
    }

    # The operator at the true skew; its colouring is computed once, here.
    true_skew = 5.0
    op = jaxamg.global_operator(
        poisson(true_skew), indices, indptr, comm=comm, mesh=mesh
    )
    b = jaxamg.make_sharded_vector(
        np.ones(n_local), comm=comm, mesh=mesh, global_size=n_global
    )
    A = jaxamg.make_sharded_matrix(op, b, comm=comm, mesh=mesh)
    solver = jaxamg.make_sharded_solver(A, b, config=config)
    x_target = solver(b)[0]  # the target: the solution at the true skew

    # Sharded arrays must be jit arguments, not closures.
    def loss(skew, b, x_target):
        # New parameters, same structure: materialized inside the transform.
        x = solver(b, A=op.with_fn(poisson(skew)))[0]
        return jnp.sum((x - x_target) ** 2) / n_global

    value_and_grad = jax.jit(jax.value_and_grad(loss))

    skew, lr = 0.0, 0.1
    if rank == 0:
        print(f"\n{'Epoch':<6} {'Skew':<12} {'Global Loss':<15} {'Gradient':<12}")
        print("-" * 50)
    # Outer transforms over sharded arrays need the mesh in context.
    with jax.set_mesh(mesh):
        for epoch in range(200):
            # The loss is a global reduction: every rank gets the same value.
            loss_value, grad = map(float, value_and_grad(skew, b, x_target))
            if rank == 0 and epoch % 10 == 0:
                print(f"{epoch:<6} {skew:<12.4f} {loss_value:<15.3e} {grad:<12.3e}")
            if loss_value < 1e-8:
                break
            skew -= lr * grad

    if rank == 0:
        print(f"\nFinal skew: {skew:.4f}, true skew: {true_skew:.4f}")

    multihost_utils.sync_global_devices("finalize")
    jaxamg.finalize()
    multihost_utils.sync_global_devices("shutdown")
    jax.distributed.shutdown()


if __name__ == "__main__":
    main()
