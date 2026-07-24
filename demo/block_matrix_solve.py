"""
Demo: Block matrices (block_dim > 1)

A coupled multi-component PDE system has one 2D matrix whose nonzeros come in
dense k x k tiles: each mesh node carries k coupled unknowns, interleaved
node-major (row i*k + c is component c of node i). Passing block_dim=k lets
AmgX see that structure -- block-aware AMG aggregates nodes instead of scalar
rows, and BLOCK_JACOBI inverts true k x k diagonal tiles -- while the matrix
and vectors keep their ordinary scalar CSR/vector form.

Built here as kron(Poisson, M): a 2D Poisson problem where every node carries
two unknowns coupled through the 2x2 matrix M.
"""

import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse

import jaxamg
from jaxamg.matrices import poisson_matrix
from jaxamg.utils import to_scipy


def coupled_system(grid_size, M):
    """kron(Poisson, M): each grid node carries len(M) coupled components."""
    P = to_scipy(poisson_matrix(grid_size))
    A_sp = scipy.sparse.kron(P, M.astype(np.float32)).tocsr()
    A = jsp.BCSR(
        (
            jnp.asarray(A_sp.data),
            jnp.asarray(A_sp.indices),
            jnp.asarray(A_sp.indptr),
        ),
        shape=A_sp.shape,
    )
    return A, jnp.ones(A_sp.shape[0])


def main():
    grid_size = 64
    M = np.array([[2.0, 1.0], [1.0, 2.0]])
    A, b = coupled_system(grid_size, M)
    n = A.shape[0]
    print(f"Coupled 2-component system: {n}x{n}, 2x2 blocks\n")

    # Block-aware solve: aggregation AMG + block-Jacobi smoothing on 2x2 tiles.
    x, info = jaxamg.solve(A, b, block_dim=2)
    print(f"block_dim=2: {info['iterations']:>4} iterations, {info['status']}")

    # Same matrix treated as scalar: classical AMG never sees the coupling.
    x1, info1 = jaxamg.solve(A, b)
    print(f"block_dim=1: {info1['iterations']:>4} iterations, {info1['status']}")

    # A strongly coupled nonsymmetric system makes the contrast starker:
    # point-Jacobi preconditioning stalls, true 2x2 block-Jacobi nails it.
    Mn = np.array([[3.0, 1.0], [0.5, 2.0]])
    P = to_scipy(poisson_matrix(grid_size, skew=0.7))
    A_sp = scipy.sparse.kron(P, Mn.astype(np.float32)).tocsr()
    An = jsp.BCSR(
        (
            jnp.asarray(A_sp.data),
            jnp.asarray(A_sp.indices),
            jnp.asarray(A_sp.indptr),
        ),
        shape=A_sp.shape,
    )
    bn = jnp.ones(A_sp.shape[0])
    config = {"solver": "FGMRES", "preconditioner": {"solver": "BLOCK_JACOBI"}}
    print("\nStrongly coupled nonsymmetric system, FGMRES + BLOCK_JACOBI:")
    for bs in (2, 1):
        _, info_n = jaxamg.solve(An, bn, block_dim=bs, config=config, max_iters=300)
        print(
            f"block_dim={bs}: {info_n['iterations']:>4} iterations, {info_n['status']}"
        )


if __name__ == "__main__":
    main()
