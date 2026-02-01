"""
Demo: Print solver information
"""

import jax.numpy as jnp
from jaxamg.jaxamg import amg_solve, AMGXStatus
from jaxamg.matrices import tridiagonal_matrix, rhs_ones


def main():
    n = 500
    b = rhs_ones(n)

    # Ill-conditioned matrix
    print(f"Sovling a ill-conditioned matrix of size {n}x{n}")
    config = {"max_iters": 50, "tolerance": 1e-5}
    A = tridiagonal_matrix(n, diagonal_value=2.0)
    x_stats, info = amg_solve(A, b, config=config)
    print(info)

    # Well-conditioned matrix
    print(f"\nSovling a well-conditioned matrix of size {n}x{n}")
    A = tridiagonal_matrix(n, diagonal_value=4.0)
    x_stats, info = amg_solve(A, b, config=config)
    print(info)

    # Well-conditioned matrix with max_iters=1
    print(f"\nSovling a well-conditioned matrix of size {n}x{n} with 1 iteration")
    config = {"max_iters": 1, "tolerance": 1e-5}
    x_stats, info = amg_solve(A, b, config=config)
    print(info)


if __name__ == "__main__":
    main()
