"""Demo: Solve a tridiagonal system.

This example demonstrates solving a large 1D tridiagonal system,
which arises from discretizing the 1D Laplacian operator.
"""

import jax.numpy as jnp
from jaxamg import amgx_solve
from jaxamg.matrices import tridiagonal_matrix, rhs_ones


def main():

    # Setup: Large 1D Laplacian (tridiagonal matrix)
    n = 1024  # System size
    print(f"Setting up {n}×{n} tridiagonal system...")
    A = tridiagonal_matrix(n, diagonal_value=4.0)  # Better conditioned

    # Right-hand side: constant vector
    b = rhs_ones(n)

    # Solve Ax = b
    print("Solving...")
    x = amgx_solve(A, b)

    # Compute residual: ||b - Ax|| / ||b||
    residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)

    # Display results
    print("Solution should be symmetric:")
    print(f"First 5 entries: {x[:5]}")
    print(f"Last 5 entries: {x[-5:]}")
    print(f"Relative residual: {residual:.2e}")


if __name__ == "__main__":
    main()
