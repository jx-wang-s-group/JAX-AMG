"""
Demo: Solving 2D Poisson equation.

This example demonstrates solving a 2D Poisson equation on a regular grid
using the 5-point finite difference stencil.
"""

import jax.numpy as jnp
from jaxamg import amgx_solve
from jaxamg.matrices import poisson_matrix, rhs_ones


def main():
    # Setup a Poisson problem on a 32x32 grid
    grid_size = 32
    print(f"\nSetting up Poisson problem on {grid_size}×{grid_size} grid...")
    A = poisson_matrix(grid_size)
    n = grid_size**2
    print(f"Matrix size: {n}×{n}")

    # Right-hand side: constant vector
    b = rhs_ones(n)

    # Solve Ax = b
    print("Solving...")
    x = amgx_solve(A, b)

    # Compute residual: ||b - Ax|| / ||b||
    residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)

    # Display results
    print(f"Solution norm: {jnp.linalg.norm(x):.6f}")
    print(f"Relative residual: {residual:.2e}")


if __name__ == "__main__":
    main()
