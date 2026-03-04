"""
Demo: Solving 2D Poisson equation.

This example demonstrates solving a 2D Poisson equation on a regular grid
using the 5-point finite difference stencil.
"""

import jax
import jax.numpy as jnp

import jaxamg
from jaxamg.matrices import poisson_matrix, rhs_linear, rhs_ones, rhs_random


def main():
    # Setup a Poisson problem on a 32x32 grid
    grid_size = 32
    print(f"Setting up Poisson problem on {grid_size}×{grid_size} grid...")
    A = poisson_matrix(grid_size)
    n = grid_size**2
    print(f"Matrix size: {n}×{n}")

    # Right-hand sides
    b_ones = rhs_ones(n)
    b_linear = rhs_linear(n)
    b_random = rhs_random(n)

    b_batched = jnp.stack([b_ones, b_linear, b_random])
    names = ["Ones", "Linear", "Random"]

    print(f"Batched RHS shape: {b_batched.shape}")

    # Solve Ax = b over the batch
    print(f"Solving {len(names)} right-hand sides using jax.vmap...")

    def solve_fn(matrix, rhs):
        return jaxamg.solve(matrix, rhs, solver="CG")

    vmap_solve = jax.vmap(solve_fn, in_axes=(None, 0))
    x_batched, infos = vmap_solve(A, b_batched)

    # Display results for each item in the batch
    for i in range(len(names)):
        x_i = x_batched[i]
        b_i = b_batched[i]
        residual = jnp.linalg.norm(b_i - A @ x_i) / jnp.linalg.norm(b_i)

        print(f"RHS: {names[i]}")
        print(f"  Status: {infos['status'][i]}")
        print(f"  Solution norm: {jnp.linalg.norm(x_i):.6f}")
        print(f"  Relative residual: {residual:.2e}\n")


if __name__ == "__main__":
    main()
