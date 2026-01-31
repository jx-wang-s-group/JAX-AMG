"""
Demo: Solve linear system with linear operator.

This example demonstrates solving linear systems (tridiagonal and Poisson) using linear operators.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxamg import amg_solve
from jaxamg.matrices import (
    tridiagonal_matrix,
    tridiagonal_operator,
    poisson_matrix,
    poisson_operator,
    rhs_ones,
)


def main():

    n = 32
    print(f"Setting up a tridiagonal system of size {n}...")
    b = rhs_ones(n)

    ## Tridigiagonal operator
    print(f"Solving tridiagonal system with operator...")
    A_tri_op = tridiagonal_operator()
    x_tri_op, _ = amg_solve(A_tri_op, b)

    # Tridiagonal system as CSR matrix
    print(f"Solving tridiagonal system with CSR matrix...")
    A_tri_csr = tridiagonal_matrix(n)
    x_tri_csr, _ = amg_solve(A_tri_csr, b)

    # Display results
    print(f"Operator solution (first 5 entries): {x_tri_op[:5]}")
    print(f"Matrix solution (first 5 entries): {x_tri_csr[:5]}")

    n_grid = 10
    print(f"\nSetting up a Poisson system of size {n_grid}x{n_grid}...")
    b = rhs_ones(n_grid**2)

    ## Poisson operator
    print(f"Solving Poisson system with operator...")
    A_poi_op = poisson_operator()
    x_poi_op, _ = amg_solve(A_poi_op, b)

    # Poisson system as CSR matrix
    print(f"Solving Poisson system with CSR matrix...")
    A_poi_csr = poisson_matrix(n_grid)
    x_poi_csr, _ = amg_solve(A_poi_csr, b)

    # Display results
    print(f"Operator solution (first 5 entries): {x_poi_op[:5]}")
    print(f"Matrix solution (first 5 entries): {x_poi_csr[:5]}")


if __name__ == "__main__":
    main()
