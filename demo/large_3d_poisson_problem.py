"""
Demo: Solving a large 3D Poisson system.

This demonstrates solver's ability to efficiently solve very large sparse linear systems.
"""

import time

import jaxamg
from jaxamg.matrices import poisson3d_matrix, rhs_ones


def main():
    n = 256  # Grid size in each dimension
    n3 = n**3
    print(f"Setting up a 3D Poisson system with {n} x {n} x {n} grid")
    print(f"Matrix size: {n3:,} x {n3:,}")

    # Construct matrix
    print("Constructing 3D Poisson matrix...")
    start = time.time()
    A = poisson3d_matrix(n)
    construction_time = time.time() - start

    nnz = len(A.data)
    print(f"Matrix construction time: {construction_time:.2f}s")
    print(f"Non-zeros: {nnz:,}")
    print(f"Sparsity: {100 * nnz / (n3 * n3):.4f}%")
    print(
        f"Memory (matrix): {(len(A.data) * 4 + len(A.indices) * 4 + len(A.indptr) * 4) / 1024**2:.1f} MB"
    )

    # Create RHS
    b = rhs_ones(n3)

    # Solve
    print("\nSolving...")
    start = time.time()
    x, info = jaxamg.solve(A, b, solver="CG", max_iters=2000, tolerance=1e-6)
    solve_time = time.time() - start

    print(f"Solve time: {solve_time:.2f}s")
    print(f"Status: {info['status']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Final residual: {info['residual']:.2e}")


if __name__ == "__main__":
    main()
