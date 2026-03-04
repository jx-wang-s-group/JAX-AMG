"""
Demo: Solving a large random sparse linear system.
"""

import time

import jaxamg
from jaxamg.matrices import random_matrix, rhs_random


def main():
    n = 100000
    seed = 42
    print(f"Setting up a random sparse matrix of size {n} x {n}...")

    start = time.time()
    A = random_matrix(n, density=0.01, seed=seed)
    construction_time = time.time() - start

    nnz = len(A.data)
    print(f"Matrix construction time: {construction_time:.2f}s")
    print(f"Non-zeros: {nnz:,}")
    print(f"Sparsity: {100 * nnz / (n * n):.4f}%")
    print(
        f"Matrix memory: {(len(A.data) * 4 + len(A.indices) * 4 + len(A.indptr) * 4) / 1024**2:.1f} MB"
    )

    # Create RHS
    b = rhs_random(n, seed=seed)

    # Solve
    print("\nSolving...")
    start = time.time()
    x, info = jaxamg.solve(
        A,
        b,
        config={"preconditioner": {"coarse_solver": "CG"}},
        save_stats_file="stats_large_random_matrix.txt",
    )
    solve_time = time.time() - start

    print(f"Solve time: {solve_time:.2f}s")
    print(f"Status: {info['status']}")
    print(f"Iterations: {info['iterations']}")
    print(f"Residual: {info['residual']:.2e}")


if __name__ == "__main__":
    main()
