"""
Demo: Print solver information
"""

import jaxamg
from jaxamg.matrices import rhs_ones, tridiagonal_matrix


def main():
    n = 500
    b = rhs_ones(n)

    # Ill-conditioned matrix
    print(f"Solving an ill-conditioned matrix of size {n}x{n}")
    config = {"max_iters": 50, "tolerance": 1e-5}
    A = tridiagonal_matrix(n, diagonal_value=2.0)
    _, info = jaxamg.solve(A, b, config=config)
    print(info)

    # Well-conditioned matrix
    print(f"\nSolving a well-conditioned matrix of size {n}x{n}")
    A = tridiagonal_matrix(n, diagonal_value=4.0)
    _, info = jaxamg.solve(A, b, config=config)
    print(info)

    # Well-conditioned matrix with max_iters=1
    print(f"\nSolving a well-conditioned matrix of size {n}x{n} with 1 iteration")
    config = {"max_iters": 1, "tolerance": 1e-5}
    _, info = jaxamg.solve(A, b, config=config)
    print(info)


if __name__ == "__main__":
    main()
