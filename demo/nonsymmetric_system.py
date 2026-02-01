"""
Demo: Solve non-symmetric system.

Use CG and BiCGSTAB solvers on symmetric and non-symmetric systems,
and demonstrate that CG fails for non-symmetric systems.
"""

import numpy as np
import jax.numpy as jnp
from jaxamg import amg_solve
from jaxamg.matrices import poisson_matrix, poisson_operator, rhs_ones


def main():
    n = 8
    b = rhs_ones(n * n)

    # Create matrices and operators
    A_sym = poisson_matrix(n, skew=0.0)
    A_nonsym = poisson_matrix(n, skew=0.5)
    A_op_nosym = poisson_operator(skew=0.5)

    x_sym_cg, info_sym_cg = amg_solve(A_sym, b, solver="CG")
    print(f"Symmetric matrix (skew=0.0), Solver: CG")
    print(info_sym_cg)

    x_nonsym_cg, info_nonsym_cg = amg_solve(A_nonsym, b, solver="CG")
    print(f"\nNon-symmetric matrix (skew=0.5), Solver: CG")
    print(info_nonsym_cg)

    x_nonsym_bicg, info_nonsym_bicg = amg_solve(A_nonsym, b, solver="BICGSTAB")
    print(f"\nNon-symmetric matrix (skew=0.5), Solver: BICGSTAB")
    print(info_nonsym_bicg)

    x_nonsym_bicg_amg, info_nonsym_bicg_amg = amg_solve(
        A_nonsym, b, solver="PBICGSTAB", preconditioner={"solver": "AMG"}
    )
    print(f"\nNon-symmetric matrix (skew=0.5), Solver: PBICGSTAB + AMG preconditioner")
    print(info_nonsym_bicg_amg)

    x_nonsym_op_bicg, info_nonsym_op_bicg = amg_solve(A_op_nosym, b, solver="BICGSTAB")
    print(f"\nNon-symmetric operator (skew=0.5), Solver: BICGSTAB")
    print(info_nonsym_op_bicg)

    x_nonsym_op_gmres, info_nonsym_op_gmres = amg_solve(A_op_nosym, b, solver="GMRES")
    print(f"\nNon-symmetric operator (skew=0.5), Solver: GMRES")
    print(info_nonsym_op_gmres)


if __name__ == "__main__":
    main()
