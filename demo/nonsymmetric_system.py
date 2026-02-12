"""
Demo: Solve non-symmetric system.

Use CG and BiCGSTAB solvers on symmetric and non-symmetric systems,
and demonstrate that CG fails for non-symmetric systems.
"""

import jaxamg
from jaxamg.matrices import poisson_matrix, poisson_operator, rhs_ones


def main():
    n = 8
    b = rhs_ones(n * n)

    # Create matrices and operators
    A_sym = poisson_matrix(n, skew=0.0)
    A_nonsym = poisson_matrix(n, skew=0.5)
    A_op_nosym = poisson_operator(skew=0.5)

    _, info_sym_cg = jaxamg.solve(A_sym, b, solver="CG")
    print("Symmetric matrix (skew=0.0), Solver: CG")
    print(info_sym_cg)

    _, info_nonsym_cg = jaxamg.solve(A_nonsym, b, solver="CG")
    print("\nNon-symmetric matrix (skew=0.5), Solver: CG")
    print(info_nonsym_cg)

    _, info_nonsym_bicg = jaxamg.solve(A_nonsym, b, solver="BICGSTAB")
    print("\nNon-symmetric matrix (skew=0.5), Solver: BICGSTAB")
    print(info_nonsym_bicg)

    _, info_nonsym_bicg_amg = jaxamg.solve(
        A_nonsym, b, solver="PBICGSTAB", preconditioner={"solver": "AMG"}
    )
    print("\nNon-symmetric matrix (skew=0.5), Solver: PBICGSTAB + AMG preconditioner")
    print(info_nonsym_bicg_amg)

    _, info_nonsym_op_bicg = jaxamg.solve(A_op_nosym, b, solver="BICGSTAB")
    print("\nNon-symmetric operator (skew=0.5), Solver: BICGSTAB")
    print(info_nonsym_op_bicg)

    _, info_nonsym_op_gmres = jaxamg.solve(A_op_nosym, b, solver="GMRES")
    print("\nNon-symmetric operator (skew=0.5), Solver: GMRES")
    print(info_nonsym_op_gmres)


if __name__ == "__main__":
    main()
