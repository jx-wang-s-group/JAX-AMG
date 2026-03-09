"""
Demo: Use JAX-AMG as a preconditioner for native JAX and Lineax Krylov solvers.

This demo constructs a simple 2D Poisson problem and demonstrates how to use the `make_preconditioner` function to create an approximate inverse that can be passed to native JAX solvers like `cg` and `bicgstab`, as well as Lineax solvers.
"""

import time

import jax
import jax.numpy as jnp
import lineax as lx
from jax.scipy.sparse.linalg import bicgstab, cg

import jaxamg
from jaxamg.matrices import poisson_matrix, rhs_ones

jax.config.update("jax_enable_x64", True)


def run_jaxamg_solver(name, A, b, *, config=None, **kwargs):
    t_start = time.time()
    try:
        x, info = jaxamg.solve(A, b, config=config, **kwargs)
        elapsed = time.time() - t_start
        residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
        iterations = info.get("iterations", "?")
        print(f"{name:40s} | time={elapsed:6.3f}s residual={float(residual):.3e}")
    except Exception:
        elapsed = time.time() - t_start
        print(f"{name:40s} | time={elapsed:6.3f}s residual=nan")


def run_solver(name, solver_fn, A, b, *, M=None, tol=1e-6, maxiter=200):
    t_start = time.time()
    try:
        x, info = solver_fn(A, b, M=M, tol=tol, maxiter=maxiter)
        elapsed = time.time() - t_start
        residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
        print(f"{name:40s} | time={elapsed:6.3f}s residual={float(residual):.3e}")
    except Exception as exc:
        elapsed = time.time() - t_start
        print(f"{name:40s} | time={elapsed:6.3f}s residaul=nan")


def run_lineax_solver(name, solver, operator, b, *, preconditioner=None):
    t_start = time.time()
    kwargs = {}
    if preconditioner is not None:
        kwargs["options"] = {"preconditioner": preconditioner}
    try:
        solution = lx.linear_solve(operator, b, solver=solver, **kwargs)
        elapsed = time.time() - t_start
        x = solution.value
        residual = jnp.linalg.norm(b - operator.mv(x)) / jnp.linalg.norm(b)
        print(f"{name:40s} | time={elapsed:6.3f}s residual={float(residual):.3e}")
    except Exception as exc:
        elapsed = time.time() - t_start
        print(f"{name:40s} | time={elapsed:6.3f}s residual=nan")


def main():
    grid_size = 64
    n = grid_size**2
    tol = 1e-6
    maxiter = 200

    print(f"Setting up symmetric Poisson system with {grid_size}×{grid_size} grid...")
    A = poisson_matrix(grid_size)
    b = rhs_ones(n)
    M = jaxamg.make_preconditioner(A)

    run_jaxamg_solver("JAX-AMG standalone", A, b)
    run_solver("JAX CG", cg, A, b, tol=tol, maxiter=maxiter)
    run_solver(
        "JAX CG + JAX-AMG preconditioner", cg, A, b, M=M, tol=tol, maxiter=maxiter
    )

    operator = lx.FunctionLinearOperator(
        lambda x: A @ x,
        input_structure=jax.ShapeDtypeStruct(b.shape, b.dtype),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    preconditioner = lx.FunctionLinearOperator(
        M,
        input_structure=jax.ShapeDtypeStruct(b.shape, b.dtype),
        tags=(lx.symmetric_tag, lx.positive_semidefinite_tag),
    )
    run_lineax_solver(
        "Lineax CG",
        lx.CG(rtol=tol, atol=tol, max_steps=maxiter),
        operator,
        b,
    )
    run_lineax_solver(
        "Lineax CG + JAX-AMG preconditioner",
        lx.CG(rtol=tol, atol=tol, max_steps=maxiter),
        operator,
        b,
        preconditioner=preconditioner,
    )

    for skew in (1.0, 2.0):
        print(
            f"\nSetting up non-symmetric Poisson system (skew={skew:g}) with {grid_size}×{grid_size} grid..."
        )
        A_nonsym = poisson_matrix(grid_size, skew=skew)
        b = rhs_ones(n)
        M_nonsym = jaxamg.make_preconditioner(A_nonsym)

        run_jaxamg_solver(
            "JAX-AMG standalone",
            A_nonsym,
            b,
        )
        run_solver(
            "JAX BiCGSTAB",
            bicgstab,
            A_nonsym,
            b,
            tol=tol,
            maxiter=maxiter,
        )
        run_solver(
            "JAX BiCGSTAB + JAX-AMG preconditioner",
            bicgstab,
            A_nonsym,
            b,
            M=M_nonsym,
            tol=tol,
            maxiter=maxiter,
        )

        nonsym_operator = lx.FunctionLinearOperator(
            lambda x, A_nonsym=A_nonsym: A_nonsym @ x,
            input_structure=jax.ShapeDtypeStruct(b.shape, b.dtype),
        )
        nonsym_preconditioner = lx.FunctionLinearOperator(
            M_nonsym,
            input_structure=jax.ShapeDtypeStruct(b.shape, b.dtype),
        )

        run_lineax_solver(
            "Lineax BiCGStab",
            lx.BiCGStab(rtol=tol, atol=tol, max_steps=maxiter),
            nonsym_operator,
            b,
        )
        run_lineax_solver(
            "Lineax BiCGStab + JAX-AMG preconditioner",
            lx.BiCGStab(rtol=tol, atol=tol, max_steps=maxiter),
            nonsym_operator,
            b,
            preconditioner=nonsym_preconditioner,
        )


if __name__ == "__main__":
    main()
