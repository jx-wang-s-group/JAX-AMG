"""
Demo: Compare performance of JAX-AMG and JAX native sparse solvers.

This demo benchmarks the CG and BiCGSTAB solvers implemented in JAX-AMG against
the native JAXsparse implementations on a large tridiagonal system, evaluating
both forward solves and backward (gradient) computations.
"""

import time

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import bicgstab, cg

import jaxamg
from jaxamg.matrices import rhs_random, tridiagonal_matrix


def main():
    n = 10000000
    max_iters = 100
    tolerance = 1e-6

    diagonal_value = 2.5
    print(f"Setting up tridiagonal system of size {n}...")

    print("\nComparing solvers performance...")
    A = tridiagonal_matrix(n, diagonal_value=diagonal_value)
    b = rhs_random(n, seed=42)

    # Solve with JAX-AMG (CG)
    t_start = time.time()
    x, info = jaxamg.solve(
        A,
        b,
        config={
            "solver": "CG",
            "tolerance": tolerance,
            "max_iters": max_iters,
            "convergence": "RELATIVE_MAX",
        },
    )
    x.block_until_ready()
    solve_time = time.time() - t_start
    residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
    print(
        f"JAX-AMG (CG):        solve time: {solve_time:.2f}s, residual: {residual:.2e}"
    )

    # Solve with JAX-AMG (BiCGSTAB)
    t_start = time.time()
    x, info = jaxamg.solve(
        A,
        b,
        config={
            "solver": "BICGSTAB",
            "tolerance": tolerance,
            "max_iters": max_iters,
            "convergence": "RELATIVE_MAX",
        },
    )
    x.block_until_ready()
    solve_time = time.time() - t_start
    residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
    print(
        f"JAX-AMG (BiCGSTAB):  solve time: {solve_time:.2f}s, residual: {residual:.2e}"
    )

    # Solve with JAX (CG)
    t_start = time.time()
    x, _ = cg(A, b, tol=tolerance, maxiter=max_iters)
    x.block_until_ready()
    solve_time = time.time() - t_start
    residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
    print(
        f"JAX     (CG):        solve time: {solve_time:.2f}s, residual: {residual:.2e}"
    )

    # Solve with JAX (BiCGSTAB)
    t_start = time.time()
    x, _ = bicgstab(A, b, tol=tolerance, maxiter=max_iters)
    x.block_until_ready()
    solve_time = time.time() - t_start
    residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)
    print(
        f"JAX     (BiCGSTAB):  solve time: {solve_time:.2f}s, residual: {residual:.2e}"
    )

    print("\nComparing JIT-compiled gradient computation performance...")

    # JAX-AMG (CG)
    def loss_fn(b):
        x, _ = jaxamg.solve(
            A,
            b,
            config={
                "solver": "CG",
                "tolerance": tolerance,
                "max_iters": max_iters,
                "convergence": "RELATIVE_MAX",
            },
        )
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    t_start = time.time()
    grad_fn(b).block_until_ready()
    warmup_time = time.time() - t_start

    t_start = time.time()
    grad = grad_fn(b)
    grad.block_until_ready()
    compute_time = time.time() - t_start
    print(
        f"JAX-AMG (CG):        warmup time: {warmup_time:6.2f}s, compute time: {compute_time:.2f}s, grad norm: {jnp.linalg.norm(grad):.2e}"
    )

    # JAX-AMG (BiCGSTAB)
    def loss_fn(b):
        x, _ = jaxamg.solve(
            A,
            b,
            config={
                "solver": "BICGSTAB",
                "tolerance": tolerance,
                "max_iters": max_iters,
                "convergence": "RELATIVE_MAX",
            },
        )
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    t_start = time.time()
    grad_fn(b).block_until_ready()
    warmup_time = time.time() - t_start

    t_start = time.time()
    grad = grad_fn(b)
    grad.block_until_ready()
    compute_time = time.time() - t_start
    print(
        f"JAX-AMG (BiCGSTAB):  warmup time: {warmup_time:6.2f}s, compute time: {compute_time:.2f}s, grad norm: {jnp.linalg.norm(grad):.2e}"
    )

    # JAX (CG)
    def loss_fn(b):
        x, _ = cg(A, b, tol=tolerance, maxiter=max_iters)
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    t_start = time.time()
    grad_fn(b).block_until_ready()
    warmup_time = time.time() - t_start

    t_start = time.time()
    grad = grad_fn(b)
    grad.block_until_ready()
    compute_time = time.time() - t_start
    print(
        f"JAX     (CG):        warmup time: {warmup_time:6.2f}s, compute time: {compute_time:.2f}s, grad norm: {jnp.linalg.norm(grad):.2e}"
    )

    # JAX (BiCGSTAB)
    def loss_fn(b):
        x, _ = bicgstab(A, b, tol=tolerance, maxiter=max_iters)
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    t_start = time.time()
    grad_fn(b).block_until_ready()
    warmup_time = time.time() - t_start

    t_start = time.time()
    grad = grad_fn(b)
    grad.block_until_ready()
    compute_time = time.time() - t_start
    print(
        f"JAX     (BiCGSTAB):  warmup time: {warmup_time:6.2f}s, compute time: {compute_time:.2f}s, grad norm: {jnp.linalg.norm(grad):.2e}"
    )

    print("\nComparing full optimization performance...")

    lr = 1e-4
    n_epochs = 20

    # JAX-AMG (CG)
    def loss_fn(b):
        x, _ = jaxamg.solve(
            A,
            b,
            config={
                "solver": "CG",
                "tolerance": tolerance,
                "max_iters": max_iters,
                "convergence": "RELATIVE_MAX",
            },
        )
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    b_opt = b
    t_start = time.time()
    for _ in range(n_epochs):
        g = grad_fn(b_opt)
        b_opt = b_opt - lr * g
    b_opt.block_until_ready()
    opt_time = time.time() - t_start
    print(f"JAX-AMG (CG):        opt time: {opt_time:.2f}s ({n_epochs} epochs)")

    # JAX-AMG (BiCGSTAB)
    def loss_fn(b):
        x, _ = jaxamg.solve(
            A,
            b,
            config={
                "solver": "BICGSTAB",
                "tolerance": tolerance,
                "max_iters": max_iters,
                "convergence": "RELATIVE_MAX",
            },
        )
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    b_opt = b
    t_start = time.time()
    for _ in range(n_epochs):
        g = grad_fn(b_opt)
        b_opt = b_opt - lr * g
    b_opt.block_until_ready()
    opt_time = time.time() - t_start
    print(f"JAX-AMG (BiCGSTAB):  opt time: {opt_time:.2f}s ({n_epochs} epochs)")

    # JAX (CG)
    def loss_fn(b):
        x, _ = cg(A, b, tol=tolerance, maxiter=max_iters)
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    b_opt = b
    t_start = time.time()
    for _ in range(n_epochs):
        g = grad_fn(b_opt)
        b_opt = b_opt - lr * g
    b_opt.block_until_ready()
    opt_time = time.time() - t_start
    print(f"JAX     (CG):        opt time: {opt_time:.2f}s ({n_epochs} epochs)")

    # JAX (BiCGSTAB)
    def loss_fn(b):
        x, _ = bicgstab(A, b, tol=tolerance, maxiter=max_iters)
        return jnp.sum(x**2)

    grad_fn = jax.jit(jax.grad(loss_fn))

    b_opt = b
    t_start = time.time()
    for _ in range(n_epochs):
        g = grad_fn(b_opt)
        b_opt = b_opt - lr * g
    b_opt.block_until_ready()
    opt_time = time.time() - t_start
    print(f"JAX     (BiCGSTAB):  opt time: {opt_time:.2f}s ({n_epochs} epochs)")


if __name__ == "__main__":
    main()
