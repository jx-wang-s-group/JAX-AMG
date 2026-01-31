"""
Demo: Optimize a parameter of a linear operator.

Use JIT with parameterized operator for end-to-end optimization of a parameter (diagonal value).
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np

from jaxamg import amgx_solve
from jaxamg.matrices import tridiagonal_operator, rhs_ones


def main():
    n = 32
    print(f"Setting up {n}x{n} diagonal system with true diagonal = 4.0...")

    # Ground truth
    true_diag = 4.0
    b = rhs_ones(n)
    A_true = tridiagonal_operator(true_diag)
    x_target = amgx_solve(A_true, b)

    # Pre-scan: Compute coloring info outside of JIT
    print("Pre-scanning operator structure...")
    diag_init = 4.5  # Use initial guess to pre-scan
    A_dummy = tridiagonal_operator(diag_init)
    _ = amgx_solve(A_dummy, b)
    coloring_cache = A_dummy._amgx_coloring_info
    print(f"Graph coloring computed. Number of colors: {coloring_cache[3]}")

    # Define loss function
    @jax.jit
    def loss_fn(diag, b, x_true):
        # Create operator for current diagonal value
        A = tridiagonal_operator(diag)

        # Attach pre-computed coloring info
        object.__setattr__(A, "_amgx_coloring_info", coloring_cache)

        # Solve
        x_pred = amgx_solve(A, b)

        # Compute loss
        loss = jnp.mean((x_pred - x_true) ** 2)
        return loss

    # Gradient Descent
    print("Starting optimization...")
    lr = 2.0  # Learning rate
    grad_fn = jax.grad(loss_fn)

    for epoch in range(100):
        l = loss_fn(diag_init, b, x_target)
        g = grad_fn(diag_init, b, x_target)

        print(f"Epoch {epoch}: diag = {diag_init:.4f}, loss = {l:.6f}, grad = {g:.6f}")

        diag_init = diag_init - lr * g

        if l < 1e-6:
            print("Converged!")
            break

    print(f"Final diag: {diag_init:.4f}, True diag: {true_diag:.4f}")


if __name__ == "__main__":
    main()
