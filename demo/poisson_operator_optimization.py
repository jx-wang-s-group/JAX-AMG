"""
Demo: Optimize skew parameter of a Poisson operator.

Use JIT with parameterized Poisson operator for end-to-end optimization of the skew parameter.
"""

import jax
import jax.numpy as jnp

import jaxamg
from jaxamg.matrices import poisson_operator, rhs_ones


def main():
    n = 9

    # Ground truth
    true_skew = 5.0
    b = rhs_ones(n)
    A_true = poisson_operator(true_skew)
    x_target, info = jaxamg.solve(A_true, b)

    # Compute coloring cache
    print("Computing operator coloring...")
    skew_init = 1.0  # Use initial guess
    coloring_cache = jaxamg.cache_coloring(poisson_operator(skew_init), shape=n)
    print(f"Graph coloring computed. Number of colors: {coloring_cache[3]}")

    # Define loss function
    def loss_fn(skew, b, x_true):
        # Create operator with cached coloring
        A = jaxamg.with_cache(poisson_operator(skew), coloring=coloring_cache)

        # Solve
        x_pred, _ = jaxamg.solve(A, b)

        # Compute loss
        loss = jnp.mean((x_pred - x_true) ** 2)
        return loss

    # Gradient Descent
    print("Starting optimization...")
    lr = 5.0  # Learning rate
    grad_fn = jax.jit(jax.grad(loss_fn))

    for epoch in range(200):
        # Force solver rebuild
        if epoch % 10 == 0:
            jaxamg.clear_solver_cache()

        l = loss_fn(skew_init, b, x_target)
        g = grad_fn(skew_init, b, x_target)

        print(f"Epoch {epoch}: skew = {skew_init:.4f}, loss = {l:.6f}, grad = {g:.6f}")

        skew_init = skew_init - lr * g

        if l < 1e-6:
            print("Converged!")
            break

    print(f"Final skew: {skew_init:.4f}, True skew: {true_skew:.4f}")


if __name__ == "__main__":
    main()
