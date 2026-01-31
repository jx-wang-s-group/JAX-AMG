"""
Demo: Optimization via automatic differentiation.

This example demonstrates using JAX's automatic differentiation
and JIT compilation with the AmgX solver for gradient-based optimization.
"""

import jax
import jax.numpy as jnp
from jaxamg import amgx_solve
from jaxamg.matrices import tridiagonal_matrix, rhs_ones, rhs_linear, rhs_random


def main():

    # Setup tridiagonal system
    n = 64
    print(f"Setting up {n}×{n} system...")
    A = tridiagonal_matrix(n, diagonal_value=4.0)  # Better conditioned

    # Initial right-hand side
    b_init = rhs_ones(n)

    # Define a loss function: L(b) = ||x||² where x = A⁻¹b
    def loss(b):
        """Loss function: sum of squared solution components."""
        x = amgx_solve(A, b)
        return jnp.sum(x * x)

    # JIT-compile the loss and gradient functions
    print("Compiling JIT functions...")
    loss_jit = jax.jit(loss)
    grad_jit = jax.jit(jax.grad(loss))

    print("Compiling loss function...")
    loss_value = loss_jit(b_init)
    loss_value.block_until_ready()

    print("Compiling gradient function...")
    gradient = grad_jit(b_init)
    gradient.block_until_ready()

    # Demonstrate gradient-based optimization with different starting points
    print("\nGradient Descent Optimization for Different Initial Conditions:")
    print("=" * 70)

    test_inputs = [
        ("Constant RHS", rhs_ones(n)),
        ("Linear RHS", rhs_linear(n)),
        ("Random RHS", rhs_random(n)),
    ]

    learning_rate = 0.01
    num_iterations = 10

    for name, b_init_test in test_inputs:
        print(f"\n{name}:")
        print(f"{'Iter':<6} {'Loss':<15} {'Grad Norm':<15} {'Loss Change':<15}")
        print("-" * 70)

        b_current = b_init_test

        for i in range(num_iterations):
            # Compute loss and gradient
            loss_current = float(loss_jit(b_current))
            gradient = grad_jit(b_current)
            grad_norm = float(jnp.linalg.norm(gradient))

            # Update b using gradient descent
            b_next = b_current - learning_rate * gradient
            loss_next = float(loss_jit(b_next))
            loss_change = loss_next - loss_current

            # Display iteration info
            print(
                f"{i:<6} {loss_current:<15.6e} {grad_norm:<15.6e} {loss_change:<15.6e}"
            )

            # Update for next iteration
            b_current = b_next


if __name__ == "__main__":
    main()
