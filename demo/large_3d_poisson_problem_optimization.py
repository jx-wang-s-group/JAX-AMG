"""
Demo: End-to-end optimization of a large 3D Poisson system.

Problem:
    Minimize L(theta) = 0.5 * ||x(theta) - x_target||^2
    subject to A x(theta) = theta * b_0

    where A is the 3D Poisson matrix (256^3 grid),
    x_target is the solution for theta = 1.0.

This demonstrates JAX's ability to differentiate through the linear solver
for large systems.
"""

import time

import jax
import jax.numpy as jnp

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
    print(f"Matrix construction time: {time.time() - start:.2f}s")

    # Base RHS vector
    b0 = rhs_ones(n3)

    # Ground truth
    print("Generating target solution (theta_true = 1.0)...")
    theta_true = 1.0
    b_true = b0 * theta_true

    start = time.time()
    # High precision solve for target
    x_target, info_target = jaxamg.solve(
        A, b_true, solver="CG", max_iters=2000, tolerance=1e-8
    )
    print(f"Status: {info_target['status']}")

    # Define loss function and optimization step
    def solve_model(theta):
        b = b0 * theta
        A_ = jaxamg.with_cache(A, is_symmetric=True)
        x, _ = jaxamg.solve(A_, b, solver="CG")
        return x

    def loss_fn(theta):
        x_pred = solve_model(theta)
        loss = 0.5 * jnp.sum((x_pred - x_target) ** 2)
        return loss

    # Update step
    @jax.jit
    def update_step(theta, learning_rate):
        loss, grad = jax.value_and_grad(loss_fn)(theta)

        # Heuristic scaling factor
        scale = jnp.sum(x_target**2)

        new_theta = theta - learning_rate * grad / scale
        return new_theta, loss, grad

    # Optimization Loop
    theta = 10.0  # Initial guess
    lr = 0.5  # Learning rate

    print("\nStarting optimization:")
    print(f"Initial theta: {theta:.4f}")
    print(f"Target theta:  {theta_true:.4f}")
    print(f"Learning rate: {lr}")

    print("\nIter | Theta  | Loss     | Gradient | Time")
    print("-" * 50)

    start_opt = time.time()

    for i in range(10):
        step_start = time.time()
        theta, loss, grad = update_step(theta, lr)
        loss.block_until_ready()
        step_time = time.time() - step_start

        print(f"{i:4d} | {theta:.4f} | {loss:.2e} | {grad:.2e} | {step_time:.2f}s")

        # Check convergence
        if abs(theta - theta_true) < 1e-4:
            break

    total_time = time.time() - start_opt
    print(f"\nTotal optimization time: {total_time:.2f}s")
    print(f"Final theta: {theta:.6f}")
    print(f"Final Error: {abs(theta - theta_true):.6e}")


if __name__ == "__main__":
    main()
