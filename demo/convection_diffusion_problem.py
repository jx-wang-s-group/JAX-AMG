"""
Demo: Solve convection-diffusion equation.

This example solves the convection-diffusion equation with a manufactured solution,
and compares the numerical solution to the exact solution.

    Exact Solution: u(x, y) = sin(pi * x) * sin(pi * y)
    Domain: [0, 1] x [0, 1]
    BCs: u = 0 on boundary.

    Equation: -e * laplacian(u) + vx * du/dx + vy * du/dy = f

    We compute f analytically and solve Ax = f.
"""

import jax.numpy as jnp
import numpy as np

import jaxamg
from jaxamg.matrices import convection_diffusion_matrix_2d


def main():

    print("Setting up convection-diffusion problem...")
    n = 256  # Grid size
    h = 1.0 / (n - 1)  # Mesh size

    # Grid
    x = jnp.linspace(0, 1, n)
    y = jnp.linspace(0, 1, n)
    X, Y = jnp.meshgrid(x, y, indexing="ij")

    # Parameters
    epsilon = 1e-3  # Diffusivity
    velocity = 1.0  # Velocity magnitude
    theta = np.pi / 4.0  # Flow angle

    vx = velocity * np.cos(theta)
    vy = velocity * np.sin(theta)

    print(f"Grid: {n}x{n} (h={h:.5f})")
    print(f"Params: epsilon={epsilon}, velocity={velocity}, theta={theta:.2f}")

    # Exact Solution and Derivatives
    # u = sin(pi x) sin(pi y)
    sin_pix = jnp.sin(np.pi * X)
    sin_piy = jnp.sin(np.pi * Y)
    cos_pix = jnp.cos(np.pi * X)
    cos_piy = jnp.cos(np.pi * Y)

    u_exact = sin_pix * sin_piy

    # Laplacian u = -2 pi^2 u
    laplacian_u = -2 * np.pi**2 * u_exact

    # Gradients
    # du/dx = pi cos(pi x) sin(pi y)
    grad_u_x = np.pi * cos_pix * sin_piy

    # du/dy = pi sin(pi x) cos(pi y)
    grad_u_y = np.pi * sin_pix * cos_piy

    # Calculate source term f
    # -eps * Delta u + v . Grad u = f
    f = -epsilon * laplacian_u + vx * grad_u_x + vy * grad_u_y

    # Flatten source term for RHS
    b = f.ravel()

    # Flatten exact solution for comparison
    u_exact_flat = u_exact.ravel()

    # Construct matrix
    print("Constructing matrix...")
    A = convection_diffusion_matrix_2d(
        n, epsilon=epsilon, theta=theta, velocity=velocity
    )

    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {
            "solver": "AMG",
            "smoother": "JACOBI_L1",
            "presweeps": 2,
            "postsweeps": 2,
        },
        "tolerance": 1e-9,
    }

    # Solve
    print("Solving...")
    u_pred, info = jaxamg.solve(
        A, b, config=config, save_stats_file="stats_convection_diffusion.txt"
    )

    print(info)

    # Display results
    error = jnp.abs(u_pred - u_exact_flat)
    l2_error = jnp.linalg.norm(error) / jnp.linalg.norm(u_exact_flat)
    max_error = jnp.max(error)

    print(f"\nComputed solution (first 5 entries): {u_pred[:5]}")
    print(f"Analytic solution (first 5 entries): {u_exact_flat[:5]}")
    print(f"L2 relative error: {l2_error:.6e}")
    print(f"Max absolute error: {max_error:.6e}")


if __name__ == "__main__":
    main()
