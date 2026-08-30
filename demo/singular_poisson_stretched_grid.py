"""
Demo: differentiating a singular Poisson solve on a stretched grid.

Volume-normalized finite-volume Poisson operator A = D⁻¹L (periodic/Neumann):
singular, and nonsymmetric on a stretched grid (A·1 = 0 but Aᵀ·V = 0).
Compares a plain solve with `nullspace="constant", transpose_nullspace=V`.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np

import jaxamg
from jaxamg import NullSpaceWarning
from jaxamg.matrices import poisson_matrix_stretched
from jaxamg.utils import to_scipy

jax.config.update("jax_enable_x64", True)

nx, ny = 64, 48
stretches = [1.00, 1.02, 1.04, 1.08, 1.10]
cfg = {"tolerance": 1e-10, "max_iters": 200}


def run_case(stretch):
    A, V = poisson_matrix_stretched(nx, ny, stretch, dtype=jnp.float64)
    n = A.shape[0]
    V_np = np.asarray(V)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    b -= (V_np @ b) / V_np.sum()  # compatible RHS
    w = rng.standard_normal(n) + 0.5
    b, w = jnp.asarray(b), jnp.asarray(w)

    A_dense = to_scipy(A).toarray()
    A_T = to_scipy(A).T.tocsr()
    pinv = np.linalg.pinv(A_dense)
    x_ref, g_ref = pinv @ np.asarray(b), pinv.T @ np.asarray(w)

    def err(v, ref, null):
        # plain solves are compared up to the null vector (1 forward, V adjoint)
        v = np.asarray(v)
        if null is not None:
            v = v - null * ((v - ref) @ null) / (null @ null)
        return np.linalg.norm(v - ref) / np.linalg.norm(ref)

    rows = []
    ones = np.ones(n)
    for kwargs, kwargs_T, null_x, null_g in [
        ({}, {}, ones, V_np),
        (
            {"nullspace": "constant", "transpose_nullspace": V},
            {"nullspace": V, "transpose_nullspace": "constant"},
            None,
            None,
        ),
    ]:
        x, info = jaxamg.solve(A, b, **kwargs, **cfg)
        # adjoint system Aᵀλ = g solved by jax.grad (explicitly, to read its iteration count)
        _, info_adj = jaxamg.solve(A_T, w, **kwargs_T, **cfg)
        g = jax.grad(lambda b_: jnp.dot(w, jaxamg.solve(A, b_, **kwargs, **cfg)[0]))(b)
        rows.append(
            f"{info['iterations']:4d} {err(x, x_ref, null_x):8.1e}   "
            f"{info_adj['iterations']:4d} {err(g, g_ref, null_g):8.1e}"
        )
    return rows


def main():
    warnings.simplefilter("ignore", NullSpaceWarning)
    print(f"{nx}x{ny} cells, float64, {cfg} (iterations and relative error)\n")
    print(
        f"{'':8s}{'------- plain solve -------':^30s}   {'----- with nullspace -----':^30s}"
    )
    print(f"{'':8s}{'forward':^15s}{'adjoint':^15s}   {'forward':^15s}{'adjoint':^15s}")
    print(
        f"{'stretch':8s}{'iters':>5s}{'error':>9s}   {'iters':>5s}{'error':>9s}   "
        f"{'iters':>5s}{'error':>9s}   {'iters':>5s}{'error':>9s}"
    )
    for stretch in stretches:
        plain, ns = run_case(stretch)
        print(f"{stretch:<8.2f}{plain}   {ns}")


if __name__ == "__main__":
    main()
