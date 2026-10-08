"""
Demo: a singular Poisson solve on a disconnected domain.

Three uncoupled stretched grids of unequal size, assembled as one matrix: the
volume-normalized finite-volume operator A = D⁻¹L on each, so null(A) holds
each part's constant and null(Aᵀ) each part's cell volumes. Declared with one
label per row (``nullspace="constant", transpose_nullspace=V, labels=(count, labels)``),
against the same null space written as dense columns and the pseudoinverse:
the forward solve and the gradient. Then a part anchored in a solve (a
positive diagonal on it) takes the label -1 there.
"""

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import jaxamg
from jaxamg.matrices import poisson_matrix_stretched
from jaxamg.utils import to_scipy

jax.config.update("jax_enable_x64", True)
cfg = {"tolerance": 1e-10, "max_iters": 200}
sizes = ((24, 16, 1.06), (16, 12, 1.10), (8, 6, 1.0))


def main():
    parts = [
        poisson_matrix_stretched(nx, ny, r, dtype=jnp.float64) for nx, ny, r in sizes
    ]
    A = sp.block_diag([to_scipy(a) for a, _ in parts], format="csr")
    labels = np.concatenate([np.full(v.shape[0], i) for i, (_, v) in enumerate(parts)])
    V = np.concatenate([np.asarray(v) for _, v in parts])
    n, count = A.shape[0], len(parts)
    rng = np.random.default_rng(0)
    b, w = jnp.asarray(rng.standard_normal(n)), jnp.asarray(rng.standard_normal(n))

    # The same null space as dense columns: one per part and per declared column.
    N = (labels[:, None] == np.arange(count)).astype(float)
    M = N * V[:, None]
    forms = {
        "labels": dict(
            nullspace="constant", transpose_nullspace=V, labels=(count, labels)
        ),
        "dense columns": dict(nullspace=N, transpose_nullspace=M),
    }
    pinv = np.linalg.pinv(A.toarray())
    print(f"{count} parts, {n} rows; relative errors against the pseudoinverse\n")
    print(f"{'':14s}{'iterations':>11s}{'forward':>10s}{'gradient':>10s}")
    results = {}
    for name, declared in forms.items():
        x, info = jaxamg.solve(A, b, **declared, **cfg)
        g = jax.jit(
            jax.grad(lambda b_: jnp.dot(w, jaxamg.solve(A, b_, **declared, **cfg)[0]))
        )(b)
        consistent = (
            np.asarray(b) - M @ np.linalg.lstsq(M, np.asarray(b), rcond=None)[0]
        )
        x_ref, g_ref = pinv @ consistent, pinv.T @ np.asarray(w)
        error = lambda v, ref: np.linalg.norm(np.asarray(v) - ref) / np.linalg.norm(ref)
        print(
            f"{name:14s}{info['iterations']:11d}{error(x, x_ref):10.1e}{error(g, g_ref):10.1e}"
        )
        results[name] = np.asarray(x)
    print(
        f"\nlabels against dense columns: {np.abs(results['labels'] - results['dense columns']).max():.1e}"
    )

    # A part anchored in a solve: label -1 there (its rows keep their own solution).
    anchored = labels == 0
    A2 = (A + sp.diags(np.where(anchored, 0.5, 0.0))).tocsr()
    rows = jnp.asarray(np.where(anchored, -1, labels), jnp.int32)
    solve = jax.jit(
        lambda b_, r: jaxamg.solve(
            A2,
            b_,
            nullspace="constant",
            transpose_nullspace=V,
            labels=(count, r),
            **cfg,
        )[0]
    )
    x = solve(b, rows)
    M2 = M[:, 1:]
    consistent = np.asarray(b) - M2 @ np.linalg.lstsq(M2, np.asarray(b), rcond=None)[0]
    reference = np.linalg.pinv(A2.toarray()) @ consistent
    print(
        f"part 0 anchored (labels traced, -1 there): {np.linalg.norm(x - reference) / np.linalg.norm(reference):.1e}"
    )


if __name__ == "__main__":
    main()
