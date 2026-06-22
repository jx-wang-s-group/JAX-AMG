import jax
import jax.numpy as jnp
import numpy as np

from jaxamg.matrices import tridiagonal_operator
from jaxamg.utils import (
    get_column_coloring,
    get_sparsity_pattern,
    materialize_sparse_matrix,
)


class TestGraphColoring:
    def test_coloring_reconstruction(self):
        """Verify that we can reconstruct a matrix using graph coloring."""
        n = 10
        shape = (n, n)

        # Get sparsity pattern of a tridiagonal operator
        operator = tridiagonal_operator(-2.0)
        rows, cols = get_sparsity_pattern(operator, shape)

        assert len(rows) >= 3 * n - 2  # Tridiagonal nnz

        # Compute coloring
        colors, n_colors = get_column_coloring(rows, cols, shape)

        # The coloring must be VALID: any two columns sharing a non-zero row
        # must get different colors (so they can be probed together). Check this
        # invariant per row -- it holds deterministically for any correct
        # coloring, independent of how many colors the algorithm chooses (the
        # color count itself is algorithm-dependent and not asserted here).
        rows_arr, cols_arr = np.asarray(rows), np.asarray(cols)
        for r in np.unique(rows_arr):
            cols_in_row = cols_arr[rows_arr == r]
            row_colors = colors[cols_in_row].tolist()
            assert len(set(row_colors)) == len(cols_in_row), (
                f"invalid coloring: row {r} columns {cols_in_row.tolist()} "
                f"have repeated colors {row_colors}"
            )

        # Materialize (verifying JIT compatibility)
        @jax.jit
        def reconstruct():
            return materialize_sparse_matrix(
                operator, shape, rows, cols, colors, n_colors
            )

        A_csr = reconstruct()

        # Verify against ground truth
        A_dense = A_csr.todense()
        A_expected = jax.vmap(operator)(jnp.eye(n)).T

        np.testing.assert_allclose(A_dense, A_expected)

    def test_coloring_autodiff(self):
        """Verify gradients flow correctly through the materialized matrix."""
        n = 5
        shape = (n, n)

        # Parameterized operator: A(theta) = theta * Laplacian
        def operator(theta, x):
            return theta * tridiagonal_operator(-2.0)(x)

        # Use Laplacian to determine the fixed sparsity pattern
        L = lambda x: operator(1.0, x)
        rows, cols = get_sparsity_pattern(L, shape)
        colors, n_colors = get_column_coloring(rows, cols, shape)

        # Define a loss function
        @jax.jit
        def loss(theta):
            # Materialize inside JIT using pre-computed coloring
            A_csr = materialize_sparse_matrix(
                lambda x: operator(theta, x), shape, rows, cols, colors, n_colors
            )

            # Simple scalar function: sum(A_ij^2)
            return jnp.sum(A_csr.data**2)

        # Calculate gradient using JAX
        theta_val = 2.0
        grad_jax = jax.grad(loss)(theta_val)

        # Calculate analytical gradient
        # A = theta * L
        # Loss = sum((theta * L)^2) = theta^2 * sum(L^2)
        # Grad = 2 * theta * sum(L^2)
        L_csr = materialize_sparse_matrix(L, shape, rows, cols, colors, n_colors)
        sum_sq_L = jnp.sum(L_csr.data**2)
        grad_analytical = 2 * theta_val * sum_sq_L

        np.testing.assert_allclose(grad_jax, grad_analytical)
