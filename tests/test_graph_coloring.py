import jax
import jax.numpy as jnp
import numpy as np

from jaxamg.matrices import poisson3d_operator, tridiagonal_operator
from jaxamg.sparsity import (
    get_column_coloring,
    materialize_sparse_matrix,
    probe_sparsity_pattern,
    trace_sparsity_pattern,
)


def _assert_valid_coloring(rows, cols, colors):
    """No two columns sharing a row may share a color."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    for r in np.unique(rows):
        cols_in_row = cols[rows == r]
        row_colors = colors[cols_in_row].tolist()
        assert len(set(row_colors)) == len(cols_in_row), (
            f"invalid coloring: row {r} columns {cols_in_row.tolist()} "
            f"have repeated colors {row_colors}"
        )


class TestGraphColoring:
    def test_coloring_reconstruction(self):
        """Verify that we can reconstruct a matrix using graph coloring."""
        n = 10
        shape = (n, n)

        # Get sparsity pattern of a tridiagonal operator
        operator = tridiagonal_operator(-2.0)
        rows, cols = probe_sparsity_pattern(operator, shape)

        assert len(rows) >= 3 * n - 2  # Tridiagonal nnz

        # Compute coloring
        colors, n_colors = get_column_coloring(rows, cols, shape)

        # The coloring must be VALID: any two columns sharing a non-zero row must
        # get different colors (so they can be probed together). The color *count*
        # is algorithm-dependent and not asserted here.
        _assert_valid_coloring(rows, cols, colors)

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
        rows, cols = probe_sparsity_pattern(L, shape)
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

    def test_valid_coloring_on_dense_3d_stencil(self):
        # A denser pattern (3D 7-point, ~7 nonzeros/row) exercises the
        # Jones-Plassman coloring over many rounds; the result must stay valid.
        n = 5**3
        op = poisson3d_operator(robin=2.0)
        rows, cols = trace_sparsity_pattern(op, (n, n))
        colors, n_colors = get_column_coloring(rows, cols, (n, n))
        _assert_valid_coloring(rows, cols, colors)
        assert n_colors >= 7  # at least max nonzeros per row


class TestColoringEdgeCases:
    def test_empty_pattern(self):
        # No nonzeros -> every column uncolored (-1), zero colors used.
        m = 8
        colors, n_colors = get_column_coloring(
            np.array([], dtype=int), np.array([], dtype=int), (m, m)
        )
        assert n_colors == 0
        assert colors.shape == (m,)
        assert (colors == -1).all()


class TestProbing:
    def test_recovers_tridiagonal_pattern(self):
        n = 10
        op = tridiagonal_operator(-2.0)
        rows, cols = probe_sparsity_pattern(op, (n, n))
        got = set(zip(np.asarray(rows).tolist(), np.asarray(cols).tolist()))
        expected = {(i, j) for i in range(n) for j in range(n) if abs(i - j) <= 1}
        assert got == expected

    def test_empty_shape_returns_empty(self):
        rows, cols = probe_sparsity_pattern(lambda x: x, (0, 0))
        assert rows.size == 0 and cols.size == 0
