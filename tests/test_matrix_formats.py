"""Test solver with differnet matrix input formats."""

import numpy as np
import pytest

import jaxamg
from jaxamg.matrices import rhs_ones, tridiagonal_matrix
from jaxamg.utils import to_scipy


class TestMatrixFormats:
    """Test solver accepts various matrix formats."""

    @pytest.fixture
    def setup_matrices(self):
        """Create test matrices in various formats."""
        n = 5
        b = rhs_ones(n)

        # Create base BCSR matrix
        A_bcsr = tridiagonal_matrix(n, diagonal_value=4.0)

        return A_bcsr, b

    def test_bcoo_format(self, setup_matrices):
        """Test with BCOO format."""
        A, b = setup_matrices

        A_bcoo = A.to_bcoo()
        x, _ = jaxamg.solve(A_bcoo, b)

        np.testing.assert_allclose(b, A @ x, rtol=1e-6)

    def test_scipy_csr_format(self, setup_matrices):
        """Test with SciPy CSR format."""
        A, b = setup_matrices

        A_scipy = to_scipy(A, format="csr")
        x, _ = jaxamg.solve(A_scipy, b)

        np.testing.assert_allclose(b, A @ x, rtol=1e-6)

    def test_scipy_coo_format(self, setup_matrices):
        """Test with SciPy COO format."""
        A, b = setup_matrices

        A_scipy_coo = to_scipy(A, format="coo")
        x, _ = jaxamg.solve(A_scipy_coo, b)

        np.testing.assert_allclose(b, A @ x, rtol=1e-6)

    def test_scipy_csc_format(self, setup_matrices):
        """Test with SciPy CSC format."""
        A, b = setup_matrices

        A_scipy_csc = to_scipy(A, format="csc")
        x, _ = jaxamg.solve(A_scipy_csc, b)

        np.testing.assert_allclose(b, A @ x, rtol=1e-6)

    def test_dense_jax_array(self, setup_matrices):
        """Test with dense JAX array."""
        A, b = setup_matrices

        A_dense = A.todense()
        x, _ = jaxamg.solve(A_dense, b)

        np.testing.assert_allclose(b, A @ x, rtol=1e-6)

    def test_dense_numpy_array(self, setup_matrices):
        """Test with dense NumPy array."""
        A, b = setup_matrices

        A_numpy = np.asarray(A.todense())
        x, _ = jaxamg.solve(A_numpy, b)

        np.testing.assert_allclose(b, A @ x, rtol=1e-6)
