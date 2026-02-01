"""Tests for matrix construction functions in jaxamg.matrices."""

import pytest
import numpy as np
import jax.numpy as jnp
from jaxamg.matrices import poisson_matrix, poisson3d_matrix
from jaxamg.utils import to_scipy


class TestMatrixConstruction:
    def test_poisson_matrix_2d(self):
        """Test values of 2D Poisson matrix for a small 3x3 grid."""
        n = 3
        # Grid layout (n=3):
        # 0 1 2
        # 3 4 5
        # 6 7 8

        # Center point is 4 (at i=1, j=1)
        # Neighbors:
        #   Top    (i-1): 1
        #   Left   (j-1): 3
        #   Right  (j+1): 5
        #   Bottom (i+1): 7

        A = poisson_matrix(n)
        A_dense = A.todense()

        # Check center row -> index 4
        np.testing.assert_allclose(A_dense[4], [0, -1, 0, -1, 4, -1, 0, -1, 0])

        # Check corner (0,0) -> index 0
        np.testing.assert_allclose(A_dense[0], [4, -1, 0, -1, 0, 0, 0, 0, 0])

        #  Check symmetry
        np.testing.assert_allclose(A_dense, A_dense.T)

    def test_poisson_matrix_2d_skew(self):
        """Test non-symmetric system with skew parameter."""
        n = 3
        skew = 1.0
        A = poisson_matrix(n, skew=skew)
        A_dense = A.todense()

        # "Negative" direction neighbors (Left, Top) -> value: -1 - skew/2
        # "Positive" direction neighbors (Right, Bottom) -> value: -1 + skew/2

        # Check center row -> index 4
        row_4 = A_dense[4]
        assert row_4[4] == 4.0

        neg_neighbors = np.array([1, 3])  # j-1, i-1
        pos_neighbors = np.array([5, 7])  # j+1, i+1

        expected_neg = -1.0 - skew / 2
        expected_pos = -1.0 + skew / 2

        np.testing.assert_allclose(row_4[neg_neighbors], expected_neg * np.ones(2))
        np.testing.assert_allclose(row_4[pos_neighbors], expected_pos * np.ones(2))

    def test_poisson_matrix_3d(self):
        """Test values of 3D Poisson matrix for a small 3x3x3 grid."""
        n = 3
        # Grid Indexing:
        # Index = i*n^2 + j*n + k
        # Center point (1,1,1) -> index 13

        A = poisson3d_matrix(n)
        A_dense = A.todense()

        row_13 = A_dense[13]

        # Diagonal element is always 6.0 in 3D 7-point stencil
        assert row_13[13] == 6.0

        # Neighbors of (1,1,1):
        # ---------------------
        # Left   (j-1): (1,0,1) -> index 10
        # Right  (j+1): (1,2,1) -> index 16
        # Front  (i-1): (0,1,1) -> index 4
        # Back   (i+1): (2,1,1) -> index 22
        # Bottom (k-1): (1,1,0) -> index 12
        # Top    (k+1): (1,1,2) -> index 14

        neighbors = np.array([10, 16, 4, 22, 12, 14])
        np.testing.assert_allclose(row_13[neighbors], -1.0 * np.ones(6))

        # Interior sum check (should be zero)
        assert jnp.sum(row_13) == 0.0

        # Check symmetry
        np.testing.assert_allclose(A_dense, A_dense.T)

    def test_poisson_matrix_3d_skew(self):
        """Test non-symmetric values for 3D matrix."""
        n = 3
        skew = 0.5
        A = poisson3d_matrix(n, skew=skew)
        A_dense = A.todense()

        # Center point (1,1,1) -> index 13
        row_13 = A_dense[13]

        # Neighbors
        # "Negative" direction neighbors -> value: -1 - skew/2
        #   Left (10), Front (4), Bottom (12)
        #
        # "Positive" direction neighbors -> value: -1 + skew/2
        #   Right (16), Back (22), Top (14)

        neg_neighbors = np.array([10, 4, 12])  # j-1, i-1, k-1
        pos_neighbors = np.array([16, 22, 14])  # j+1, i+1, k+1

        expected_neg = -1.0 - skew / 2
        expected_pos = -1.0 + skew / 2

        np.testing.assert_allclose(row_13[neg_neighbors], [expected_neg] * 3)
        np.testing.assert_allclose(row_13[pos_neighbors], [expected_pos] * 3)
