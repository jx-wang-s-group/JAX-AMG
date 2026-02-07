"""
Standard test matrices and RHS vectors.

This module provides common sparse matrix patterns and right-hand-side
vector generators for testing and demonstration purposes.
"""

import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np
import scipy.sparse as sp

from .mpi_utils import get_partition_info

from typing import cast, Callable
from jax.typing import DTypeLike


def tridiagonal_matrix(
    n: int, diagonal_value: float = 2.0, dtype: DTypeLike = jnp.float32
) -> jsp.BCSR:
    """Create a tridiagonal matrix in BCSR format with [-1, diagonal_value, -1] pattern.

    Args:
        n: Size of the matrix (n x n)
        diagonal_value: Value to place on the main diagonal (default 2.0)
        dtype: Data type for matrix values (default jnp.float32)

    Returns:
        JAX BCSR matrix with [-1, diagonal_value, -1] pattern
    """
    # Total non-zeros: 2 + 3*(n-2) + 2 = 3*n - 2
    nnz = 3 * n - 2

    # Build values array efficiently using vectorized operations
    # Pattern: [diag, -1] + [-1, diag, -1] * (n-2) + [-1, diag]

    if n == 1:
        # Special case: 1x1 matrix
        values = jnp.array([diagonal_value], dtype=dtype)
        indices = jnp.array([0], dtype=jnp.int32)
        indptr = jnp.array([0, 1], dtype=jnp.int32)
    elif n == 2:
        # Special case: 2x2 matrix
        values = jnp.array([diagonal_value, -1.0, -1.0, diagonal_value], dtype=dtype)
        indices = jnp.array([0, 1, 0, 1], dtype=jnp.int32)
        indptr = jnp.array([0, 2, 4], dtype=jnp.int32)
    else:
        # General case: n >= 3
        # Build middle rows pattern: [-1, diag, -1] repeated (n-2) times
        middle_pattern = jnp.array([-1.0, diagonal_value, -1.0], dtype=dtype)
        middle_values = jnp.tile(middle_pattern, n - 2)

        # Concatenate: first row + middle rows + last row
        values = jnp.concatenate(
            [
                jnp.array([diagonal_value, -1.0], dtype=dtype),  # First row
                middle_values,  # Middle rows
                jnp.array([-1.0, diagonal_value], dtype=dtype),  # Last row
            ]
        )

        # Build indices array
        # First row: [0, 1]
        # Middle rows: for row i: [i-1, i, i+1]
        middle_indices = jnp.stack(
            [
                jnp.arange(n - 2, dtype=jnp.int32),  # i-1
                jnp.arange(1, n - 1, dtype=jnp.int32),  # i
                jnp.arange(2, n, dtype=jnp.int32),  # i+1
            ],
            axis=1,
        ).ravel()

        # Last row: [n-2, n-1]
        indices = jnp.concatenate(
            [
                jnp.array([0, 1], dtype=jnp.int32),  # First row
                middle_indices,  # Middle rows
                jnp.array([n - 2, n - 1], dtype=jnp.int32),  # Last row
            ]
        )

        # Build indptr: [0, 2, 5, 8, ..., 3n-4, 3n-2]
        # First row has 2 entries, middle rows have 3 each, last row has 2
        row_lengths = jnp.concatenate(
            [
                jnp.array([2], dtype=jnp.int32),  # First row
                jnp.full(n - 2, 3, dtype=jnp.int32),  # Middle rows
                jnp.array([2], dtype=jnp.int32),  # Last row
            ]
        )
        indptr = jnp.concatenate(
            [jnp.array([0], dtype=jnp.int32), jnp.cumsum(row_lengths)]
        )

    return jsp.BCSR((values, indices, indptr), shape=(n, n))


def poisson_matrix(n: int, skew: float = 0.0) -> jsp.BCSR:
    """Create a 2D Poisson matrix on an n×n grid in BCSR format.

    The matrix represents the discretization of -Δu + skew * (∂u/∂x + ∂u/∂y)
    on a regular grid with standard 5-point stencil.

    Args:
        n: Grid size in each dimension (results in n² × n² matrix)
        skew: Skew-symmetric coefficient (default 0.0 for symmetric Poisson)
              Non-zero values add convection-like terms, making the matrix non-symmetric.
              Positive values create upwind-biased discretization.

    Returns:
        JAX BCSR matrix representing the 2D Poisson operator with optional skew
    """
    n2 = n * n  # Total size

    # Create grid indices using meshgrid
    i_grid, j_grid = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    row_indices = i_flat * n + j_flat

    # Build all entries using vectorized operations
    # For each grid point (i,j), we have up to 5 non-zeros:
    # - Diagonal: 4.0
    # - Left (j-1): -1.0 - skew/2 if j > 0
    # - Right (j+1): -1.0 + skew/2 if j < n-1
    # - Top (i-1): -1.0 - skew/2 if i > 0
    # - Bottom (i+1): -1.0 + skew/2 if i < n-1

    # Diagonal entries (always present)
    diag_rows = row_indices
    diag_cols = row_indices
    diag_vals = jnp.full(n2, 4.0, dtype=jnp.float32)

    # Left neighbors (j > 0)
    left_mask = j_flat > 0
    left_rows = row_indices[left_mask]
    left_cols = row_indices[left_mask] - 1
    left_vals = jnp.full(jnp.sum(left_mask), -1.0 - skew / 2.0, dtype=jnp.float32)

    # Right neighbors (j < n-1)
    right_mask = j_flat < n - 1
    right_rows = row_indices[right_mask]
    right_cols = row_indices[right_mask] + 1
    right_vals = jnp.full(jnp.sum(right_mask), -1.0 + skew / 2.0, dtype=jnp.float32)

    # Top neighbors (i > 0)
    top_mask = i_flat > 0
    top_rows = row_indices[top_mask]
    top_cols = row_indices[top_mask] - n
    top_vals = jnp.full(jnp.sum(top_mask), -1.0 - skew / 2.0, dtype=jnp.float32)

    # Bottom neighbors (i < n-1)
    bottom_mask = i_flat < n - 1
    bottom_rows = row_indices[bottom_mask]
    bottom_cols = row_indices[bottom_mask] + n
    bottom_vals = jnp.full(jnp.sum(bottom_mask), -1.0 + skew / 2.0, dtype=jnp.float32)

    # Concatenate all entries
    rows = jnp.concatenate([diag_rows, left_rows, right_rows, top_rows, bottom_rows])
    cols = jnp.concatenate([diag_cols, left_cols, right_cols, top_cols, bottom_cols])
    vals = jnp.concatenate([diag_vals, left_vals, right_vals, top_vals, bottom_vals])

    # Sort by (row, col)
    sort_idx = jnp.lexsort((cols, rows))
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    vals = vals[sort_idx]

    # Build indptr
    indptr = jnp.zeros(n2 + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows, length=n2)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts))

    return jsp.BCSR((vals, cols, indptr), shape=(n2, n2))


def poisson3d_matrix(n: int, skew: float = 0.0) -> jsp.BCSR:
    """Create a 3D Poisson matrix on an n×n×n grid in BCSR format.

    The matrix represents the discretization of -Δu + skew * (∂u/∂x + ∂u/∂y + ∂u/∂z)
    on a regular 3D grid with standard 7-point stencil.

    Args:
        n: Grid size in each dimension (results in n³ × n³ matrix)
        skew: Skew-symmetric coefficient (default 0.0 for symmetric Poisson)
              Non-zero values add convection-like terms, making the matrix non-symmetric.

    Returns:
        JAX BCSR matrix representing the 3D Poisson operator with optional skew
    """
    n3 = n * n * n  # Total size

    # Create grid indices
    # indexing='ij': i varies slowest, k varies fastest
    # Shape of grids: (n, n, n)
    i_grid, j_grid, k_grid = jnp.meshgrid(
        jnp.arange(n), jnp.arange(n), jnp.arange(n), indexing="ij"
    )
    # i corresponds to index i in (i, j, k) -> stride n*n
    # j corresponds to index j in (i, j, k) -> stride n
    # k corresponds to index k in (i, j, k) -> stride 1

    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    k_flat = k_grid.ravel()
    row_indices = (i_flat * n * n + j_flat * n + k_flat).astype(jnp.int32)

    # Diagonal entries (always present)
    diag_rows = row_indices
    diag_cols = row_indices
    diag_vals = jnp.full(n3, 6.0, dtype=jnp.float32)

    # Left neighbors (j > 0), stride n
    left_mask = j_flat > 0
    left_rows = row_indices[left_mask]
    left_cols = row_indices[left_mask] - n
    left_vals = jnp.full(jnp.sum(left_mask), -1.0 - skew / 2.0, dtype=jnp.float32)

    # Right neighbors (j < n-1), stride n
    right_mask = j_flat < n - 1
    right_rows = row_indices[right_mask]
    right_cols = row_indices[right_mask] + n
    right_vals = jnp.full(jnp.sum(right_mask), -1.0 + skew / 2.0, dtype=jnp.float32)

    # Front neighbors (i > 0), stride n*n
    front_mask = i_flat > 0
    front_rows = row_indices[front_mask]
    front_cols = row_indices[front_mask] - n * n
    front_vals = jnp.full(jnp.sum(front_mask), -1.0 - skew / 2.0, dtype=jnp.float32)

    # Back neighbors (i < n-1), stride n*n
    back_mask = i_flat < n - 1
    back_rows = row_indices[back_mask]
    back_cols = row_indices[back_mask] + n * n
    back_vals = jnp.full(jnp.sum(back_mask), -1.0 + skew / 2.0, dtype=jnp.float32)

    # Bottom neighbors (k > 0), stride 1
    bottom_mask = k_flat > 0
    bottom_rows = row_indices[bottom_mask]
    bottom_cols = row_indices[bottom_mask] - 1
    bottom_vals = jnp.full(jnp.sum(bottom_mask), -1.0 - skew / 2.0, dtype=jnp.float32)

    # Top neighbors (k < n-1), stride 1
    top_mask = k_flat < n - 1
    top_rows = row_indices[top_mask]
    top_cols = row_indices[top_mask] + 1
    top_vals = jnp.full(jnp.sum(top_mask), -1.0 + skew / 2.0, dtype=jnp.float32)

    # Concatenate all entries
    rows = jnp.concatenate(
        [diag_rows, left_rows, right_rows, front_rows, back_rows, bottom_rows, top_rows]
    )
    cols = jnp.concatenate(
        [diag_cols, left_cols, right_cols, front_cols, back_cols, bottom_cols, top_cols]
    )
    vals = jnp.concatenate(
        [diag_vals, left_vals, right_vals, front_vals, back_vals, bottom_vals, top_vals]
    )

    # Sort by (row, col)
    sort_idx = jnp.lexsort((cols, rows))
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    vals = vals[sort_idx]

    # Build indptr
    indptr = jnp.zeros(n3 + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows, length=n3)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts))

    return jsp.BCSR((vals, cols, indptr), shape=(n3, n3))


def random_matrix(
    n: int,
    density: float = 0.01,
    dtype: DTypeLike = jnp.float32,
    seed: int = 0,
) -> jsp.BCSR:
    """Create a random sparse matrix in BCSR format.

    Args:
        n: Size of the matrix (n x n)
        density: Density of non-zero entries (default 0.01)
        dtype: Data type for matrix values (default jnp.float32)
        seed: Random seed for reproducibility (default 0)

    Returns:
        JAX BCSR random sparse matrix
    """
    # Generate random sparse matrix using SciPy
    np_dtype: type = np.float32 if dtype == jnp.float32 else np.float64
    rng = np.random.default_rng(seed)
    A = cast(
        sp.csr_matrix,
        sp.random(
            n, n, density=density, format="csr", dtype=np_dtype, random_state=rng
        ),
    )

    # Make it diagonally dominant for better conditioning
    diag = abs(A).sum(axis=1).A1 + 1.0
    A.setdiag(diag)

    # Convert to JAX BCSR
    values = jnp.array(A.data)
    indices = jnp.array(A.indices, dtype=jnp.int32)
    indptr = jnp.array(A.indptr, dtype=jnp.int32)

    return jsp.BCSR((values, indices, indptr), shape=(n, n))


def tridiagonal_matrix_distributed(
    n_global: int,
    rank: int,
    nranks: int,
    diagonal_value: float = 2.0,
    dtype: DTypeLike = jnp.float32,
) -> tuple[jsp.BCSR, int, int]:
    """Create distributed tridiagonal matrix [-1, diagonal_value, -1] for MPI.

    Args:
        n_global: Global matrix size
        rank: MPI rank (0-indexed)
        nranks: Total number of MPI ranks
        diagonal_value: Value on main diagonal (default: 2.0)
        dtype: Data type for matrix values (default: jnp.float32)

    Returns:
        A_local: Local BCSR matrix partition (JAX)
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)
    """
    # Row-based partitioning
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)

    data, indices, indptr = [], [], [0]

    for local_i in range(n_local):
        global_i = row_start + local_i

        # Lower diagonal
        if global_i > 0:
            data.append(-1.0)
            indices.append(global_i - 1)

        # Main diagonal
        data.append(diagonal_value)
        indices.append(global_i)

        # Upper diagonal
        if global_i < n_global - 1:
            data.append(-1.0)
            indices.append(global_i + 1)

        indptr.append(len(data))

    A_local = jsp.BCSR(
        (
            jnp.array(data, dtype=dtype),
            jnp.array(indices, dtype=jnp.int32),
            jnp.array(indptr, dtype=jnp.int32),
        ),
        shape=(n_local, n_global),
    )

    return A_local, row_start, row_end


def poisson_matrix_distributed(
    nx: int, ny: int, rank: int, nranks: int, dtype: DTypeLike = jnp.float32
) -> tuple[jsp.BCSR, int, int]:
    """Create distributed 2D Poisson matrix with 5-point stencil for MPI.

    Args:
        nx: Grid size in x-direction
        ny: Grid size in y-direction
        rank: MPI rank (0-indexed)
        nranks: Total number of MPI ranks
        dtype: Data type for matrix values (default: jnp.float32)

    Returns:
        A_local: Local BCSR matrix partition (JAX)
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)
    """
    n = nx * ny

    # Row-based partitioning
    row_start, row_end, n_local = get_partition_info(n, rank, nranks)

    rows, cols, vals = [], [], []

    for local_i in range(n_local):
        global_i = row_start + local_i
        ix = global_i % nx
        iy = global_i // nx

        # Diagonal entry
        rows.append(local_i)
        cols.append(global_i)
        vals.append(4.0)

        # Left neighbor
        if ix > 0:
            rows.append(local_i)
            cols.append(global_i - 1)
            vals.append(-1.0)

        # Right neighbor
        if ix < nx - 1:
            rows.append(local_i)
            cols.append(global_i + 1)
            vals.append(-1.0)

        # Bottom neighbor
        if iy > 0:
            rows.append(local_i)
            cols.append(global_i - nx)
            vals.append(-1.0)

        # Top neighbor
        if iy < ny - 1:
            rows.append(local_i)
            cols.append(global_i + nx)
            vals.append(-1.0)

    # Convert JAX dtype to NumPy dtype
    np_dtype: type = np.float32 if dtype == jnp.float32 else np.float64

    A_local_scipy = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(n_local, n),
        dtype=np_dtype,
    )

    A_local = jsp.BCSR(
        (
            jnp.array(A_local_scipy.data),
            jnp.array(A_local_scipy.indices, dtype=jnp.int32),
            jnp.array(A_local_scipy.indptr, dtype=jnp.int32),
        ),
        shape=(n_local, n),
    )

    return A_local, row_start, row_end


def random_matrix_distributed(
    n_global: int,
    rank: int,
    nranks: int,
    density: float = 0.01,
    dtype: DTypeLike = jnp.float32,
    seed: int = 0,
) -> tuple[jsp.BCSR, int, int]:
    """Create a distributed random sparse matrix in BCSR format for MPI.

    Args:
        n_global: Global matrix size (n_global x n_global)
        rank: MPI rank (0-indexed)
        nranks: Total number of MPI ranks
        density: Density of non-zero entries (default 0.01)
        dtype: Data type for matrix values (default jnp.float32)
        seed: Random seed for reproducibility (default 0)

    Returns:
        A_local: Local BCSR matrix partition (JAX)
        row_start: Starting row index (global)
        row_end: Ending row index (global, exclusive)
    """
    # Row-based partitioning
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)

    np_dtype: type = np.float32 if dtype == jnp.float32 else np.float64
    # Offset seed for each rank for reproducibility
    rng = np.random.default_rng(seed + rank)
    A_local = cast(
        sp.csr_matrix,
        sp.random(
            n_local,
            n_global,
            density=density,
            format="csr",
            dtype=np_dtype,
            random_state=rng,
        ),
    )
    # Make it diagonally dominant for better conditioning
    diag = abs(A_local).sum(axis=1).A1 + 1.0
    A_local.setdiag(diag, k=row_start)

    values = jnp.array(A_local.data)
    indices = jnp.array(A_local.indices, dtype=jnp.int32)
    indptr = jnp.array(A_local.indptr, dtype=jnp.int32)

    return (
        jsp.BCSR((values, indices, indptr), shape=(n_local, n_global)),
        row_start,
        row_end,
    )


def tridiagonal_operator(diagonal_value: float = 2.0) -> Callable:
    """Create a tridiagonal operator with [-1, diagonal_value, -1] pattern."""
    kernel = jnp.array([-1.0, diagonal_value, -1.0])
    matvec = lambda x: jnp.convolve(x, kernel, mode="same")
    return matvec


def poisson_operator(skew: float = 0.0) -> Callable:
    """Create a 2D Poisson operator (flat input).

    The operator represents the discretization of -Δu + skew * (∂u/∂x + ∂u/∂y)
    on a regular grid with standard 5-point stencil.

    Args:
        skew: Skew-symmetric coefficient (default 0.0 for symmetric Poisson)
              Non-zero values add convection-like terms, making the operator non-symmetric.

    Returns:
        Callable operator that applies the Poisson stencil to a flattened 2D array
    """
    # Create kernel with skew parameter
    # Standard symmetric: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    # With skew: left/top get -1-skew/2, right/bottom get -1+skew/2
    kernel = jnp.array(
        [
            [0.0, -1.0 - skew / 2.0, 0.0],
            [-1.0 - skew / 2.0, 4.0, -1.0 + skew / 2.0],
            [0.0, -1.0 + skew / 2.0, 0.0],
        ],
        dtype=jnp.float32,
    )

    def matvec(u_flat):
        size = u_flat.shape[0]
        n = int(size**0.5 + 0.5)

        if n * n != size:
            raise ValueError(f"Input size {size} is not a perfect square (n^2).")

        u = u_flat.reshape((n, n))
        # Boundary='fill', fillvalue=0.0 corresponds to Dirichlet BCs
        Au = jax.scipy.signal.convolve2d(
            u, kernel, mode="same", boundary="fill", fillvalue=0.0
        )
        return Au.ravel()

    return matvec


def poisson_operator_distributed(
    n_side: int, row_start: int, row_end: int, skew: float = 0.0
) -> Callable:
    """
    Create a distributed Poisson operator.

    The operator takes a global vector u, applies the Poisson stencil,
    and returns the local portion of the result (rows [row_start, row_end)).

    Args:
        n_side: Grid size (results in n_side^2 global size)
        row_start: Starting global row index for this rank
        row_end: Ending global row index for this rank (exclusive)
        skew: Skew-symmetric coefficient (default 0.0)

    Returns:
        Callable operator u_global_flat -> local_segment_flat
    """
    # Create kernel with skew parameter
    # Standard symmetric: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    kernel = jnp.array(
        [
            [0.0, -1.0 - skew / 2.0, 0.0],
            [-1.0 - skew / 2.0, 4.0, -1.0 + skew / 2.0],
            [0.0, -1.0 + skew / 2.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    def matvec(u_flat):
        # Reshape flat global vector to 2D grid
        u = u_flat.reshape((n_side, n_side))

        # Apply convolution with zero padding (Dirichlet BCs)
        Au = jax.scipy.signal.convolve2d(
            u, kernel, mode="same", boundary="fill", fillvalue=0.0
        )

        # Flatten and extract local rows
        return Au.ravel()[row_start:row_end]

    return matvec


def convection_diffusion_matrix_2d(
    n: int, epsilon: float = 1.0, theta: float = 0.0, velocity: float = 100.0
) -> jsp.BCSR:
    """Create a 2D convection-diffusion matrix on an n×n grid.

    Equation: -ε Δu + v⋅∇u = f

    The velocity field v is constant magnitude `velocity` rotated by angle `theta`.
    v = (vx, vy) = (velocity * cos(theta), velocity * sin(theta))

    Discretization:
    - Diffusion: Standard 5-point central difference.
    - Convection: First-order Upwind differencing for stability at high Peclet numbers.
    - Grid spacing h = 1/(n-1).

    Args:
        n: Grid size (n x n nodes)
        epsilon: Diffusion coefficient
        theta: Flow angle in radians
        velocity: Flow velocity magnitude

    Returns:
        JAX BCSR matrix
    """
    n2 = n * n
    h = 1.0 / (n - 1)  # Grid spacing

    vx = velocity * jnp.cos(theta)
    vy = velocity * jnp.sin(theta)

    # Grid construction
    i_grid, j_grid = jnp.meshgrid(jnp.arange(n), jnp.arange(n), indexing="ij")
    i_flat = i_grid.ravel()
    j_flat = j_grid.ravel()
    row_indices = i_flat * n + j_flat

    # Initialize coefficients
    diag_vals = jnp.zeros(n2, dtype=jnp.float32)
    left_vals = jnp.zeros(n2, dtype=jnp.float32)
    right_vals = jnp.zeros(n2, dtype=jnp.float32)
    top_vals = jnp.zeros(n2, dtype=jnp.float32)
    bottom_vals = jnp.zeros(n2, dtype=jnp.float32)

    # Diffusion (Standard 5-point central difference)
    # Coefficients scaled by h^2 to keep matrix well-scaled
    inv_h2 = 1.0 / (h * h)

    diag_vals += 4.0 * epsilon * inv_h2
    left_vals -= epsilon * inv_h2
    right_vals -= epsilon * inv_h2
    top_vals -= epsilon * inv_h2
    bottom_vals -= epsilon * inv_h2

    # Convection (First-order Upwind)
    inv_h = 1.0 / h

    # X-direction
    if vx > 0:
        diag_vals += vx * inv_h
        left_vals -= vx * inv_h
    else:
        diag_vals -= vx * inv_h
        right_vals += vx * inv_h

    # Y-direction
    if vy > 0:
        diag_vals += vy * inv_h
        top_vals -= vy * inv_h
    else:
        diag_vals -= vy * inv_h
        bottom_vals += vy * inv_h

    # Assemble Matrix
    # ----------------
    masks = [
        j_flat > 0,  # Left
        j_flat < n - 1,  # Right
        i_flat > 0,  # Top (i-1)
        i_flat < n - 1,  # Bottom (i+1)
    ]
    offsets = [-1, 1, -n, n]
    coeffs = [left_vals, right_vals, top_vals, bottom_vals]

    all_rows = [row_indices]
    all_cols = [row_indices]
    all_vals = [diag_vals]

    for mask, offset, val_arr in zip(masks, offsets, coeffs):
        all_rows.append(row_indices[mask])
        all_cols.append(row_indices[mask] + offset)
        all_vals.append(val_arr[mask])

    rows = jnp.concatenate(all_rows)
    cols = jnp.concatenate(all_cols)
    vals = jnp.concatenate(all_vals)

    # Sort
    sort_idx = jnp.lexsort((cols, rows))
    rows = rows[sort_idx]
    cols = cols[sort_idx]
    vals = vals[sort_idx]

    # Indptr
    indptr = jnp.zeros(n2 + 1, dtype=jnp.int32)
    row_counts = jnp.bincount(rows, length=n2)
    indptr = indptr.at[1:].set(jnp.cumsum(row_counts))

    return jsp.BCSR((vals, cols, indptr), shape=(n2, n2))


def rhs_ones(n: int, dtype=jnp.float32) -> jax.Array:
    """Create a constant RHS vector of ones.

    Args:
        n: Vector length
        dtype: Data type of the vector

    Returns:
        JAX array of ones
    """
    return jnp.ones(n, dtype=dtype)


def rhs_linear(n: int, dtype=jnp.float32) -> jax.Array:
    """Create a linearly increasing RHS vector.

    Args:
        n: Vector length
        dtype: Data type of the vector

    Returns:
        JAX array with values linearly spaced from 0 to 1
    """
    return jnp.linspace(0, 1, n, dtype=dtype)


def rhs_random(n: int, seed: int = 0, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """Create a random RHS vector.

    Args:
        n: Vector length
        seed: Random seed for reproducibility
        dtype: Data type of the vector

    Returns:
        JAX array with random normal values
    """
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (n,), dtype=dtype)
