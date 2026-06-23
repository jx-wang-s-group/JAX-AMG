"""
Helper functions for BCSR matrix operations.
"""

import contextlib
import os
from collections.abc import Callable
from typing import cast

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from jax import core
from jax.typing import ArrayLike, DTypeLike

Matrix = ArrayLike | jsp.JAXSparse | sp.spmatrix
MatrixOrOperator = Matrix | Callable


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@contextlib.contextmanager
def temp_enable_x64():
    """Context manager to temporarily enable x64 mode."""
    original_x64 = jax.config.read("jax_enable_x64")
    if not original_x64:
        jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        if not original_x64:
            jax.config.update("jax_enable_x64", False)


def ensure_int64_array(data: ArrayLike) -> jax.Array:
    """Create a JAX array with int64 dtype, temporarily enabling x64 if needed.

    This avoids truncation warnings when running in float32 mode (default JAX).

    Args:
        data: Data to convert to int64 array

    Returns:
        Device array with int64 dtype
    """
    with temp_enable_x64():
        return jnp.array(data, dtype=jnp.int64)


def to_scipy(A: jsp.BCSR, format: str = "csr") -> sp.spmatrix:
    """Convert a JAX BCSR matrix to Scipy sparse matrix format.

    Args:
        A: JAX BCSR matrix
        format: Scipy sparse matrix format (default "csr")
                Supported: "csr", "csc", "coo", "lil", "dok", "bsr"

    Returns:
        Scipy sparse matrix in the specified format
    """
    # First convert to CSR
    data = np.asarray(A.data).ravel()
    indices = np.asarray(A.indices).astype(int)
    indptr = np.asarray(A.indptr).astype(int)
    nrow, ncol = A.shape
    A_csr = sp.csr_matrix((data, indices, indptr), shape=(nrow, ncol))

    # Convert to requested format
    if format == "csr":
        return A_csr
    elif format == "csc":
        return A_csr.tocsc()
    elif format == "coo":
        return A_csr.tocoo()
    elif format == "lil":
        return A_csr.tolil()
    elif format == "dok":
        return A_csr.todok()
    elif format == "bsr":
        return A_csr.tobsr()
    else:
        raise ValueError(
            f"Unsupported format: {format}. Supported formats: csr, csc, coo, lil, dok, bsr"
        )


def get_preferred_dtype(A: MatrixOrOperator, b: ArrayLike) -> DTypeLike:
    """
    Determine the preferred precision (float32 or float64) for the solver.

    Logic:
    - Default is float32.
    - If b is float64, use float64.
    - If A provides float64 data, use float64 (promoting b if necessary).
    """

    target_dtype = getattr(b, "dtype", jnp.float32)

    a_dtype = getattr(A, "dtype", None)
    a_data = getattr(A, "data", None)
    a_data_dtype = getattr(a_data, "dtype", None)

    # Check A.dtype
    if a_dtype is not None:
        # e.g. JAX arrays, sparse matrices
        if a_dtype == jnp.float64 or a_dtype == np.float64:
            target_dtype = jnp.float64

    # Check A.data.dtype
    if a_data_dtype is not None:
        # e.g. BCOO, BCSR
        if a_data_dtype == jnp.float64 or a_data_dtype == np.float64:
            target_dtype = jnp.float64

    return target_dtype


def _ensure_bcsr_properties(
    A: jsp.BCSR, target_dtype: DTypeLike, use_int64_indices: bool = False
) -> jsp.BCSR:
    """Ensure BCSR matrix has correct data type and index types.

    Args:
        A: BCSR matrix to normalize
        target_dtype: Target dtype for matrix values (float32 or float64)
        use_int64_indices: If True, use int64 for column indices (required for MPI).
                          If False (default), use int32 (for single-GPU mode).
    """
    # check data type
    if A.data.dtype != target_dtype:
        A = jsp.BCSR(
            (A.data.astype(target_dtype), A.indices, A.indptr),
            shape=A.shape,
        )

    # check indices type
    if use_int64_indices:
        # For MPI: need int64 indices
        # Temporarily enable X64 mode to allow int64 arrays
        if A.indices.dtype != np.int64:
            try:
                # Use context manager to keep x64 enabled during BCSR construction
                # This prevents JAX from truncating the indices back to int32
                with temp_enable_x64():
                    indices_int64 = jnp.array(A.indices, dtype=jnp.int64)
                    A = jsp.BCSR(
                        (A.data, indices_int64, A.indptr),
                        shape=A.shape,
                    )
            except Exception as e:
                raise ValueError(
                    f"Matrix column indices must be int64 for MPI, got {A.indices.dtype}. Error: {e}"
                )
    else:
        # For single-GPU: use int32
        if A.indices.dtype != jnp.int32:
            try:
                A = jsp.BCSR(
                    (A.data, A.indices.astype(jnp.int32), A.indptr),
                    shape=A.shape,
                )
            except Exception:
                raise ValueError(
                    f"Matrix column indices must be int32, got {A.indices.dtype}."
                )

    # check indptr type (always int32)
    if A.indptr.dtype != jnp.int32:
        try:
            A = jsp.BCSR(
                (A.data, A.indices, A.indptr.astype(jnp.int32)),
                shape=A.shape,
            )
        except Exception:
            raise ValueError(
                f"Matrix row pointers must be int32, got {A.indptr.dtype}."
            )

    if A.data.dtype not in [jnp.float32, jnp.float64]:
        raise ValueError(
            f"Matrix values must be float32 or float64, got {A.data.dtype}."
        )

    return A


def to_bcsr_matrix(
    A: MatrixOrOperator, b: ArrayLike, use_int64_indices: bool = False
) -> jsp.BCSR:
    """
    Normalize input 'A' to a BCSR matrix.

    Supports multiple input formats, including BCSR, BCOO, SciPy sparse (CSR, CSC, COO), dense arrays (NumPy, JAX), and callable operators.

    Args:
        A: Input matrix or callable operator
        b: RHS vector (required for callable operators)
        use_int64_indices: If True, use int64 for column indices (required for MPI).
                          If False (default), use int32 (for single-GPU mode).
    """

    A_bcsr: jsp.BCSR

    # 1. Handle Callables (materialize via graph coloring)
    if callable(A):
        A = cast(Callable, A)
        from .sparsity import cache_coloring, materialize_sparse_matrix

        # Check for cached coloring info attached to the callable
        cached_info = getattr(A, "_coloring_info", None)

        if b is None:
            raise TypeError("Callable A requires RHS b to infer size.")

        b = jnp.asarray(b)

        if b.ndim != 1:
            raise ValueError(f"RHS b must be 1D, got shape {b.shape}.")

        n_rows = int(b.shape[0])

        # Determine shape
        if cached_info is not None:
            shape = cached_info[4]
            # CHANGED
            # if shape[0] != n_rows:
            #     raise ValueError(
            #         f"Cached operator has {shape[0]} rows, but RHS b has {n_rows} elements."
            #     )
        else:
            shape = (n_rows, n_rows)

        is_jit = isinstance(b, core.Tracer)

        if cached_info is None:
            if is_jit:
                # Inside JIT without cache: Impossible to determine sparsity dynamically.
                raise ValueError(
                    "Callable operators must be pre-scanned before JIT compilation to determine sparsity.\n"
                    "Call solve(A, b) once outside of JIT to compute and cache the sparsity pattern."
                )

            # Outside JIT: detect the sparsity + colouring via cache_coloring,
            # which traces the operator's jaxpr (exact, in one trace) and falls
            # back to one-hot probing only when tracing is unavailable. It also
            # attaches `_coloring_info` to A for reuse.
            cached_info = cache_coloring(A, shape)

        rows, cols, column_colors, n_colors, cached_shape = cached_info

        if shape != cached_shape:
            raise ValueError(f"Operator shape changed from {cached_shape} to {shape}.")

        # Materialize using graph coloring (works efficiently inside JIT)
        # Note: materialize_sparse_matrix already returns a BCSR
        A_bcsr = materialize_sparse_matrix(
            A, shape, rows, cols, column_colors, n_colors
        )

    # 2. Convert to BCSR from other formats
    target_dtype = get_preferred_dtype(A, b)

    if isinstance(A, jsp.BCSR):
        A_bcsr = A
    elif isinstance(A, jsp.BCOO):
        A_bcsr = jsp.BCSR.from_bcoo(A)
    elif sp.issparse(A):
        # Use numpy dtype for scipy conversion
        np_dtype = np.float64 if target_dtype == jnp.float64 else np.float32
        if isinstance(
            A,
            (
                sp.csr_matrix,
                sp.csc_matrix,
                sp.coo_matrix,
                sp.bsr_matrix,
                sp.lil_matrix,
                sp.dok_matrix,
            ),
        ):
            A_bcsr = jsp.BCSR.from_scipy_sparse(A.astype(np_dtype))
        else:
            raise TypeError(
                f"Unsupported scipy sparse matrix type: {type(A).__name__}."
            )
    elif isinstance(A, (np.ndarray, jnp.ndarray)):
        if A.ndim != 2:
            raise ValueError(f"Dense matrix must be 2D, got shape {A.shape}")
        if isinstance(A, np.ndarray):
            A = jnp.array(
                A
            )  # Ensure JAX array before fromdense if needed, or just let fromdense handle it
        A_bcsr = jsp.BCSR.fromdense(A)
    elif callable(A):
        pass  # Already handled above
    else:
        raise TypeError(
            f"Matrix A must be one of: BCSR, BCOO, SciPy sparse, dense array (NumPy/JAX), or callable. "
            f"Got {type(A).__name__}."
        )

    # 3. Standardize properties
    A_bcsr = _ensure_bcsr_properties(
        A_bcsr, target_dtype, use_int64_indices=use_int64_indices
    )

    if A_bcsr.n_batch > 0:
        raise ValueError(
            f"jaxamg.solve does not support batched BCSR matrices (n_batch > 0). "
            f"Input matrix has n_batch={A_bcsr.n_batch}. Use jax.vmap for batched solves."
        )

    return A_bcsr


def format_amgx_stats(
    stats_str: str, filepath: str | os.PathLike, rank: int | None = None
) -> None:
    """Parse raw AmgX captured output and write a formatted stats file.

    Segments output into AMG Grid Statistics and Solver/Coarse Solver Iterations.
    In MPI mode pass ``rank``; only rank 0 writes the file.
    """
    if rank is not None and rank != 0:
        return

    import re as _re

    filename = filepath
    lines = stats_str.splitlines()

    # Segment raw lines into "grid" and "table" blocks.
    blocks: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("AMG Grid:"):
            block: list[str] = []
            while i < len(lines) and not (
                lines[i].strip().startswith("iter") and "residual" in lines[i]
            ):
                block.append(lines[i])
                i += 1
            blocks.append(("grid", block))
        elif s.startswith("iter") and "residual" in s:
            block = []
            in_summary = False
            while i < len(lines):
                s = lines[i].strip()
                if in_summary and (
                    not s or s.startswith("iter") or s.startswith("AMG Grid")
                ):
                    break
                if s.startswith("---"):
                    in_summary = True
                if s:
                    block.append(lines[i])
                i += 1
            blocks.append(("table", block))
        else:
            i += 1

    def _fmt_table(tb_lines: list[str], indent: str = "  ") -> list[str]:
        result = [
            indent + f"{'Iter':>6}  {'Mem (GB)':>13}  {'Residual':>15}  {'Rate':>8}",
            indent + "-" * 50,
        ]
        in_summary = False
        saw_first_sep = False
        for line in tb_lines:
            s = line.strip()
            if not s or (s.startswith("iter") and "residual" in s):
                continue
            if s.startswith("---"):
                if not saw_first_sep:
                    saw_first_sep = True
                else:
                    result.append(indent + "-" * 50)
                    in_summary = True
                continue
            if in_summary:
                if ":" in s:
                    key, _, val = s.partition(":")
                    result.append(indent + f"  {key.strip():<30}: {val.strip()}")
                else:
                    result.append(indent + "  " + s)
            else:
                if not saw_first_sep:
                    continue
                parts = s.split()
                label = parts[0]
                mem = parts[1] if len(parts) > 1 else ""
                resid = parts[2] if len(parts) > 2 else ""
                rate = parts[3] if len(parts) > 3 else ""
                result.append(indent + f"{label:>6}  {mem:>13}  {resid:>15}  {rate:>8}")
        return result

    out_lines: list[str] = []

    grid_blocks = [b for t, b in blocks if t == "grid"]
    if grid_blocks:
        out_lines += ["=" * 60, "  AMG GRID STATISTICS", "=" * 60]
        for gb in grid_blocks:
            for line in gb:
                s = line.strip()
                if not s or s.startswith("AMG Grid:"):
                    continue
                m = _re.match(
                    r"^\s*(\d+)\(([DC])\)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", line
                )
                if m:
                    lvl, t, rows, nnz, parts, sprsty, mem = m.groups()
                    out_lines.append(
                        f"  {lvl+'('+t+')':>5}  {rows:>12}  {nnz:>16}  {parts:>5}  {sprsty:>9}  {mem:>12}"
                    )
                elif s.startswith("LVL"):
                    out_lines.append(
                        f"  {'LVL':>5}  {'ROWS':>12}  {'NNZ':>16}  {'PARTS':>5}  {'SPRSTY':>9}  {'Mem (GB)':>12}"
                    )
                else:
                    out_lines.append("  " + s)
        out_lines.append("")

    table_blocks = [b for t, b in blocks if t == "table"]
    if table_blocks:
        tb = table_blocks[0]
        out_lines += ["=" * 60, "  SOLVER ITERATIONS", "=" * 60]
        out_lines.extend(_fmt_table(tb))
        out_lines.append("")

    with open(filename, "w") as f:
        f.write("\n".join(out_lines) + "\n")
