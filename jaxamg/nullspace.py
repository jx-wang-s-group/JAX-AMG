"""Null-space handling for singular systems.

Orthogonal projections around the solver primitive, as JAX ops::

    forward :  b' = b - M(MᵀM)⁻¹Mᵀb     x = solve(A, b')     x' = x - N(NᵀN)⁻¹Nᵀx
    backward:  g' = g - N(NᵀN)⁻¹Nᵀg     λ = solve(Aᵀ, g')    b̄ = λ - M(MᵀM)⁻¹Mᵀλ

``N`` spans ``null(A)``, ``M`` spans ``null(Aᵀ)``. The backward line is JAX's
transpose of the forward line, so the forward returns ``A⁺b`` and ``jax.grad``
returns ``(Aᵀ)⁺g``. For nonsymmetric ``A`` the two null spaces differ (e.g.
``A = D⁻¹L``: ``null(A) = span(1)``, ``null(Aᵀ) = span(V)``).
"""

import warnings
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

__all__ = ["NullSpaceWarning"]

NullSpaceSpec = ArrayLike | str | None


class NullSpaceWarning(UserWarning):
    """Null-space issue in `jaxamg.solve`: a singular matrix without a declared
    `nullspace`, a missing `nullspace`/`transpose_nullspace`, a basis that is
    not a null space, or a `DENSE_LU_SOLVER` coarse solve on a singular system."""


_SINGULAR_MSG = (
    "A·1 = 0: the system is singular. Pass nullspace='constant' (and "
    "transpose_nullspace for nonsymmetric A); otherwise adjoint solves may "
    "stall on an inconsistent right-hand side."
)

_MISSING_TRANSPOSE_MSG = (
    "nullspace given without transpose_nullspace for a matrix not marked "
    "symmetric: b is not projected onto range(A) and the gradient w.r.t. b is "
    "defined only up to a component in null(Aᵀ). Pass transpose_nullspace, or "
    "mark A symmetric with with_cache(A, is_symmetric=True)."
)

_MISSING_NULLSPACE_MSG = (
    "transpose_nullspace given without nullspace for a matrix not marked "
    "symmetric: the solution is not pinned (its null(A) component is whatever "
    "the solver returns) and the adjoint right-hand side is not projected. "
    "Pass nullspace, or mark A symmetric with with_cache(A, is_symmetric=True)."
)

_DENSE_LU_MSG = (
    "DENSE_LU_SOLVER coarse solve with a declared null space can diverge. Use "
    "coarse smoother sweeps (the default when a null space is declared); for "
    "cached MPI metadata pass singular=True to cache_mpi_metadata()."
)


def _detect_rtol(dtype: DTypeLike) -> float:
    return 100.0 * float(jnp.finfo(dtype).eps)


def _verify_rtol(dtype: DTypeLike) -> float:
    return float(np.sqrt(jnp.finfo(dtype).eps))


def _is_traced(*arrays: Any) -> bool:
    return any(isinstance(a, jax.core.Tracer) for a in arrays)


class _Memo:
    """Verdicts keyed by array identity; entries vanish with the arrays."""

    def __init__(self) -> None:
        self._data: dict[tuple[int, ...], Any] = {}

    def get(self, *arrays: Any) -> Any:
        return self._data.get(tuple(id(a) for a in arrays))

    def put(self, value: Any, *arrays: Any) -> None:
        key = tuple(id(a) for a in arrays)
        try:
            for a in arrays:
                weakref.finalize(a, self._data.pop, key, None)
        except TypeError:
            return  # not weak-referenceable: do not memoize
        self._data[key] = value


_singular_memo = _Memo()
_verified_memo = _Memo()


def _all_ranks(flag: bool, comm: "Comm | None") -> bool:
    """Collective AND, so memo hits cannot desynchronize the checks below."""
    if comm is None:
        return flag
    from mpi4py import MPI

    return bool(comm.allreduce(flag, op=MPI.LAND))


def _global_max(value: float, comm: "Comm | None") -> float:
    if comm is None:
        return value
    from mpi4py import MPI

    return comm.allreduce(value, op=MPI.MAX)


def as_nullspace_basis(
    spec: NullSpaceSpec, n: int, dtype: DTypeLike, name: str
) -> jax.Array | None:
    """Normalize ``"constant"`` / vector / ``(n, k)`` array to ``(n, k)`` (or None)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec.lower() != "constant":
            raise ValueError(
                f"{name} must be 'constant', an array, or None; got {spec!r}"
            )
        # numpy-backed so it stays concrete when solve() is traced
        return jnp.asarray(np.ones((n, 1), dtype=dtype))
    basis = jnp.asarray(spec)
    if basis.ndim == 1:
        basis = basis[:, None]
    if basis.ndim != 2 or basis.shape[0] != n or not 0 < basis.shape[1] <= n:
        raise ValueError(
            f"{name} must be a vector of length {n} or an ({n}, k) array with "
            f"k ≤ {n}; got shape {basis.shape}"
        )
    if basis.dtype != dtype:
        basis = basis.astype(dtype)
    return basis


def validate_basis(basis: jax.Array, name: str, comm: "Comm | None" = None) -> None:
    """Raise unless the (concrete) basis is finite with independent columns.
    Uses the ``k × k`` Gram matrix, reduced across ranks in MPI mode."""
    if _is_traced(basis):
        return
    with jax.ensure_compile_time_eval():
        gram = np.asarray(basis.T @ basis, dtype=np.float64)
    if comm is not None:
        from mpi4py import MPI

        gram = comm.allreduce(gram, op=MPI.SUM)
    if not np.isfinite(gram).all():
        raise ValueError(f"{name} contains non-finite values")
    ev = np.linalg.eigvalsh(gram)
    if ev[-1] <= 0 or ev[0] <= 1e3 * float(jnp.finfo(basis.dtype).eps) * ev[-1]:
        raise ValueError(f"{name} has zero or linearly dependent columns")


def project_out(
    v: jax.Array,
    basis: jax.Array,
    reduce_sum: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """``v - B(BᵀB)⁻¹Bᵀv``. ``reduce_sum`` (MPI) reduces the stacked
    ``[Bᵀv, vec(BᵀB)]`` in one collective."""
    k = basis.shape[1]
    stats = jnp.concatenate([basis.T @ v, (basis.T @ basis).reshape(-1)])
    if reduce_sum is not None:
        stats = reduce_sum(stats)
    coeffs = stats[:k]
    gram = stats[k:].reshape(k, k)
    if k == 1:
        return v - basis[:, 0] * (coeffs[0] / gram[0, 0])
    return v - basis @ jnp.linalg.solve(gram, coeffs)


def relative_norm(
    d: jax.Array,
    v: jax.Array,
    reduce_sum: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """``‖d‖/‖v‖`` (0 when ``v = 0``), reduced across ranks in MPI mode."""
    stats = jnp.stack([jnp.sum(d * d), jnp.sum(v * v)])
    if reduce_sum is not None:
        stats = reduce_sum(stats)
    nonzero = stats[1] > 0
    denom = jnp.where(nonzero, stats[1], 1.0)
    return jnp.where(nonzero, jnp.sqrt(stats[0] / denom), 0.0)


def make_mpi_reduce_sum(comm: "Comm") -> Callable[[jax.Array], jax.Array]:
    """Sum all-reduce whose VJP is also a sum all-reduce.

    The reduced value is used in every rank's rows, so its cotangent is the sum
    of the per-rank contributions; mpi4jax's own transpose rule (identity)
    would only project out each rank's local component in the backward pass.
    """
    import mpi4jax
    from mpi4py import MPI

    def allreduce(x: jax.Array) -> jax.Array:
        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)

    @jax.custom_vjp
    def reduce_sum(x: jax.Array) -> jax.Array:
        return allreduce(x)

    def fwd(x: jax.Array) -> tuple[jax.Array, None]:
        return allreduce(x), None

    def bwd(_: None, g: jax.Array) -> tuple[jax.Array]:
        return (allreduce(g),)

    reduce_sum.defvjp(fwd, bwd)
    return reduce_sum


def row_index(A: jsp.BCSR) -> jax.Array:
    """Row index of every stored entry."""
    n = A.shape[0]
    row_lengths = A.indptr[1:] - A.indptr[:-1]
    return jnp.repeat(
        jnp.arange(n, dtype=jnp.int32), row_lengths, total_repeat_length=len(A.data)
    )


def _row_sums(A: jsp.BCSR) -> tuple[jax.Array, jax.Array]:
    """Row sums (``A·1``) and row absolute sums."""
    rows = row_index(A)
    n = A.shape[0]
    return (
        jax.ops.segment_sum(A.data, rows, num_segments=n),
        jax.ops.segment_sum(jnp.abs(A.data), rows, num_segments=n),
    )


def warn_if_singular(
    A: jsp.BCSR, comm: "Comm | None" = None, stacklevel: int = 2
) -> None:
    """Warn when ``A·1 = 0`` to round-off. Skipped for traced values; in MPI
    mode every rank must call it. Memoized per matrix buffer."""
    if _is_traced(A.data, A.indptr) or A.shape[0] == 0:
        return
    singular = _singular_memo.get(A.data)
    if not _all_ranks(singular is not None, comm):
        # Under a jit trace every op is staged; evaluate the check eagerly.
        with jax.ensure_compile_time_eval():
            rowsum, absrow = _row_sums(A)
            num = float(jnp.max(jnp.abs(rowsum)))
            den = float(jnp.max(absrow))
        num = _global_max(num, comm)
        den = _global_max(den, comm)
        singular = den > 0 and num <= _detect_rtol(A.data.dtype) * den
        _singular_memo.put(singular, A.data)
    if singular:
        warnings.warn(_SINGULAR_MSG, NullSpaceWarning, stacklevel=stacklevel)


def verify_nullspace(
    A: jsp.BCSR,
    basis: jax.Array,
    matvec: Callable[[jax.Array], jax.Array],
    name: str,
    comm: "Comm | None" = None,
    stacklevel: int = 2,
) -> None:
    """Warn unless ``max|matvec(B)| ≤ sqrt(eps)·‖A‖∞·max|B|``. Skipped for
    traced values; ``matvec`` must be collective in MPI mode. Memoized per
    (matrix buffer, basis)."""
    if _is_traced(A.data, A.indptr, basis) or A.shape[0] == 0:
        return
    ratio = _verified_memo.get(A.data, basis)
    if not _all_ranks(ratio is not None, comm):
        with jax.ensure_compile_time_eval():
            _, absrow = _row_sums(A)
            a_norm = float(jnp.max(absrow))
            b_norm = float(jnp.max(jnp.abs(basis)))
            residual = float(jnp.max(jnp.abs(matvec(basis))))
        scale = _global_max(a_norm, comm) * _global_max(b_norm, comm)
        residual = _global_max(residual, comm)
        ratio = residual / scale if scale > 0 else 0.0
        _verified_memo.put(ratio, A.data, basis)
    tol = _verify_rtol(A.data.dtype)
    if ratio > tol:
        target = "Aᵀ" if name == "transpose_nullspace" else "A"
        warnings.warn(
            f"{name} does not appear to span the null space of {target}: "
            f"relative residual {ratio:.2e} (expected ≤ {tol:.1e}).",
            NullSpaceWarning,
            stacklevel=stacklevel,
        )
