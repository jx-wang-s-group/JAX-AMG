"""Null-space handling for singular systems.

Orthogonal projections around the solver primitive, as JAX ops::

    forward :  b' = b - M(MᵀM)⁻¹Mᵀb     x = solve(A, b')     x' = x - N(NᵀN)⁻¹Nᵀx
    backward:  g' = g - N(NᵀN)⁻¹Nᵀg     λ = solve(Aᵀ, g')    b̄ = λ - M(MᵀM)⁻¹Mᵀλ

``N`` spans ``null(A)``, ``M`` spans ``null(Aᵀ)``. The backward line is JAX's
transpose of the forward line, so the forward returns ``A⁺b`` and ``jax.grad``
returns ``(Aᵀ)⁺g``. For nonsymmetric ``A`` the two null spaces differ (e.g.
``A = D⁻¹L``: ``null(A) = span(1)``, ``null(Aᵀ) = span(V)``).

**Labels** (a disconnected domain). With one integer label per row, every
declared column applies separately on each label's rows: the null space is
block diagonal over the labels, ``N_l = N·1_l`` per label ``l``, and rows
labelled -1 belong to none. The projection is the same formula per label:
``BᵀB`` and ``Bᵀv`` as segment sums over the labels, and a ``k × k`` solve
per label (a division when ``k = 1``). Scale statistics are reduced first,
then the products of the scaled columns; both use the caller's sum map. A label
whose columns vanish on its rows (no rows, or all of them -1 in this solve)
is skipped exactly. The label count is static, so shapes are fixed.
"""

import warnings
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

__all__ = ["NullSpaceWarning"]

NullSpaceSpec = ArrayLike | str | None
LabelSpec = tuple[int, ArrayLike] | None
Labels = tuple[int, jax.Array | np.ndarray]


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


def _skip_on_any_rank(skip: bool, comm: "Comm | None") -> bool:
    """Whether any rank skips an eager check: the checks below communicate, so
    a rank-local reason to skip (traced values here, concrete elsewhere) must
    make every rank skip."""
    return not _all_ranks(not skip, comm)


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
    spec: NullSpaceSpec,
    n: int,
    dtype: DTypeLike,
    name: str,
    n_global: int | None = None,
) -> jax.Array | None:
    """Normalize ``"constant"`` / vector / ``(n, k)`` array to ``(n, k)`` (or None).

    ``n_global`` bounds ``k`` for a distributed basis, whose local slice may
    have fewer rows than columns."""
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
    max_k = n if n_global is None else n_global
    if basis.ndim != 2 or basis.shape[0] != n or not 0 < basis.shape[1] <= max_k:
        raise ValueError(
            f"{name} must be a vector of length {n} or an ({n}, k) array with "
            f"k ≤ {max_k}; got shape {basis.shape}"
        )
    if basis.dtype != dtype:
        basis = basis.astype(dtype)
    return basis


def validate_label_values(
    labels: Labels, comm: "Comm | None" = None, name: str = "labels"
) -> None:
    """Range-check concrete local labels, rejecting on every rank together.

    Inspect values before narrowing to int32. If any rank is tracing its
    labels, all ranks skip this eager check.
    """
    count, rows = labels
    if _skip_on_any_rank(_is_traced(rows), comm):
        return
    with jax.ensure_compile_time_eval():
        values = np.asarray(rows)
    valid = not np.any((values < -1) | (values >= count))
    if not _all_ranks(bool(valid), comm):
        raise ValueError(f"{name} must lie in [-1, {count}) on every rank")


def as_labels(
    spec: LabelSpec, n: int, name: str = "labels", comm: "Comm | None" = None
) -> Labels | None:
    """Normalize the pair ``(count, labels)`` (as
    ``scipy.sparse.csgraph.connected_components`` returns it: a static label
    count, and an integer vector of ``n`` rows, local rows in MPI mode, with
    values in ``[-1, count)``) to int32 labels, or None. The labels may be
    traced; concrete ones are range-checked."""
    if spec is None:
        return None
    if not isinstance(spec, tuple) or len(spec) != 2:
        raise ValueError(f"{name} must be a (count, labels) pair")
    count, rows = spec
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"{name}' count must be a Python integer")
    count = int(count)
    if count < 1:
        raise ValueError(f"{name}' count must be positive; got {count}")
    # Host labels stay on the host (concrete under a trace as well).
    if not isinstance(rows, jax.Array):
        rows = np.asarray(rows)
    if rows.shape != (n,) or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(
            f"{name} must be an integer vector of length {n}; got "
            f"{rows.dtype} of shape {rows.shape}"
        )
    validate_label_values((count, rows), comm, name)
    return count, rows.astype(np.int32)


def restrict(basis: jax.Array, labels: Labels) -> jax.Array:
    """The declared columns on the labelled rows (zero on rows labelled -1)."""
    count, rows = labels
    return jnp.where(((rows >= 0) & (rows < count))[:, None], basis, 0)


def _segments(labels: Labels) -> jax.Array:
    """Each row's label, ``count`` (dropped by segment sums) where it has none."""
    count, rows = labels
    return jnp.where((rows >= 0) & (rows < count), rows, count)


def _pairs(k: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(k) for j in range(i, k)]


def label_scale_stats(basis: jax.Array, labels: Labels) -> jax.Array:
    """Additive scale statistics, safe before squaring extreme basis values.

    Each rank contributes sqrt(max(abs(B))) and a nonzero indicator per
    label and column. Roots keep the sum in range even for extreme finite
    inputs. Only addition is needed, including with caller-numbered labels.
    """
    count, _ = labels
    B = jax.lax.stop_gradient(restrict(basis, labels))
    if count == 1:
        largest = jnp.max(jnp.abs(B), axis=0, initial=0.0, keepdims=True)
    else:
        largest = jnp.maximum(
            jax.ops.segment_max(jnp.abs(B), _segments(labels), num_segments=count),
            0,
        )
    return jnp.concatenate([jnp.sqrt(largest), (largest > 0).astype(B.dtype)], axis=1)


def scale_label_basis(basis: jax.Array, labels: Labels, stats: jax.Array) -> jax.Array:
    """Rescale extreme columns separately per label, from reduced statistics.

    For p contributing ranks, s = (sum sqrt(local_max))**2 / p lies between
    global_max/p and p*global_max. Form its exponent without squaring the
    magnitude itself; then use a power of two, shared by every participating
    rank. Ordinary scales and genuinely zero columns remain unchanged.
    """
    roots, counts = jnp.split(jax.lax.stop_gradient(stats), 2, axis=1)
    root_m, root_e = jnp.frexp(roots)
    count_m, count_e = jnp.frexp(jnp.maximum(counts, 1))
    mantissa, exponent = jnp.frexp(root_m * root_m / count_m)
    exponent = exponent + 2 * root_e - count_e
    limit = safe_exponent_limit(basis.dtype)
    extreme = (exponent <= -limit) | (exponent > limit + 1)
    extreme |= (exponent == limit + 1) & (mantissa > 0.5)
    shift = jnp.where(jnp.isfinite(roots) & (roots > 0) & extreme, exponent, 0)
    shift = jnp.concatenate([shift, jnp.zeros((1, basis.shape[1]), jnp.int32)])
    shift = shift[_segments(labels)]
    B = restrict(basis, labels)
    return jnp.where(shift != 0, jnp.ldexp(B, -shift), B)


def unit_label_basis(
    basis: jax.Array,
    labels: Labels,
    reduce_sum: Callable[[jax.Array], jax.Array] | None = None,
) -> jax.Array:
    """Scale-safe per-label columns using the same reduction as projections."""
    stats = label_scale_stats(basis, labels)
    if reduce_sum is not None:
        stats = reduce_sum(stats)
    return scale_label_basis(basis, labels, stats)


def label_sums(v: jax.Array, basis: jax.Array, labels: Labels) -> jax.Array:
    """This rank's per-label partial sums ``[Bᵀv, upper(BᵀB)]``, shape
    ``(count, k + k(k+1)/2)``, for `project_out` (summed over the ranks in
    between)."""
    count, _ = labels
    B = restrict(basis, labels)
    parts = [B * v[:, None]] + [
        (B[:, i] * B[:, j])[:, None] for i, j in _pairs(B.shape[1])
    ]
    columns = jnp.concatenate(parts, axis=1)
    if count == 1:
        # One label: plain sums (a scatter into one segment contends on GPUs).
        return jnp.sum(columns, axis=0, keepdims=True)
    return jax.ops.segment_sum(columns, _segments(labels), num_segments=count)


def remove_label_sums(
    v: jax.Array, basis: jax.Array, labels: Labels, sums: jax.Array
) -> jax.Array:
    """``v`` less each label's projection, from the labels' total sums
    (`label_sums` reduced over the ranks). Labels with vanishing columns are
    left out exactly."""
    k = basis.shape[1]
    coeffs, products = sums[:, :k], sums[:, k:]
    if k == 1:
        gram = products[:, 0]
        empty = gram == 0
        c = jnp.where(empty, 0, coeffs[:, 0] / jnp.where(empty, 1, gram))[:, None]
    else:
        gram = jnp.zeros((sums.shape[0], k, k), sums.dtype)
        for column, (i, j) in enumerate(_pairs(k)):
            gram = gram.at[:, i, j].set(products[:, column])
            gram = gram.at[:, j, i].set(products[:, column])
        empty = jnp.trace(gram, axis1=1, axis2=2) == 0
        gram = jnp.where(empty[:, None, None], jnp.eye(k, dtype=gram.dtype), gram)
        c = jnp.linalg.solve(gram, coeffs[:, :, None])[:, :, 0]
    c = jnp.concatenate([c, jnp.zeros((1, k), c.dtype)])  # rows without a label
    return v - jnp.sum(restrict(basis, labels) * c[_segments(labels)], axis=1)


def validate_basis(
    basis: jax.Array,
    name: str,
    comm: "Comm | None" = None,
    labels: Labels | None = None,
    label_sum: Callable[[jax.Array], jax.Array] | None = None,
) -> None:
    """Raise unless the (concrete) basis is finite with independent columns.
    Uses the ``k × k`` Gram matrix, reduced across ranks in MPI mode. With
    ``labels``, the columns on each label's rows: one Gram matrix per label
    (reduced by ``label_sum`` if given), labels whose columns vanish skipped."""
    if _skip_on_any_rank(
        _is_traced(basis) or (labels is not None and _is_traced(labels[1])), comm
    ):
        return
    if labels is not None:
        k = basis.shape[1]
        reduce = label_sum
        if comm is not None and reduce is None:
            from mpi4py import MPI

            def reduce(value):
                return jnp.asarray(comm.allreduce(np.asarray(value), op=MPI.SUM))

        with jax.ensure_compile_time_eval():
            basis = unit_label_basis(basis, labels, reduce)
            sums = label_sums(jnp.zeros(basis.shape[0], basis.dtype), basis, labels)
            if reduce is not None:
                sums = reduce(sums)
            products = np.asarray(sums[:, k:], dtype=np.float64)
        if not _all_ranks(bool(np.isfinite(products).all()), comm):
            raise ValueError(f"{name} contains non-finite values")
        gram = np.zeros((products.shape[0], k, k))
        for column, (i, j) in enumerate(_pairs(k)):
            gram[:, i, j] = gram[:, j, i] = products[:, column]
        ev = np.linalg.eigvalsh(gram)
        used = ev[:, -1] > 0
        bad = used & (ev[:, 0] <= 1e3 * float(jnp.finfo(basis.dtype).eps) * ev[:, -1])
        if not _all_ranks(not bool(bad.any()), comm):
            location = (
                f"label {int(np.flatnonzero(bad)[0])}" if bad.any() else "another rank"
            )
            raise ValueError(f"{name} has linearly dependent columns on {location}")
        return
    reduce_max = None
    if comm is not None:
        from mpi4py import MPI

        def reduce_max(value):
            maximum = np.array(value, copy=True)
            comm.Allreduce(MPI.IN_PLACE, maximum, op=MPI.MAX)
            return jnp.asarray(maximum)

    with jax.ensure_compile_time_eval():
        basis = unit_bases((basis,), reduce_max)[0]
        gram = np.asarray(basis.T @ basis, dtype=np.float64)
    if comm is not None:
        from mpi4py import MPI

        gram = comm.allreduce(gram, op=MPI.SUM)
    if not np.isfinite(gram).all():
        raise ValueError(f"{name} contains non-finite values")
    ev = np.linalg.eigvalsh(gram)
    if ev[-1] <= 0 or ev[0] <= 1e3 * float(jnp.finfo(basis.dtype).eps) * ev[-1]:
        raise ValueError(f"{name} has zero or linearly dependent columns")


def safe_exponent_limit(dtype: DTypeLike) -> int:
    """Columns whose largest entry ``m`` satisfies ``2^-limit <= m <= 2^limit``
    are left as they are: the Gram matrix's sums of absolute products (over up
    to 2^64 rows) stay below ``2^(2 limit + 64)``, and its diagonal entries
    stay normal."""
    return (int(jnp.finfo(dtype).maxexp) - 64) // 2 - 1


def rescaled_columns(largest: Any, limit: int, xp: Any = jnp) -> Any:
    """Which columns to rescale: a positive, finite largest entry outside the
    closed interval ``[2^-limit, 2^limit]`` (the same decision in JAX and
    NumPy, ``xp``)."""
    return (
        xp.isfinite(largest)
        & (largest > 0)
        & ((largest < 2.0**-limit) | (largest > 2.0**limit))
    )


def unit_bases(
    bases: tuple,
    reduce_max: Callable[[jax.Array], jax.Array] | None = None,
) -> tuple:
    """The declared bases (``None`` entries kept) with each column whose
    largest entry (over every rank) lies outside ``[2^-limit, 2^limit]``
    (`safe_exponent_limit`) rescaled by a power of two into [0.5, 1); other
    columns are untouched, so
    ordinary bases are bitwise unchanged. The scaling is applied directly to
    the entries (``ldexp``, never through a separately formed factor that
    may be out of range), and the column maxima of all bases go through one
    reduction."""
    present = [b for b in bases if b is not None]
    if not present:
        return tuple(bases)
    largest = jnp.concatenate(
        [jnp.max(jnp.abs(b), axis=0, initial=0.0) for b in present]
    )
    if reduce_max is not None:
        largest = reduce_max(largest)
    largest = jax.lax.stop_gradient(largest)
    _, exponent = jnp.frexp(largest)
    limit = safe_exponent_limit(largest.dtype)
    shift = jnp.where(rescaled_columns(largest, limit), exponent, 0).astype(jnp.int32)
    out: list[jax.Array | None] = []
    start = 0
    for basis in bases:
        if basis is None:
            out.append(None)
            continue
        columns = basis.shape[1]
        column_shift = shift[start : start + columns]
        # Unshifted columns are selected as they are (jnp.ldexp(x, 0) is not
        # always exactly x).
        out.append(jnp.where(column_shift != 0, jnp.ldexp(basis, -column_shift), basis))
        start += columns
    return tuple(out)


def project_out(
    v: jax.Array,
    basis: jax.Array,
    reduce_sum: Callable[[jax.Array], jax.Array] | None = None,
    labels: Labels | None = None,
) -> jax.Array:
    """``v - B(BᵀB)⁻¹Bᵀv``. ``reduce_sum`` (MPI) reduces the stacked
    ``[Bᵀv, vec(BᵀB)]`` in one collective. With ``labels``, per label (see
    the module): ``reduce_sum`` first combines scale statistics, then maps
    the scaled per-label partial sums to the labels' totals."""
    if labels is not None:
        basis = unit_label_basis(basis, labels, reduce_sum)
        sums = label_sums(v, basis, labels)
        if reduce_sum is not None:
            sums = reduce_sum(sums)
        return remove_label_sums(v, basis, labels, sums)
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


_reduce_sum_keyval: int | None = None


def make_mpi_reduce_sum(comm: "Comm") -> Callable[[jax.Array], jax.Array]:
    """Sum all-reduce whose transpose is also a sum all-reduce, as a linear
    map (every derivative mode and order; see `jaxamg.linear_map`).

    The reduced value is used in every rank's rows, so its cotangent is the sum
    of the per-rank contributions; mpi4jax's own transpose rule (identity)
    would only project out each rank's local component in the backward pass.
    One map per communicator, kept as an attribute of it.
    """
    global _reduce_sum_keyval
    import mpi4jax
    from mpi4py import MPI

    from .linear_map import LinearMap

    if _reduce_sum_keyval is None:
        _reduce_sum_keyval = MPI.Comm.Create_keyval()
    cached = comm.Get_attr(_reduce_sum_keyval)
    if cached is not None:
        return cast(Callable[[jax.Array], jax.Array], cached)

    def allreduce(x: jax.Array) -> jax.Array:
        return mpi4jax.allreduce(x, op=MPI.SUM, comm=comm)

    reduce_sum = LinearMap(allreduce, allreduce, "global_sum")
    comm.Set_attr(_reduce_sum_keyval, reduce_sum)
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
    """Warn when ``A·1 = 0`` to round-off. Skipped when any rank's values are
    traced; in MPI mode every rank must call it. Memoized per matrix buffer."""
    if _skip_on_any_rank(_is_traced(A.data, A.indptr), comm):
        return
    singular = _singular_memo.get(A.data)
    if not _all_ranks(singular is not None, comm):
        # Under a jit trace every op is staged; evaluate the check eagerly.
        num = den = 0.0
        if A.data.size:
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
    labels: Labels | None = None,
    column_labels: Callable[[jax.Array], jax.Array] | None = None,
) -> None:
    """Warn unless ``max|matvec(B)| ≤ sqrt(eps)·‖A‖∞·max|B|``. Skipped when
    any rank's values are traced; ``matvec`` must be collective in MPI mode.
    Memoized per (matrix buffer, basis).

    With ``labels``: the columns on the labelled rows, and no stored nonzero
    entry between two rows of different labels (together, ``A·v = 0`` for
    every per-label vector). ``column_labels(rows)`` gives each stored entry's
    column label, -2 where it cannot be compared (default:
    ``rows[A.indices]``)."""
    arrays = (A.data, A.indptr, basis) + (() if labels is None else (labels[1],))
    if _skip_on_any_rank(_is_traced(*arrays), comm):
        return
    if labels is not None:
        basis = restrict(basis, labels)
    key = (A.data, basis) if labels is None else (A.data, basis, labels[1])
    verdict = _verified_memo.get(*key)
    if not _all_ranks(verdict is not None, comm):
        with jax.ensure_compile_time_eval():
            _, absrow = _row_sums(A)
            # Empty local blocks contribute zeros; ``matvec`` communicates, so
            # every rank applies it.
            a_norm = float(jnp.max(absrow, initial=0.0))
            b_norm = float(jnp.max(jnp.abs(basis), initial=0.0))
            residual = float(jnp.max(jnp.abs(matvec(basis)), initial=0.0))
            coupled = 0
            if labels is not None:
                rows = labels[1]
                columns = (
                    rows[A.indices] if column_labels is None else column_labels(rows)
                )
                coupled = int(
                    jnp.sum(
                        (A.data != 0)
                        & (columns != -2)
                        & (rows[row_index(A)] != columns)
                    )
                )
        scale = _global_max(a_norm, comm) * _global_max(b_norm, comm)
        residual = _global_max(residual, comm)
        ratio = residual / scale if scale > 0 else 0.0
        verdict = (ratio, _global_max(coupled, comm))
        _verified_memo.put(verdict, *key)
    ratio, coupled = verdict
    tol = _verify_rtol(A.data.dtype)
    if ratio > tol:
        target = "Aᵀ" if name == "transpose_nullspace" else "A"
        warnings.warn(
            f"{name} does not appear to span the null space of {target}: "
            f"relative residual {ratio:.2e} (expected ≤ {tol:.1e}).",
            NullSpaceWarning,
            stacklevel=stacklevel,
        )
    if coupled:
        warnings.warn(
            "labels split a connected part of the matrix: stored nonzero "
            f"entries couple two labels, so {name}'s per-label vectors are not "
            "null vectors.",
            NullSpaceWarning,
            stacklevel=stacklevel,
        )
