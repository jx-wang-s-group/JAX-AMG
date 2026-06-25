from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from .config import _AMG_DEFAULTS, _prepare_solver_config_dict
from .jaxamg import solve
from .utils import MatrixOrOperator, deep_merge

if TYPE_CHECKING:
    import lineax
    from mpi4py.MPI import Comm

# Sentinel: by default the Lineax preconditioner inherits the operator's tags.
_INHERIT_TAGS = object()

_DEFAULT_PRECONDITIONER_SOLVER_CONFIG = {
    **deep_merge({}, _AMG_DEFAULTS),
    "convergence": "RELATIVE_INI",
    "tolerance": 1e-6,
    "norm": "L2",
    "exact_coarse_solve": 1,
}


def _prepare_preconditioner_config(
    config: dict[str, Any] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Build a nested AmgX config for approximate-inverse applications."""
    return _prepare_solver_config_dict(
        config,
        _DEFAULT_PRECONDITIONER_SOLVER_CONFIG,
        kwargs=kwargs,
    )


def make_preconditioner(
    A: MatrixOrOperator,
    config: dict[str, Any] | None = None,
    *,
    comm: "Comm | None" = None,
    nglobal: int | None = None,
    partition_info: tuple[int, int] | None = None,
    save_stats_file: str | None = None,
    return_info: bool = False,
    **kwargs: Any,
) -> Callable:
    """Create a callable approximate inverse for external Krylov solvers.

    The returned callable can be passed directly as the `M` argument to
    `jax.scipy.sparse.linalg.cg(...)` or `jax.scipy.sparse.linalg.bicgstab(...)`.

    By default the approximate inverse is a *single* AMG V-cycle (`solver="AMG"`,
    `max_iters=1`), so each application is one cheap AMG sweep. This is deliberately
    different from `jaxamg.solve`, whose default is a full Krylov solve (`PBICGSTAB`)
    preconditioned by AMG: here AMG *is* the preconditioner and the outer Krylov
    method owns the iteration. Pass `config`/`kwargs` for a stronger inner
    application (e.g. more sweeps, a W-cycle, or `max_iters=2`).

    Args:
        A: Matrix or callable operator to precondition.
        config: Optional AmgX configuration. If omitted, a single-cycle AMG
            approximate-inverse config is used.
        comm: Optional MPI communicator for distributed solves. If `A` already
            has MPI metadata attached via `jaxamg.with_cache(..., mpi=...)`, this
            may be omitted.
        nglobal: Global matrix row count for MPI mode.
        partition_info: Local row partition `(row_start, row_end)` for MPI mode.
        save_stats_file: Optional stats output path passed to `jaxamg.solve(...)`.
        return_info: If `True`, the returned callable yields `(x, info)` instead of
            only `x`.
        **kwargs: Additional solver config overrides.

    Returns:
        A callable representing an approximate inverse `M^{-1}`.
    """

    preconditioner_config = _prepare_preconditioner_config(config, **kwargs)

    def apply(rhs: jax.Array) -> jax.Array | tuple[jax.Array, dict]:
        x, info = solve(
            A,
            rhs,
            config=preconditioner_config,
            comm=comm,
            nglobal=nglobal,
            partition_info=partition_info,
            save_stats_file=save_stats_file,
        )
        if return_info:
            return x, info
        return x

    return apply


def make_lineax_preconditioner(
    operator: "lineax.AbstractLinearOperator",
    config: dict[str, Any] | None = None,
    *,
    tags: Any = _INHERIT_TAGS,
    comm: "Comm | None" = None,
    nglobal: int | None = None,
    partition_info: tuple[int, int] | None = None,
    save_stats_file: str | None = None,
    **kwargs: Any,
) -> "lineax.FunctionLinearOperator":
    """Wrap a Lineax operator as an AMG preconditioner operator.

    This is the operator->operator counterpart of `make_preconditioner`: it maps a
    system operator ``A`` (a `lineax.AbstractLinearOperator`) to a preconditioner
    operator ``M`` with ``M.mv(r) ≈ A⁻¹ r``, ready to hand to a Lineax solver via
    ``options={"preconditioner": M}``. It folds the usual ``make_preconditioner``
    plus `FunctionLinearOperator` wiring into a single call.

    The operator's matrix-free action (`operator.mv`) is handed to JAX-AMG, whose
    sparsity detection assembles the explicit matrix AmgX needs (traced in one pass
    when possible, probed otherwise). The pattern is detected and cached eagerly
    here, since Lineax solvers apply the preconditioner under `jax.jit` where
    on-the-fly detection is impossible. A `MatrixLinearOperator` is assembled
    directly from its concrete matrix instead.

    Args:
        operator: The system operator to precondition.
        config: Optional AmgX configuration (see `make_preconditioner`).
        tags: Lineax tags for the returned preconditioner. By default the operator's
            own tags are inherited (``A⁻¹`` shares ``A``'s symmetry/definiteness),
            which CG needs in order to accept the preconditioner. Pass an explicit
            value (e.g. ``()``) to override.
        comm: Optional MPI communicator for distributed solves.
        nglobal: Global matrix row count for MPI mode.
        partition_info: Local row partition ``(row_start, row_end)`` for MPI mode.
        save_stats_file: Optional stats output path passed to `jaxamg.solve(...)`.
        **kwargs: Additional solver config overrides forwarded to `make_preconditioner`.

    Returns:
        A `lineax.FunctionLinearOperator` approximating ``A⁻¹``.
    """
    try:
        import lineax as lx
    except ImportError as e:  # pragma: no cover - exercised only without lineax
        raise ImportError(
            "make_lineax_preconditioner requires the optional `lineax` package "
            "(`pip install lineax`)."
        ) from e

    if not isinstance(operator, lx.AbstractLinearOperator):
        raise TypeError(
            "make_lineax_preconditioner expects a lineax.AbstractLinearOperator; got "
            f"{type(operator).__name__}. Wrap a matrix with lineax.MatrixLinearOperator(A), "
            "or use jaxamg.make_preconditioner(A) for the non-Lineax path."
        )

    in_structure = operator.in_structure()
    leaves = jax.tree_util.tree_leaves(in_structure)
    if len(leaves) != 1 or getattr(leaves[0], "ndim", None) != 1:
        raise ValueError(
            "make_lineax_preconditioner supports operators acting on a single 1D "
            f"vector; got input structure {in_structure}."
        )

    if tags is _INHERIT_TAGS:
        tags = getattr(operator, "tags", ())

    # A MatrixLinearOperator already holds the assembled matrix, so hand it over
    # directly; otherwise wrap the matrix-free action in a plain function (the bound
    # `operator.mv` cannot hold the coloring cache) and pre-scan it now, so the
    # sparsity pattern is detected and cached before Lineax applies it under JIT.
    if isinstance(operator, lx.MatrixLinearOperator):
        amg_input: MatrixOrOperator = operator.as_matrix()
    else:
        mv = operator.mv
        in_dtype = leaves[0].dtype

        # Cast probes to the operator's declared input dtype: sparsity detection
        # and JAX-AMG's materialization feed one-hot vectors through this action,
        # and a closure-converted Lineax operator rejects a mismatched dtype.
        def _action(x: jax.Array) -> jax.Array:
            return mv(jnp.asarray(x, dtype=in_dtype))

        if comm is None:
            from .sparsity import cache_coloring

            n = int(leaves[0].shape[0])
            cache_coloring(_action, (n, n))
        amg_input = _action

    apply = make_preconditioner(
        amg_input,
        config,
        comm=comm,
        nglobal=nglobal,
        partition_info=partition_info,
        save_stats_file=save_stats_file,
        **kwargs,
    )
    return lx.FunctionLinearOperator(apply, in_structure, tags=tags)
