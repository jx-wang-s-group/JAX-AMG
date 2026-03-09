from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import jax

from .config import _AMG_DEFAULTS, _prepare_solver_config_dict
from .jaxamg import solve
from .utils import MatrixOrOperator, deep_merge

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

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

    Args:
        A: Matrix or callable operator to precondition.
        config: Optional AmgX configuration. If omitted, an AMG-only
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
