import json
from collections.abc import Callable
from typing import Any, cast

from .utils import deep_merge

_AMG_DEFAULTS = {
    "solver": "AMG",
    "algorithm": "CLASSICAL",
    "selector": "PMIS",
    "interpolator": "D2",
    "smoother": {
        "solver": "BLOCK_JACOBI",
        "relaxation_factor": 0.9,
    },
    "presweeps": 1,
    "postsweeps": 1,
    "max_levels": 100,
    "strength_threshold": 0.5,
    "dense_lu_num_rows": 1,
    "aggressive_levels": 0,
    "coarse_solver": "DENSE_LU_SOLVER",
    "max_iters": 1,
    "cycle": "V",
}

# Block matrices (block_dim > 1): AmgX's classical AMG only supports scalar
# matrices, so the AMG defaults switch to aggregation, which supports square
# blocks (pairwise SIZE_2 aggregates + block-Jacobi smoothing). The coarse
# floor is deliberately generous: aggregating all the way down to a couple of
# rows produces degenerate distributed coarse levels (empty per-rank parts)
# whose dense-LU solve yields NaNs, analogous to the MULTICOLOR_DILU case
# guarded in validate_config.
_AMG_AGGREGATION_DEFAULTS = {
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_2",
    "smoother": {
        "solver": "BLOCK_JACOBI",
        "relaxation_factor": 0.9,
    },
    "presweeps": 1,
    "postsweeps": 1,
    "max_levels": 100,
    "min_coarse_rows": 32,
    "dense_lu_num_rows": 64,
    "coarse_solver": "DENSE_LU_SOLVER",
    "max_iters": 1,
    "cycle": "V",
}


def _is_nested_config(config: dict | None) -> bool:
    """Return whether *config* already uses AMGX's nested config structure."""
    if not config:
        return False
    return "config_version" in config or isinstance(config.get("solver"), dict)


def _prepare_solver_config_dict(
    user_config: dict | None,
    solver_defaults: dict,
    *,
    kwargs: dict[str, Any] | None = None,
    solver_merger: Callable[[dict, dict], dict] | None = None,
) -> dict:
    """Normalize flat or nested solver config input into a nested config dict."""
    if user_config is not None and not isinstance(user_config, dict):
        raise TypeError(
            f"Config must be a dictionary, got {type(user_config).__name__}."
        )

    kwargs = kwargs or {}

    if solver_merger is None:
        solver_merger = lambda solver_config, defaults: deep_merge(  # noqa: E731
            defaults, solver_config
        )

    if _is_nested_config(user_config):
        merged_config = cast(dict[str, Any], user_config).copy()
        solver_block = merged_config.get("solver", {})
        if not isinstance(solver_block, dict):
            raise TypeError("Nested config must contain a dictionary at key 'solver'.")
        merged_solver = solver_merger(solver_block, solver_defaults)
        merged_solver = deep_merge(merged_solver, kwargs)
        merged_config["config_version"] = merged_config.get("config_version", 2)
        merged_config["solver"] = merged_solver
        return merged_config

    merged_solver = solver_merger(user_config or {}, solver_defaults)
    merged_solver = deep_merge(merged_solver, kwargs)
    return {"config_version": 2, "solver": merged_solver}


def _merge_solver_with_defaults(
    solver_config: dict, solver_defaults: dict, amg_defaults: dict
) -> dict:
    """Merge solver config into defaults with non-AMG preconditioner cleanup."""
    merged = solver_defaults.copy()

    # If user selected a non-AMG preconditioner, drop AMG-only defaults first.
    user_precond = solver_config.get("preconditioner", {})
    if isinstance(user_precond, dict):
        user_precond_solver = user_precond.get("solver", "AMG")
    else:
        user_precond_solver = user_precond
    if user_precond_solver != "AMG":
        merged["preconditioner"] = {}

    merged = deep_merge(merged, solver_config)

    # If AMG is selected, ensure missing AMG sub-settings are filled by defaults.
    precond = merged.get("preconditioner", {})
    if isinstance(precond, dict) and precond.get("solver") == "AMG":
        merged["preconditioner"] = deep_merge(amg_defaults, precond)

    return merged


def _format_config(config: dict | None) -> str:
    """
    Format configuration for AmgX.

    Serializes the config dictionary to a canonical JSON string.
    The stable serialization is used as the solver cache key in C++.
    """
    if config is None:
        return ""

    if not isinstance(config, dict):
        raise TypeError(
            "Config must be a dictionary. String configuration is no longer supported."
        )

    # Canonical JSON to guarantee deterministic cache keys for identical configs.
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def _as_upper(value: Any) -> str:
    return str(value).upper() if value is not None else ""


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _amg_blocks(solver_block: Any) -> list[dict]:
    if not isinstance(solver_block, dict):
        return []

    blocks = []
    if _as_upper(solver_block.get("solver")) == "AMG":
        blocks.append(solver_block)

    preconditioner = solver_block.get("preconditioner")
    if (
        isinstance(preconditioner, dict)
        and _as_upper(preconditioner.get("solver")) == "AMG"
    ):
        blocks.append(preconditioner)

    return blocks


def _require_positive(block: dict, key: str, path: str) -> None:
    if key not in block:
        return
    value = _as_float(block[key])
    if value is None or value <= 0:
        raise ValueError(f"Invalid AmgX config: {path}.{key} must be positive.")


def _require_nonnegative(block: dict, key: str, path: str) -> None:
    if key not in block:
        return
    value = _as_float(block[key])
    if value is None or value < 0:
        raise ValueError(f"Invalid AmgX config: {path}.{key} must be nonnegative.")


def _validate_smoother(amg: dict, path: str) -> str:
    smoother = amg.get("smoother")
    if smoother is None:
        return ""
    if isinstance(smoother, str):
        return _as_upper(smoother)
    if isinstance(smoother, dict):
        return _as_upper(smoother.get("solver"))
    raise TypeError(
        f"Invalid AmgX config: {path}.smoother must be a string or dictionary."
    )


def validate_config(config: dict, *, mpi: bool = False, block_dim: int = 1) -> None:
    """Validate a prepared AmgX config for known unsupported combinations.

    The validator is intentionally conservative: it catches configurations that
    are known to trigger opaque AmgX/CUDA failures before entering the native
    solver, but otherwise leaves AmgX's own config handling in charge.
    """
    solver_block = config.get("solver", config)
    if not isinstance(solver_block, dict):
        return

    if block_dim > 1:
        # AmgX's classical AMG hard-fails on block matrices ("Classical AMG
        # not implemented for block_size != 1"); only aggregation AMG supports
        # square blocks.
        for amg in _amg_blocks(solver_block):
            path = "solver" if amg is solver_block else "solver.preconditioner"
            algorithm = _as_upper(amg.get("algorithm", "CLASSICAL"))
            if algorithm == "CLASSICAL":
                raise ValueError(
                    f"Invalid AmgX config: {path} uses CLASSICAL AMG, which "
                    "does not support block matrices (block_dim > 1). Use "
                    "'algorithm': \"AGGREGATION\" (e.g. with 'selector': "
                    '"SIZE_2") or a non-AMG preconditioner.'
                )

    communicator = solver_block.get("communicator")
    if communicator is not None and _as_upper(communicator) not in {
        "MPI",
        "MPI_DIRECT",
    }:
        raise ValueError(
            "Invalid AmgX config: solver.communicator must be 'MPI' or 'MPI_DIRECT'."
        )

    _require_positive(solver_block, "max_iters", "solver")
    _require_positive(solver_block, "tolerance", "solver")

    solver_name = _as_upper(solver_block.get("solver"))
    if solver_name in {"GMRES", "FGMRES"} and "gmres_n_restart" in solver_block:
        _require_positive(solver_block, "gmres_n_restart", "solver")

    for amg in _amg_blocks(solver_block):
        path = "solver" if amg is solver_block else "solver.preconditioner"
        _require_positive(amg, "max_iters", path)
        _require_positive(amg, "max_levels", path)
        _require_positive(amg, "dense_lu_num_rows", path)
        _require_positive(amg, "min_coarse_rows", path)
        _require_nonnegative(amg, "presweeps", path)
        _require_nonnegative(amg, "postsweeps", path)

        smoother_solver = _validate_smoother(amg, path)
        if not mpi or smoother_solver != "MULTICOLOR_DILU":
            continue

        dense_lu_num_rows = _as_int(amg.get("dense_lu_num_rows"), default=1)
        min_coarse_rows = _as_int(amg.get("min_coarse_rows"))
        dense_lu_too_small = dense_lu_num_rows is not None and dense_lu_num_rows <= 1
        min_coarse_too_small = min_coarse_rows is not None and min_coarse_rows <= 1
        if dense_lu_too_small or min_coarse_too_small:
            raise ValueError(
                "Invalid MPI AmgX config: AMG with MULTICOLOR_DILU and "
                "a tiny coarse-grid floor can coarsen to degenerate distributed "
                "levels and fail inside AmgX's DILU halo setup. Set "
                "dense_lu_num_rows and min_coarse_rows to larger values (for "
                "example 64), or use a smoother such as BLOCK_JACOBI."
            )


def prepare_config(
    user_config: dict | None = None,
    save_stats: bool = False,
    mpi: bool = False,
    block_dim: int = 1,
    **kwargs: Any,
) -> str:
    """
    Prepare the final configuration string for AmgX.

    Merges the user config into defaults, applies overrides (kwargs),
    injects residual tracking settings, and wraps the result in AMGX's
    ``config_version: 2`` nested JSON format. For block matrices
    (``block_dim > 1``) the AMG defaults switch from classical to
    aggregation AMG, the only AmgX AMG algorithm that supports blocks.
    """
    # Clean copy of AMG defaults
    amg_defaults = deep_merge(
        {}, _AMG_AGGREGATION_DEFAULTS if block_dim > 1 else _AMG_DEFAULTS
    )

    defaults = {
        "solver": "PBICGSTAB",
        "preconditioner": dict(amg_defaults),
        "convergence": "RELATIVE_INI",
        "tolerance": 1e-6,
        "max_iters": 1000,
        "norm": "L2",
        "exact_coarse_solve": 1,
    }

    merged_config = _prepare_solver_config_dict(
        user_config,
        defaults,
        kwargs=kwargs,
        solver_merger=lambda solver_config, solver_defaults: _merge_solver_with_defaults(
            solver_config, solver_defaults, amg_defaults
        ),
    )

    # Inject residual tracking into the correct solver scope.
    target_dict = merged_config
    if "solver" in merged_config and isinstance(merged_config["solver"], dict):
        target_dict = merged_config["solver"]

    target_dict["store_res_history"] = 1
    target_dict["monitor_residual"] = 1
    if save_stats:
        target_dict["print_solve_stats"] = 1

        # Inject AMG grid stats into any AMG block
        def _inject_amg_stats(d: dict) -> None:
            if isinstance(d, dict) and d.get("solver") == "AMG":
                d["print_grid_stats"] = 1

        _inject_amg_stats(target_dict)
        _inject_amg_stats(target_dict.get("preconditioner", {}))

    validate_config(merged_config, mpi=mpi, block_dim=block_dim)

    return _format_config(merged_config)


def outer_max_iters(config_str: str) -> int:
    """Return the outer solver scope's ``max_iters`` from a prepared config string.

    Used to size the residual-history slots of the solve's stats output. The
    scope mirrors where ``prepare_config`` injects ``store_res_history`` (which
    is also the scope whose history AmgX exposes); AmgX's registered default of
    100 is the fallback when the value is absent.
    """
    try:
        cfg = json.loads(config_str)
    except (TypeError, ValueError):
        return 100
    scope = cfg
    if isinstance(cfg, dict) and isinstance(cfg.get("solver"), dict):
        scope = cfg["solver"]
    if not isinstance(scope, dict):
        return 100
    value = _as_int(scope.get("max_iters"), 100)
    return value if value is not None and value > 0 else 100
