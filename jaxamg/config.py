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


def prepare_config(
    user_config: dict | None = None, save_stats: bool = False, **kwargs: Any
) -> str:
    """
    Prepare the final configuration string for AmgX.

    Merges the user config into defaults, applies overrides (kwargs),
    injects residual tracking settings, and wraps the result in AMGX's
    ``config_version: 2`` nested JSON format.
    """
    # Clean copy of AMG defaults
    amg_defaults = deep_merge({}, _AMG_DEFAULTS)

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

    return _format_config(merged_config)
