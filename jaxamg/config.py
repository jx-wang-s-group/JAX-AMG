import json
from typing import Any


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into a copy of *base*."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


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

    merged = _deep_merge(merged, solver_config)

    # If AMG is selected, ensure missing AMG sub-settings are filled by defaults.
    precond = merged.get("preconditioner", {})
    if isinstance(precond, dict) and precond.get("solver") == "AMG":
        merged["preconditioner"] = _deep_merge(amg_defaults, precond)

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


def prepare_config(user_config: dict | None = None, **kwargs: Any) -> str:
    """
    Prepare the final configuration string for AmgX.

    Merges the user config into defaults, applies overrides (kwargs),
    injects residual tracking settings, and wraps the result in AMGX's
    ``config_version: 2`` nested JSON format.
    """
    if user_config is not None and not isinstance(user_config, dict):
        raise TypeError(
            f"Config must be a dictionary, got {type(user_config).__name__}."
        )

    amg_defaults = {
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

    defaults = {
        "solver": "PBICGSTAB",
        "preconditioner": dict(amg_defaults),
        "convergence": "RELATIVE_INI",
        "tolerance": 1e-6,
        "max_iters": 1000,
        "norm": "L2",
    }

    is_nested = False
    if user_config:
        if "config_version" in user_config:
            is_nested = True
        elif isinstance(user_config.get("solver"), dict):
            is_nested = True

    if user_config and is_nested:
        merged_config = user_config.copy()
        solver_block = merged_config.get("solver", {})
        if not isinstance(solver_block, dict):
            raise TypeError("Nested config must contain a dictionary at key 'solver'.")
        merged_solver = _merge_solver_with_defaults(
            solver_block, defaults, amg_defaults
        )
        merged_solver = _deep_merge(merged_solver, kwargs)
        merged_config["solver"] = merged_solver
    else:
        merged_config = _merge_solver_with_defaults(
            user_config or {}, defaults, amg_defaults
        )
        merged_config = _deep_merge(merged_config, kwargs)

    # Inject residual tracking into the correct solver scope.
    target_dict = merged_config
    if "solver" in merged_config and isinstance(merged_config["solver"], dict):
        target_dict = merged_config["solver"]

    target_dict["store_res_history"] = 1
    target_dict["monitor_residual"] = 1

    # Wrap flat configs into AMGX's config_version 2 nested format.
    if "config_version" not in merged_config:
        merged_config = {
            "config_version": 2,
            "solver": merged_config,
        }

    return _format_config(merged_config)
