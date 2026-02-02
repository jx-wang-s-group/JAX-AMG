import json
import tempfile
import os
from typing import Dict, Any, Union, Optional


def _format_config(config: Dict) -> str:
    """
    Format configuration for AmgX.

    Serializes the config dictionary to a temporary JSON file and returns its path.
    """
    if config is None:
        return ""

    if not isinstance(config, dict):
        raise TypeError(
            "Config must be a dictionary. String configuration is no longer supported."
        )

    # Serialize to temp JSON file
    fd, path = tempfile.mkstemp(suffix=".json", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(config, f, indent=2)
        return path
    except:
        os.close(fd)
        os.unlink(path)
        raise


def prepare_config(user_config: Optional[Dict] = None, **kwargs) -> str:
    """
    Prepare the final configuration string for AmgX.

    Merges the user config into defaults (PBICGSTAB+AMG), applies overrides (kwargs),
    and injects residual tracking settings.
    """
    if user_config is not None and not isinstance(user_config, dict):
        raise TypeError(
            f"Config must be a dictionary, got {type(user_config).__name__}. String configuration is no longer supported."
        )

    # Default config: PBICGSTAB + AMG with block Jacobi smoother
    defaults = {
        "solver": "PBICGSTAB",
        "preconditioner": {
            "solver": "AMG",
            "smoother": {"solver": "JACOBI_L1", "relaxation_factor": 0.8},
            "presweeps": 1,
            "postsweeps": 1,
            "coarse_solver": "NOSOLVER",
            "max_levels": 50,
            "cycle": "V",
        },
        "tolerance": 1e-6,
        "max_iters": 1000,
        "print_solve_stats": 1,
        "norm": "L2",
    }

    # Determine base: use defaults for flat configs, but respect nested/versioned configs
    # by starting empty to avoid polluting user intent.
    is_nested = False
    if user_config:
        if "config_version" in user_config:
            is_nested = True
        elif isinstance(user_config.get("solver"), dict):
            is_nested = True

    if is_nested:
        merged_config = user_config.copy()
    else:
        merged_config = defaults.copy()
        if user_config:
            merged_config.update(user_config)

    # Apply kwargs (highest priority overrides)
    merged_config.update(kwargs)

    # Inject residual tracking into the correct solver scope
    target_dict = merged_config
    if "solver" in merged_config and isinstance(merged_config["solver"], dict):
        # We are likely in a nested config where 'solver' contains the solver params
        target_dict = merged_config["solver"]

    target_dict["store_res_history"] = 1
    target_dict["monitor_residual"] = 1

    return _format_config(merged_config)
