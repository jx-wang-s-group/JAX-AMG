import json
import tempfile
import os
from typing import Dict, Any, Union, Optional


def _has_nested_dict(d: Dict) -> bool:
    """Check if dictionary contains nested dictionaries."""
    if not isinstance(d, dict):
        return False
    for v in d.values():
        if isinstance(v, dict):
            return True
    return False


def _format_config(config: Union[Dict, str, None]) -> str:
    """
    Format configuration for AmgX.

    - Strings are passed through as-is
    - Flat dicts are converted to "key=val, key2=val2" format
    - Nested dicts are written to a temporary JSON file and the path is returned
    """
    if config is None:
        return ""
    if isinstance(config, str):
        return config
    if isinstance(config, dict):
        # Check if dict has nested structures
        if _has_nested_dict(config):
            # Write to temporary JSON file (following pyamgx approach)
            # Note: We can't use NamedTemporaryFile with delete=True because
            # the file needs to persist until AmgX reads it in the C++ layer
            fd, path = tempfile.mkstemp(suffix=".json", text=True)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(config, f, indent=2)
                return path
            except:
                os.close(fd)
                os.unlink(path)
                raise
        else:
            # Flat dict: convert to "key=val, key2=val2"
            return ", ".join(f"{k}={v}" for k, v in config.items())
    raise TypeError("Config must be a string or dictionary.")


def prepare_config(user_config: Optional[Union[Dict, str]] = None, **kwargs) -> str:
    """
    Prepare the final configuration string for AmgX.

    Merges default settings, user-provided config, and any keyword arguments.
    Handles serialization of nested dictionaries if present.

    Args:
        user_config: Base configuration (dict or string).
        **kwargs: Overrides for configuration parameters.

    Returns:
        A string that AmgX can interpret (key=val string or path to JSON file).
    """
    # If config is a simple string and no kwargs are provided, return it directly
    if user_config and isinstance(user_config, str) and not kwargs:
        return user_config

    # Default configuration
    merged_config = {
        "config_version": 2,
        "solver": "CG",
        "preconditioner": "AMG",
        "max_iters": 100,
        "tolerance": 1e-6,
        "norm": "L2",
        "print_solve_stats": 1,
        "monitor_residual": 1,
        "cycle": "V",
        "smoother": "JACOBI_L1",
    }

    # Update with user configuration
    if user_config:
        if isinstance(user_config, dict):
            merged_config.update(user_config)
        elif isinstance(user_config, str):
            # If user provides a string config AND kwargs, we can't easily merge.
            # Strategy: strings bypass defaults if no kwargs are present (handled above).
            # If kwargs ARE present, we ignore the string (as it's unstructured)
            # and use defaults + kwargs. This preserves behavior from the original implementation.
            pass

    # Apply overrides (kwargs)
    merged_config.update(kwargs)

    return _format_config(merged_config)
