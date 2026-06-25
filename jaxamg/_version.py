"""Single source of truth for the package version.

Read at build time by ``pyproject.toml`` (``[tool.setuptools.dynamic]``) and at
runtime as ``jaxamg.__version__``. Kept in a dependency-free module so the build
backend can extract it statically without importing the native extension.
"""

__version__ = "0.1.1"
