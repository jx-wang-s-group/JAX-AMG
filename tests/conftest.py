"""Pytest configuration and fixtures for jaxamg tests."""

import jax
import pytest


@pytest.fixture(scope="session", autouse=True)
def configure_jax():
    """Configure JAX for testing."""
    # Ensure JAX uses 32-bit floats by default
    jax.config.update("jax_enable_x64", False)


def pytest_collection_modifyitems(config, items):
    """Centralized skip logic for the `gpu` and `mpi` markers.

    - `gpu`: tests that run the native AmgX solver; skipped unless JAX has a
      GPU backend.
    - `mpi`: the skip behavior normally comes from the optional pytest-mpi
      plugin; without it the MPI tests would execute in a plain pytest run
      (no mpirun, no ranks) and fail or hang instead of skipping.
    """
    if jax.default_backend() != "gpu":
        skip_gpu = pytest.mark.skip(reason="requires a GPU JAX backend")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    if not config.pluginmanager.hasplugin("pytest_mpi"):
        skip_mpi = pytest.mark.skip(
            reason="pytest-mpi is not active; MPI tests require mpirun + --only-mpi"
        )
        for item in items:
            if "mpi" in item.keywords:
                item.add_marker(skip_mpi)
