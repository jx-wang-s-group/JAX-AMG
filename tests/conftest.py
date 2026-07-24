"""Pytest configuration and fixtures for jaxamg tests."""

import os

# Under mpirun, pin each rank to a single distinct GPU before JAX initializes
# its CUDA backend (test collection already allocates on device). Otherwise
# every rank allocates on the first visible device, whose memory is exhausted
# once a few ranks preallocate on it.
_local_rank = os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK")
_visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
if _local_rank is not None and _visible_gpus:
    _gpus = _visible_gpus.split(",")
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus[int(_local_rank) % len(_gpus)]

import jax  # noqa: E402
import pytest  # noqa: E402


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
