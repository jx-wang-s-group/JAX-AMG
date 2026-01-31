"""Pytest configuration and fixtures for jaxamg tests."""
import pytest
import jax

@pytest.fixture(scope="session", autouse=True)
def configure_jax():
    """Configure JAX for testing."""
    # Ensure JAX uses 32-bit floats by default
    jax.config.update("jax_enable_x64", False)
