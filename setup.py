from setuptools import setup, Extension
import os
import pybind11
from dotenv import load_dotenv
from jax import ffi

# Load environment variables from .env file
load_dotenv()

# Require AMGX_ROOT, AMGX_BUILD, and CUDA_HOME to be set in the environment or .env file
AMGX_ROOT = os.environ.get("AMGX_ROOT")
if not AMGX_ROOT:
    raise RuntimeError("Environment variable AMGX_ROOT must be set")

AMGX_BUILD = os.environ.get("AMGX_BUILD")
if not AMGX_BUILD:
    raise RuntimeError("Environment variable AMGX_BUILD must be set")

CUDA_HOME = os.environ.get("CUDA_HOME")
if not CUDA_HOME:
    raise RuntimeError("Environment variable CUDA_HOME must be set")

# XLA FFI include path from jaxlib
XLA_FFI_INCLUDE = ffi.include_dir()

include_dirs = [
    pybind11.get_include(),
    XLA_FFI_INCLUDE,
    os.path.join(AMGX_ROOT, "include"),
    os.path.join(CUDA_HOME, "include"),
]

library_dirs = [
    AMGX_BUILD,
    os.path.join(CUDA_HOME, "lib64"),
]

runtime_library_dirs = [
    AMGX_BUILD,
    os.path.join(CUDA_HOME, "lib64"),
]

ext = Extension(
    "jaxamg._amgx_ext",
    sources=["jaxamg/amgx_custom_call.cc"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
    libraries=["amgxsh", "cudart"],
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="jaxamg",
    version="0.0.1",
    packages=["jaxamg"],
    ext_modules=[ext],
    setup_requires=["pybind11>=2.6.0", "python-dotenv>=1.0.0"],
)
