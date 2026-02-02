from setuptools import setup, Extension
import os
import subprocess
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

# Get MPI compile and link flags
try:
    mpi_compile_output = subprocess.check_output(
        ["mpicxx", "--showme:compile"], text=True
    ).strip()
    mpi_compile_flags = mpi_compile_output.split()

    mpi_link_output = subprocess.check_output(
        ["mpicxx", "--showme:link"], text=True
    ).strip()
    mpi_link_flags = mpi_link_output.split()

    # Extract MPI include directories
    mpi_include_dirs = [flag[2:] for flag in mpi_compile_flags if flag.startswith("-I")]

    # Extract MPI library directories
    mpi_library_dirs = [flag[2:] for flag in mpi_link_flags if flag.startswith("-L")]

    # Extract MPI libraries
    mpi_libraries = [flag[2:] for flag in mpi_link_flags if flag.startswith("-l")]

except (subprocess.CalledProcessError, FileNotFoundError):
    print("Warning: mpicxx not found. Building without MPI support.")
    mpi_include_dirs = []
    mpi_library_dirs = []
    mpi_libraries = []

include_dirs = [
    pybind11.get_include(),
    XLA_FFI_INCLUDE,
    os.path.join(AMGX_ROOT, "include"),
    os.path.join(CUDA_HOME, "include"),
] + mpi_include_dirs

library_dirs = [
    AMGX_BUILD,
    os.path.join(CUDA_HOME, "lib64"),
] + mpi_library_dirs

runtime_library_dirs = [
    AMGX_BUILD,
    os.path.join(CUDA_HOME, "lib64"),
] + mpi_library_dirs

ext = Extension(
    "jaxamg._amgx_ext",
    sources=["jaxamg/amgx_custom_call.cc"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
    libraries=["amgxsh", "cudart"] + mpi_libraries,
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="jaxamg",
    version="0.0.1",
    packages=["jaxamg"],
    ext_modules=[ext],
    setup_requires=["pybind11>=2.6.0", "python-dotenv>=1.0.0"],
)
