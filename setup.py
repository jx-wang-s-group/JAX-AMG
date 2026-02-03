import os
import subprocess
from setuptools import setup, Extension, find_packages
import warnings

# Auto-detection functions


def find_cuda():
    # Print CUDA detection attempt
    print("\033[1;34m[setup.py] Detecting CUDA installation...\033[0m")
    # Try environment variable
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and os.path.exists(cuda_home):
        return cuda_home

    # Try which nvcc
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"], text=True).strip()
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        if os.path.exists(cuda_home):
            return cuda_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Common locations with version globbing
    import glob

    cuda_paths = glob.glob("/usr/local/cuda-*") + ["/usr/local/cuda", "/usr/cuda"]
    # Sort to prefer newest version
    cuda_paths.sort(reverse=True)
    for path in cuda_paths:
        if os.path.exists(path) and os.path.isdir(path):
            return path

    return None


def find_amgx():
    print("\033[1;34m[setup.py] Detecting AMGX installation...\033[0m")
    # Try environment variables
    amgx_root = os.environ.get("AMGX_ROOT")
    amgx_build = os.environ.get("AMGX_BUILD")

    if amgx_root and os.path.exists(amgx_root):
        if not amgx_build:
            # Try to guess build directory
            guess = os.path.join(amgx_root, "build")
            if os.path.exists(guess):
                amgx_build = guess
        return amgx_root, amgx_build

    # Try common locations
    home = os.path.expanduser("~")
    common_amgx_parents = [
        home,
        "/opt",
        "/usr/local",
    ]

    for parent in common_amgx_parents:
        if os.path.exists(parent):
            amgx_path = os.path.join(parent, "amgx")
            if os.path.exists(amgx_path):
                # Found something named amgx, check for include/amgx_c.h
                if os.path.exists(os.path.join(amgx_path, "include", "amgx_c.h")):
                    # Found it! Now check for build directory
                    guess = os.path.join(amgx_path, "build")
                    if os.path.exists(guess):
                        return amgx_path, guess
                    # If include exists but build doesn't, still return root
                    return amgx_path, None

    return None, None


# Find MPI C++ compiler
def find_mpicxx():
    print("\033[1;34m[setup.py] Detecting MPI C++ compiler...\033[0m")
    # Try environment variable
    mpicxx_bin = os.environ.get("MPICXX")
    if mpicxx_bin:
        # Check if it exists in PATH or as a file
        if os.path.isabs(mpicxx_bin) and os.path.exists(mpicxx_bin):
            return mpicxx_bin
        # Try to locate in PATH
        try:
            mpicxx_path = subprocess.check_output(
                ["which", mpicxx_bin], text=True
            ).strip()
            if os.path.exists(mpicxx_path):
                return mpicxx_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    # Try common MPI compiler names
    for candidate in ["mpicxx", "mpiCC", "mpic++"]:
        try:
            mpicxx_path = subprocess.check_output(
                ["which", candidate], text=True
            ).strip()
            if os.path.exists(mpicxx_path):
                return mpicxx_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


# Defer heavy imports until after dependency checks
def get_build_vars():
    print("\033[1;34m[setup.py] Gathering build configuration...\033[0m")
    try:
        import pybind11
    except ImportError:
        raise RuntimeError(
            "pybind11 is required to build this package. Please install it first."
        )
    try:
        from jax import ffi
    except ImportError:
        raise RuntimeError(
            "jax is required to build this package. Please install it first."
        )

    CUDA_HOME = find_cuda()
    print(f"\033[1;34m[setup.py] CUDA_HOME: {CUDA_HOME}\033[0m")
    if not CUDA_HOME:
        raise RuntimeError(
            "CUDA_HOME not found. Please ensure CUDA Toolkit is installed, "
            "or set CUDA_HOME environment variable manually."
        )

    AMGX_ROOT, AMGX_BUILD = find_amgx()
    print(f"\033[1;34m[setup.py] AMGX_ROOT: {AMGX_ROOT}\033[0m")
    print(f"\033[1;34m[setup.py] AMGX_BUILD: {AMGX_BUILD}\033[0m")
    if not AMGX_ROOT or not AMGX_BUILD:
        warnings.warn(
            f"AMGX not found (AMGX_ROOT: {AMGX_ROOT}, AMGX_BUILD: {AMGX_BUILD}). "
            "Building without AMGX support."
        )

    XLA_FFI_INCLUDE = ffi.include_dir()

    # Get MPI compile and link flags
    mpicxx_bin = find_mpicxx()
    print(f"\033[1;34m[setup.py] MPICXX: {mpicxx_bin}\033[0m")
    if mpicxx_bin:
        try:
            mpi_compile_output = subprocess.check_output(
                [mpicxx_bin, "--showme:compile"], text=True
            ).strip()
            mpi_compile_flags = mpi_compile_output.split()

            mpi_link_output = subprocess.check_output(
                [mpicxx_bin, "--showme:link"], text=True
            ).strip()
            mpi_link_flags = mpi_link_output.split()

            mpi_include_dirs = [
                flag[2:] for flag in mpi_compile_flags if flag.startswith("-I")
            ]
            mpi_library_dirs = [
                flag[2:] for flag in mpi_link_flags if flag.startswith("-L")
            ]
            mpi_libraries = [
                flag[2:] for flag in mpi_link_flags if flag.startswith("-l")
            ]
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.warn(
                "mpicxx found but could not extract flags. Building without MPI support."
            )
            mpi_include_dirs = []
            mpi_library_dirs = []
            mpi_libraries = []
    else:
        warnings.warn("mpicxx not found. Building without MPI support.")
        mpi_include_dirs = []
        mpi_library_dirs = []
        mpi_libraries = []

    include_dirs = [
        pybind11.get_include(),
        XLA_FFI_INCLUDE,
    ]
    if AMGX_ROOT:
        include_dirs.append(os.path.join(AMGX_ROOT, "include"))
    include_dirs.append(os.path.join(CUDA_HOME, "include"))
    include_dirs += mpi_include_dirs

    library_dirs = []
    if AMGX_BUILD:
        library_dirs.append(AMGX_BUILD)
    library_dirs.append(os.path.join(CUDA_HOME, "lib64"))
    library_dirs += mpi_library_dirs

    runtime_library_dirs = list(library_dirs)

    libraries = []
    if AMGX_BUILD:
        libraries.append("amgxsh")
    libraries.append("cudart")
    libraries += mpi_libraries

    print(f"\033[1;34m[setup.py] include_dirs: {include_dirs}\033[0m")
    print(f"\033[1;34m[setup.py] library_dirs: {library_dirs}\033[0m")
    print(f"\033[1;34m[setup.py] runtime_library_dirs: {runtime_library_dirs}\033[0m")
    print(f"\033[1;34m[setup.py] libraries: {libraries}\033[0m")

    return {
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "runtime_library_dirs": runtime_library_dirs,
        "libraries": libraries,
    }


build_vars = get_build_vars()

ext = Extension(
    "jaxamg._amgx",
    sources=["jaxamg/amgx_custom_call.cc"],
    include_dirs=build_vars["include_dirs"],
    library_dirs=build_vars["library_dirs"],
    runtime_library_dirs=build_vars["runtime_library_dirs"],
    libraries=build_vars["libraries"],
    extra_compile_args=["-O3", "-std=c++17"],
)

setup(
    name="jaxamg",
    version="0.0.1",
    packages=["jaxamg"],
    ext_modules=[ext],
    setup_requires=["pybind11>=2.6.0", "jax"],
    install_requires=["pybind11>=2.6.0", "jax"],
    python_requires=">=3.8",
)
