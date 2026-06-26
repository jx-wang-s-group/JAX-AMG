"""
JAX-AMG package setup with native extension for AmgX bindings.

This setup.py builds the C++ extension that bridges JAX FFI with NVIDIA AmgX.

Environment Variables:
    CUDA_HOME: Path to CUDA toolkit (auto-detected if not set)
    AMGX_ROOT: Path to AmgX source directory (required)
    AMGX_BUILD: Path to AmgX build directory (defaults to AMGX_ROOT/build)
    JAXAMG_ENABLE_MPI: Override to force MPI linkage (usually not needed)
"""

import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def _fail(message):
    """Print an actionable error and abort the build without a Python traceback."""
    sys.stdout.flush()
    print(f"\n[jaxamg] ERROR: {message}\n", file=sys.stderr)
    raise SystemExit(1)


def find_cuda() -> str | None:
    """Auto-detect CUDA installation path."""
    print("\033[1;34m[setup.py] Detecting CUDA installation...\033[0m")

    # Try environment variable first
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home and Path(cuda_home).exists():
        return cuda_home

    # Try which nvcc
    try:
        nvcc_path = subprocess.check_output(["which", "nvcc"], text=True).strip()
        cuda_home = str(Path(nvcc_path).parent.parent)
        if Path(cuda_home).exists():
            return cuda_home
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Common locations with version globbing
    import glob

    cuda_paths = glob.glob("/usr/local/cuda-*") + ["/usr/local/cuda", "/usr/cuda"]
    cuda_paths.sort(reverse=True)  # Prefer newest version
    for path in cuda_paths:
        if Path(path).is_dir():
            return path

    return None


def find_amgx() -> tuple[str | None, str | None]:
    """Auto-detect AmgX installation path."""
    print("\033[1;34m[setup.py] Detecting AmgX installation...\033[0m")

    amgx_root = os.environ.get("AMGX_ROOT")
    amgx_build = os.environ.get("AMGX_BUILD")

    if amgx_root and Path(amgx_root).exists():
        if not amgx_build:
            guess = Path(amgx_root) / "build"
            if guess.exists():
                amgx_build = str(guess)
        return amgx_root, amgx_build

    # Try common locations
    common_dirs = [
        Path.home(),
        Path("/opt"),
        Path("/usr/local"),
    ]

    for parent in common_dirs:
        amgx_path = parent / "amgx"
        if amgx_path.exists() and (amgx_path / "include" / "amgx_c.h").exists():
            build_path = amgx_path / "build"
            return str(amgx_path), str(build_path) if build_path.exists() else None

    return None, None


def find_mpicxx() -> str | None:
    """Find MPI C++ compiler if available."""
    print("\033[1;34m[setup.py] Detecting MPI C++ compiler...\033[0m")

    mpicxx_env = os.environ.get("MPICXX")
    if mpicxx_env:
        if Path(mpicxx_env).exists():
            return mpicxx_env
        try:
            result = subprocess.check_output(["which", mpicxx_env], text=True).strip()
            if Path(result).exists():
                return result
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Try common MPI compiler names
    for candidate in ["mpicxx", "mpiCC", "mpic++"]:
        try:
            result = subprocess.check_output(["which", candidate], text=True).strip()
            if Path(result).exists():
                return result
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    return None


def get_mpi_flags(mpicxx_bin: str) -> tuple[list[str], list[str], list[str]]:
    """Extract MPI compile and link flags from MPI compiler wrapper."""
    try:
        compile_output = subprocess.check_output(
            [mpicxx_bin, "--showme:compile"], text=True
        ).strip()
        compile_flags = compile_output.split()

        link_output = subprocess.check_output(
            [mpicxx_bin, "--showme:link"], text=True
        ).strip()
        link_flags = link_output.split()

        include_dirs = [f[2:] for f in compile_flags if f.startswith("-I")]
        library_dirs = [f[2:] for f in link_flags if f.startswith("-L")]
        libraries = [f[2:] for f in link_flags if f.startswith("-l")]

        return include_dirs, library_dirs, libraries
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [], [], []


def get_build_config() -> dict:
    """Gather all build configuration."""
    print("\033[1;34m[setup.py] Gathering build configuration...\033[0m")

    # Check for required build dependencies
    try:
        import pybind11
    except ImportError:
        _fail(
            "pybind11 is required to build this package. "
            "Install with: pip install pybind11>=2.10.0"
        )

    try:
        from jax import ffi
    except ImportError:
        _fail(
            "jax is required to build this package. "
            "Install with: pip install jax>=0.5.0"
        )

    # Find CUDA
    cuda_home = find_cuda()
    print(f"\033[1;34m[setup.py] CUDA_HOME: {cuda_home}\033[0m")
    if not cuda_home:
        _fail(
            "CUDA not found. Please install CUDA Toolkit and either:\n"
            "  1. Set CUDA_HOME environment variable, or\n"
            "  2. Ensure nvcc is in your PATH"
        )

    # Find AmgX
    amgx_root, amgx_build = find_amgx()
    print(f"\033[1;34m[setup.py] AMGX_ROOT: {amgx_root}\033[0m")
    print(f"\033[1;34m[setup.py] AMGX_BUILD: {amgx_build}\033[0m")
    if not amgx_root or not amgx_build:
        _fail(
            "AmgX not found. Please:\n"
            "  1. Set AMGX_ROOT to the AmgX source directory\n"
            "  2. Set AMGX_BUILD to the AmgX build directory (or build in AMGX_ROOT/build)\n"
            "  See https://github.com/NVIDIA/AMGX for details on building AmgX."
        )

    # Build include and library paths
    include_dirs = [
        pybind11.get_include(),
        ffi.include_dir(),
        str(Path(amgx_root) / "include"),
        str(Path(cuda_home) / "include"),
    ]

    library_dirs = [
        amgx_build,
        str(Path(cuda_home) / "lib64"),
    ]

    libraries = ["amgxsh", "cudart", "cusparse"]

    # Check if AmgX was built with MPI by inspecting for undefined MPI symbols
    def amgx_requires_mpi() -> bool:
        """Check if libamgxsh.so has undefined MPI symbols (requires MPI at runtime)."""
        libamgx = Path(amgx_build) / "libamgxsh.so"
        if not libamgx.exists():
            return False
        try:
            # Use nm to check for undefined MPI symbols
            # AmgX uses dynamic MPI loading, so ldd won't show MPI
            nm_output = subprocess.check_output(
                ["nm", "-D", str(libamgx)], text=True, stderr=subprocess.DEVNULL
            )
            # Look for undefined MPI symbols (lines starting with U)
            return "U MPI_" in nm_output
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    amgx_needs_mpi = amgx_requires_mpi()
    mpi_pref = os.environ.get("JAXAMG_ENABLE_MPI", "").strip().lower()
    env_enable_mpi = mpi_pref in ("1", "true", "yes", "on")
    env_disable_mpi = mpi_pref in ("0", "false", "no", "off")

    # Resolve the MPI build mode:
    #   JAXAMG_ENABLE_MPI=0/false/no/off -> force a clean non-MPI build, even if
    #     AmgX itself was built with MPI (no mpi.h, no libmpi linkage, MPI entry
    #     points raise at runtime).
    #   JAXAMG_ENABLE_MPI=1/true/yes/on  -> force MPI on.
    #   unset                            -> auto: on iff AmgX requires MPI.
    if env_disable_mpi:
        enable_mpi = False
    else:
        enable_mpi = env_enable_mpi or amgx_needs_mpi

    if env_disable_mpi:
        print(
            "\033[1;34m[setup.py] MPI explicitly disabled (JAXAMG_ENABLE_MPI=0)\033[0m"
        )
        if amgx_needs_mpi:
            print(
                "\033[1;33m[setup.py] Note: linked AmgX (libamgxsh.so) has undefined "
                "MPI symbols, so libmpi must still be loadable at runtime even though "
                "jaxamg's own MPI paths are compiled out.\033[0m"
            )
    elif amgx_needs_mpi and not env_enable_mpi:
        print(
            "\033[1;33m[setup.py] Auto-detected: AmgX was built with MPI support\033[0m"
        )

    if enable_mpi:
        mpicxx_bin = find_mpicxx()
        print(f"\033[1;34m[setup.py] MPICXX: {mpicxx_bin}\033[0m")

        if mpicxx_bin:
            mpi_includes, mpi_libdirs, mpi_libs = get_mpi_flags(mpicxx_bin)
            include_dirs.extend(mpi_includes)
            library_dirs.extend(mpi_libdirs)
            libraries.extend(mpi_libs)
            print("\033[1;32m[setup.py] MPI support enabled\033[0m")
        else:
            if amgx_needs_mpi:
                print(
                    "\033[1;31m[setup.py] ERROR: AmgX requires MPI but mpicxx not found.\n"
                    "  Please install MPI (e.g., 'apt install libopenmpi-dev') or\n"
                    "  rebuild AmgX without MPI: cmake .. -DAMGX_NO_MPI=ON\033[0m"
                )
            else:
                print(
                    "\033[1;33m[setup.py] WARNING: JAXAMG_ENABLE_MPI=1 but mpicxx not found. "
                    "Building without MPI linkage.\033[0m"
                )
    else:
        print(
            "\033[1;34m[setup.py] MPI support disabled (AmgX built without MPI)\033[0m"
        )

    runtime_library_dirs = list(library_dirs)

    # Compile the MPI code paths only when MPI is enabled. Without this macro the
    # extension never includes <mpi.h> or references MPI symbols, so it builds and
    # links cleanly on machines without an MPI development toolchain.
    define_macros = [("JAXAMG_WITH_MPI", "1")] if enable_mpi else []

    print(f"\033[1;34m[setup.py] include_dirs: {include_dirs}\033[0m")
    print(f"\033[1;34m[setup.py] library_dirs: {library_dirs}\033[0m")
    print(f"\033[1;34m[setup.py] libraries: {libraries}\033[0m")
    print(f"\033[1;34m[setup.py] define_macros: {define_macros}\033[0m")

    return {
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "runtime_library_dirs": runtime_library_dirs,
        "libraries": libraries,
        "define_macros": define_macros,
    }


class BuildExt(build_ext):
    """Resolve CUDA/AmgX/MPI paths at build time rather than at import time.

    Keeping the heavy (and failure-prone) native-dependency detection inside
    ``build_extensions()`` means commands that do not compile the extension --
    notably ``sdist`` and metadata generation -- succeed without CUDA/AmgX
    present. Only an actual ``build_ext`` triggers detection, so the source
    distribution can be built and published from a machine without a GPU
    toolchain (e.g. CI).
    """

    def build_extensions(self) -> None:
        cfg = get_build_config()
        for ext in self.extensions:
            ext.include_dirs = cfg["include_dirs"] + list(ext.include_dirs)
            ext.library_dirs = cfg["library_dirs"] + list(ext.library_dirs)
            ext.runtime_library_dirs = cfg["runtime_library_dirs"] + list(
                ext.runtime_library_dirs or []
            )
            ext.libraries = cfg["libraries"] + list(ext.libraries)
            ext.define_macros = list(ext.define_macros or []) + cfg["define_macros"]
        super().build_extensions()


# C++ extension module. The native dependency paths (CUDA/AmgX/MPI) are filled
# in at build time by BuildExt so that sdist/metadata commands work without
# CUDA/AmgX installed.
ext_module = Extension(
    "jaxamg._amgx",
    sources=["jaxamg/_amgx.cc"],
    extra_compile_args=["-O3", "-std=c++17"],
    language="c++",
)

# Setup (metadata now in pyproject.toml)
setup(
    ext_modules=[ext_module],
    cmdclass={"build_ext": BuildExt},
)
