# Installation Guide

## Prerequisites: CUDA and AmgX

JAX-AMG compiles a native extension against CUDA and AmgX, so a build toolchain
and these libraries must be in place before installing:

- Python 3.10+ and a C++ compiler
- [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) 12.0+
- [NVIDIA AmgX](https://developer.nvidia.com/amgx) 2.5.0+, built from source (see the [build instructions](https://github.com/NVIDIA/AMGX#quickstart))
- For distributed (MPI) mode: an MPI library (e.g., OpenMPI or MPICH), with AmgX built against it. A CUDA-aware MPI build is optional but recommended for GPU-direct communication.

Next, set the required environment variables so the build system can locate the dependencies. If CUDA and AmgX are installed in standard locations, they may be detected automatically.

```bash
export CUDA_HOME=/usr/local/cuda
export AMGX_ROOT=/usr/local/amgx
export AMGX_BUILD=/usr/local/amgx/build   # Optional (defaults to $AMGX_ROOT/build)
```


## Installation

=== "pip (PyPI)"

    Run the command for your CUDA version:

    ```bash
    pip install "jaxamg[cuda12]"   # CUDA 12
    pip install "jaxamg[cuda13]"   # CUDA 13
    ```

    **Distributed (MPI) mode.** Build the MPI bindings against your own MPI first
    (a generic `mpi4py` wheel may not match the MPI AmgX was built with), then
    install the `mpi` extra:

    ```bash
    # Build mpi4py against your system MPI
    pip install mpi4py --no-binary mpi4py
    # mpi4jax (built with nanobind)
    pip install nanobind
    CUDA_ROOT=$CUDA_HOME pip install mpi4jax --no-build-isolation
    # JAX-AMG with the matching CUDA extra + MPI
    pip install "jaxamg[cuda13,mpi]"
    ```

=== "Installation Script (from source)"

    Clone the repository and run the script, which auto-detects your CUDA version
    and handles all dependencies:

    ```bash
    git clone https://github.com/jx-wang-s-group/JAX-AMG.git
    cd JAX-AMG
    bash scripts/install.sh          # add --mpi for distributed mode
    ```

=== "Manual (from source)"

    ```bash
    git clone https://github.com/jx-wang-s-group/JAX-AMG.git
    cd JAX-AMG

    # Install JAX with CUDA support (or jax[cuda13])
    pip install "jax[cuda12]>=0.5.0"

    # Single-GPU
    pip install -e .

    # Distributed (MPI): build mpi4py/mpi4jax first (see the pip tab), then:
    pip install -e ".[mpi]"
    ```

=== "Conda Environments"

    ```bash
    git clone https://github.com/jx-wang-s-group/JAX-AMG.git
    cd JAX-AMG

    # Single-GPU mode
    conda env create -f environment.yml
    conda activate jaxamg

    # Distributed mode
    conda env create -f environment-mpi.yml
    conda activate jaxamg-mpi
    ```

    The environment files use `jax[cuda13]`. If you need CUDA 12 instead, edit the `.yml` files and replace `cuda13` with `cuda12` before creating the environment.

    If `mpi4jax` was built without CUDA support, rebuild it after creating the conda environment:

    ```bash
    CUDA_ROOT=$CUDA_HOME pip install mpi4jax --no-build-isolation --no-cache-dir --force-reinstall
    ```

## Post-Installation Setup

After installation, you must configure the runtime environment to locate the AmgX and CUDA shared libraries.

Add the following to your shell profile (e.g., `~/.bashrc`):

```bash
# JAX-AMG runtime library paths
export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

For additional runtime environment variables, see [Environment Variables Reference](environ.md).