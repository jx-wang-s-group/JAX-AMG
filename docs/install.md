# Installation Guide

## Prerequisites: CUDA and AmgX

First, ensure that [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) and [AmgX](https://developer.nvidia.com/amgx) are installed. Installation details for AmgX are available in its [GitHub repository](https://github.com/NVIDIA/AMGX). For MPI support, you also need an MPI library (such as OpenMPI or MPICH) and build AmgX accordingly.

Next, set the required environment variables so the build system can locate the dependencies. If CUDA and AmgX are installed in standard locations, they may be detected automatically.

```bash
export CUDA_HOME=/usr/local/cuda
export AMGX_ROOT=/usr/local/amgx
export AMGX_BUILD=/usr/local/amgx/build
```


## Installation

=== "Installation Script"

    The install script auto-detects your CUDA version and handles all dependencies:

    ```bash
    # Single-GPU installation (default)
    bash scripts/install.sh

    # Distributed (MPI) installation
    bash scripts/install.sh --mpi
    ```

=== "Manual pip Installation"

    1. **Install JAX**:

        ```bash
        # Install JAX with CUDA support
        pip install "jax[cuda12]>=0.4.35" # For CUDA 12
        pip install "jax[cuda13]>=0.4.35" # For CUDA 13
        ```

    2. **Install JAX-AMG**:

        - **For Single-GPU Mode:**
        ```bash
        pip install -e .
        ```

        - **For Distributed (MPI) Mode:**
        ```bash
        # Install MPI library (skip if you already have one)
        # For instance, install OpenMPI via conda:
        # conda install -c conda-forge openmpi-mpicc

        # Build mpi4py from source
        pip install mpi4py --no-binary mpi4py

        # Install mpi4jax
        pip install cython
        CUDA_ROOT=$CUDA_HOME pip install mpi4jax --no-build-isolation

        # Install jaxamg with MPI dependencies
        pip install -e ".[mpi]"
        ```

=== "Conda Environments"

    Use the provided conda environment files.

    ```bash
    # Single-GPU mode
    conda env create -f environment.yml
    conda activate jaxamg

    # Distributed mode
    conda env create -f environment-mpi.yml
    conda activate jaxamg-mpi
    ```

    The environment files use `jax[cuda13]`. If you need CUDA 12 instead, edit the `.yml` files and replace `cuda13` with `cuda12` before creating the environment.

    If `mpi4jax` was built without CUDA support, you need to rebuild it after creating the conda environment:

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