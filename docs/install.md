# Installation Guide

## Set Environment Variables

These variables are required for the installation process to locate dependencies (can be auto-detected if not set).

```bash
export CUDA_HOME=/usr/local/cuda
export AMGX_ROOT=/usr/local/amgx
export AMGX_BUILD=/usr/local/amgx/build
```

## Installation

### Option 1: Installation Script

The install script auto-detects your CUDA version and handles all dependencies:

```bash
# Single-GPU installation (default)
bash scripts/install.sh

# Distributed (MPI) installation
bash scripts/install.sh --mpi
```

### Option 2: Manual pip Installation

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

### Option 3: Conda Environments

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

## Post-Installation

After installation, you must configure the runtime environment to locate the AmgX and CUDA shared libraries.

Add the following to your shell profile (e.g., `~/.bashrc`):

```bash
# JAX-AMG runtime library paths
export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

For additional runtime environment variables, see [Environment Variables Reference](environ.md).