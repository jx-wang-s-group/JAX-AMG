# JAX-AMG Installation Guide

JAX-AMG provides JAX bindings for NVIDIA's AmgX sparse linear solver with automatic differentiation support.

## Prerequisites

| Requirement | Version |
|-------------|---------|
| NVIDIA GPU | CC 6.0+ |
| CUDA Toolkit | 12.0+ |
| Python | 3.10+ |
| [AmgX](https://github.com/NVIDIA/AMGX) | 2.5+ |

**Additional for Distributed (MPI) Mode:**

| Requirement | Notes |
|-------------|-------|
| OpenMPI/MPICH | MPI library |
| CUDA-aware MPI | Optional, for GPU-direct communication |

---

## Installation

### Step 1: Set Environment Variables

These variables are required for the installation process to locate dependencies (can be auto-detected if not set).

```bash
export CUDA_HOME=/usr/local/cuda
export AMGX_ROOT=/usr/local/amgx
export AMGX_BUILD=/usr/local/amgx/build
```

### Step 2: Choose Installation Method

#### Option 1: Installation Script

The install script auto-detects your CUDA version and handles all dependencies:

```bash
# Single-GPU installation (default)
bash scripts/install.sh

# Distributed (MPI) installation
bash scripts/install.sh --mpi
```

#### Option 2: Manual pip Installation

1. **Install JAX**:
   ```bash
   # Install JAX with CUDA support
   pip install "jax[cuda12]>=0.4.35" # For CUDA 12
   pip install "jax[cuda13]>=0.4.35" # For CUDA 13
   ```

2. **Install JAX-AMG**:

   **For Single-GPU Mode:**
   ```bash
   pip install -e .
   ```

   **For Distributed (MPI) Mode:**
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

#### Option 3: Conda Environments

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

### Post-Installation

After installation, you must configure the runtime environment to locate the AmgX and CUDA shared libraries.

Add the following to your shell profile (e.g., `~/.bashrc`):

```bash
# JAX-AMG runtime library paths
export LD_LIBRARY_PATH=$AMGX_BUILD:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

For additional environment variables for GPU-aware MPI configuration, see the [GPU-Aware MPI](#gpu-aware-mpi) section below.


---

## GPU-Aware MPI

To enable GPU-aware MPI, first verify that your MPI library was compiled with CUDA support. For OpenMPI, you can check with:

```bash
ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
```

Next, set the necessary environment variables for GPU-aware operation:

```bash
export OMPI_MCA_opal_cuda_support=true   # required for OpenMPI
export MPI4JAX_USE_CUDA_MPI=1            # for mpi4jax
```

When using AmgX with GPU-aware MPI, specify the communicator in the configuration:

```python
config = {
    "communicator": "MPI_DIRECT",
    ...
}
```

---

## Environment Variables Reference

| Variable | When Needed | Description |
|----------|------------|------------|
| `AMGX_ROOT` | Installation | Path to AmgX source directory (auto-detected if not set) |
| `AMGX_BUILD` | Installation | Path to AmgX build dir (defaults to `$AMGX_ROOT/build`) |
| `CUDA_HOME` | Installation | Path to CUDA toolkit (auto-detected if not set) |
| `MPI_HOME` | Installation | Path to custom MPI installation (optional) |
| `JAXAMG_ENABLE_MPI` | Installation | Force MPI linkage (optional, usually auto-detected) |
| `LD_LIBRARY_PATH` | Runtime | Need to include AmgX and CUDA library paths |
| `OMPI_MCA_opal_cuda_support` | Runtime | Set to `true` for GPU-aware MPI (when using OpenMPI) |
| `MPI4JAX_USE_CUDA_MPI` | Runtime | Set to `1` for GPU-aware MPI (for mpi4jax) |
