# JAX-AMG

JAX-AMG provides a JAX wrapper for NVIDIA's AmgX sparse linear solver with automatic differentiation support.

## Features

- **GPU Acceleration**: Leverages NVIDIA AmgX for fast sparse linear solves
- **Automatic Differentiation**: Full support for JAX's autodiff via custom VJP
- **JIT Compatible**: Works seamlessly with `jax.jit`
- **MPI Support**: Distributed solving across multiple GPUs with MPI

## Prerequisites

- Python 3.10+
- JAX 0.9.0+ with CUDA support
- [AmgX](https://github.com/NVIDIA/AMGX) 2.5.0+
- CUDA Toolkit 12.0+

**Additional for Distributed (MPI) Mode:**

- MPI library (e.g., OpenMPI, MPICH)
- CUDA-aware MPI (optional, for GPU-direct communication)

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

## Quick Start

A simple tridiagonal system can be solved as:

```python
import jax.numpy as jnp
import jaxamg
from jaxamg.matrices import tridiagonal_matrix

# Create a simple tridiagonal system
n = 100
A = tridiagonal_matrix(n, diagonal_value=2.0)
b = jnp.ones(n, dtype=jnp.float32)

# Solve Ax = b
x, info = jaxamg.solve(A, b)
```

### MPI Distributed Solving

A distributed 2D Poisson system can be solved with GPU-aware MPI as:

```python
from mpi4py import MPI
import jaxamg
from jaxamg.mpi_utils import partition_vector, gather_solution
from jaxamg.matrices import poisson_matrix_distributed, rhs_ones

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

# Create distributed 2D Poisson matrix
n = 16
A_local, row_start, row_end = poisson_matrix_distributed(n, n, rank, nranks)
b_local, _, _ = partition_vector(rhs_ones(n * n), rank, nranks)

# Solve in distributed mode
x_local, info = jaxamg.solve(
    A_local, b_local,
    comm=comm,
    nglobal=n * n,
    partition_info=(row_start, row_end),
    config={
        "solver": "CG",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
    }
)

# Gather solution at root rank
x_global = gather_solution(x_local, comm, root=0)
if rank == 0: print(x_global)
```

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
