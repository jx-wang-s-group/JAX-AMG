# JAX-AMG

[![PyPI](https://img.shields.io/pypi/v/jaxamg.svg?style=flat-square)](https://pypi.org/project/jaxamg/)
[![Docs](https://img.shields.io/github/actions/workflow/status/jx-wang-s-group/JAX-AMG/docs.yml?style=flat-square&label=docs)](https://jx-wang-s-group.github.io/JAX-AMG/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](https://github.com/jx-wang-s-group/JAX-AMG/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.softx.2026.102966-blue?style=flat-square)](https://doi.org/10.1016/j.softx.2026.102966)

**JAX-AMG** brings the power of NVIDIA's [AmgX](https://developer.nvidia.com/amgx) library to the JAX ecosystem, providing high-performance, GPU-accelerated sparse linear solvers with full support for automatic differentiation.

Documentation: <https://jx-wang-s-group.github.io/JAX-AMG/>

## Features

- **GPU-Accelerated Solvers**: Leverages NVIDIA AmgX for a broad range of GPU-accelerated sparse linear solvers, including algebraic multigrid (AMG), Krylov methods, and various variants, with flexible configuration options for solvers, smoothers, and preconditioners.
- **Automatic Differentiation**: Supports adjoint-based gradient computation and integrates seamlessly with JAX for end-to-end differentiable workflows.
- **JIT Compilation**: Built as a native JAX primitive, fully compatible with Just-in-Time compilation (`jax.jit`) for efficient, low-overhead execution.
- **Distributed Solving**: Supports multi-GPU MPI, GPU-aware communication, and JAX sharding.
- **Matrix-Free Operators**: Beyond explicit matrices, `A` can be a callable operator. The library recovers the exact sparsity pattern in a single pass by tracing the operator's computation graph, then assembles the matrix the solver needs.

## Prerequisites

- Python 3.10+
- JAX 0.5.0+ with CUDA support
- AmgX 2.5.0+
- CUDA Toolkit 12.0+

**Additional for Distributed (MPI) Mode:**

- MPI library (e.g., OpenMPI, MPICH)
- CUDA-aware MPI (optional, for GPU-direct communication)

## Installation

JAX-AMG is installed with pip. It compiles a native extension against a CUDA toolkit and a source build of [NVIDIA AmgX](https://github.com/NVIDIA/AMGX), so set `CUDA_HOME` and `AMGX_ROOT` first, then run the command for your CUDA version:

```bash
pip install "jaxamg[cuda12]"   # or jaxamg[cuda13]
```

At runtime, add the AmgX and CUDA libraries to your library path:

```bash
export LD_LIBRARY_PATH=$AMGX_ROOT/build:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

For distributed (MPI) mode, the install script, conda, or building from source, see the full [Installation Guide](https://jx-wang-s-group.github.io/JAX-AMG/install/).

---

## Quick Start

A simple tridiagonal system can be solved as:

```python
import jaxamg
from jaxamg.matrices import tridiagonal_matrix, rhs_ones

# Create a simple tridiagonal system
n = 100
A = tridiagonal_matrix(n, diagonal_value=2.0)
b = rhs_ones(n)

# Solve Ax = b
x, info = jaxamg.solve(A, b)
```

### MPI Distributed Solving

A distributed 2D Poisson system can be solved with GPU-aware MPI as:

```python
from mpi4py import MPI
import jaxamg
from jaxamg.mpi_utils import partition_vector, gather_vector
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
x_global = gather_vector(x_local, comm, root=0)
if rank == 0: print(x_global)
```

---

## Citation

If you use JAX-AMG in your work, please consider using the following citation:

```bibtex
@article{jaxamg2026,
  title={JAX-AMG: A GPU-accelerated differentiable sparse linear solver library for JAX},
  author={Yi Liu and Xiantao Fan and Jian-Xun Wang},
  journal={SoftwareX},
  volume={35},
  pages={102966},
  year={2026},
  doi={10.1016/j.softx.2026.102966},
}
```
