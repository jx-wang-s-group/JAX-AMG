# JAX-AMG

[![Docs](https://img.shields.io/github/actions/workflow/status/jx-wang-s-group/JAX-AMG/docs.yml?style=flat-square&label=docs)](https://jx-wang-s-group.github.io/JAX-AMG/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg?style=flat-square)](https://www.python.org/)
[![arXiv](https://img.shields.io/badge/arXiv-2606.09001-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2606.09001)

**JAX-AMG** brings the power of NVIDIA's [AmgX](https://developer.nvidia.com/amgx) library to the JAX ecosystem, providing high-performance, GPU-accelerated sparse linear solvers with full support for automatic differentiation.

Documentation: <https://jx-wang-s-group.github.io/JAX-AMG/>

## Features

- **GPU-Accelerated Solvers**: Leverages NVIDIA AmgX for a broad range of GPU-accelerated sparse linear solvers, including algebraic multigrid (AMG), Krylov methods, and various variants, with flexible configuraiton options for solvers, smoothers, and preconditioners.
- **Automatic Differentiation**: Supports adjoint-based gradient computation and integrates seamlessly with JAX for end-to-end differentiable workflows.
- **JIT Compilation**: Built as a native JAX primitive, fully compatible with Just-in-Time compilation (`jax.jit`) for efficient, low-overhead execution.
- **MPI Support**: Enables distributed linear solves across multiple GPUs, with GPU-aware MPI support.
- **Matrix-Free Operators**: Beyond explicit matrices, `A` can be a callable operator. The library recovers the exact sparsity pattern in a single pass by tracing the operator's computation graph, then assembles the matrix the solver needs.

## Prerequisites

- Python 3.10+
- JAX 0.4.35+ with CUDA support
- AmgX 2.5.0+
- CUDA Toolkit 12.0+

**Additional for Distributed (MPI) Mode:**

- MPI library (e.g., OpenMPI, MPICH)
- CUDA-aware MPI (optional, for GPU-direct communication)

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

## Citation

If you use JAX-AMG in your work, please consider using the following citation ([arXiv:2606.09001](https://arxiv.org/abs/2606.09001)):

```bibtex
@misc{jaxamg2026,
      title={JAX-AMG: A GPU-Accelerated Differentiable Sparse Linear Solver Library for JAX},
      author={Yi Liu and Xiantao Fan and Jian-Xun Wang},
      year={2026},
      eprint={2606.09001},
      archivePrefix={arXiv},
      primaryClass={cs.MS},
      url={https://arxiv.org/abs/2606.09001},
}
```