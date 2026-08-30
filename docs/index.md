# JAX-AMG

**JAX-AMG** brings the power of NVIDIA's [AmgX](https://developer.nvidia.com/amgx) library to the JAX ecosystem, providing high-performance, GPU-accelerated sparse linear solvers with full support for automatic differentiation.

## Features

- **GPU-Accelerated Solvers**: Leverages NVIDIA AmgX for a broad range of GPU-accelerated sparse linear solvers, including algebraic multigrid (AMG), Krylov methods, and various variants, with flexible configuration options for solvers, smoothers, and preconditioners.
- **Automatic Differentiation**: Supports adjoint-based gradient computation and integrates seamlessly with JAX for end-to-end differentiable workflows.
- **JIT Compilation**: Built as a native JAX primitive, fully compatible with Just-in-Time compilation (`jax.jit`) for efficient, low-overhead execution.
- **Distributed Solving**: Supports multi-GPU MPI, GPU-aware communication, and JAX sharding.
- **Matrix-Free Operators**: Beyond explicit matrices, `A` can be a callable operator. The library recovers the exact sparsity pattern in a single pass by tracing the operator's computation graph, then assembles the matrix the solver needs.

## Dependencies

- CUDA Toolkit
- NVIDIA AmgX
- JAX (with CUDA support)
- mpi4py & mpi4jax (optional, for MPI support)

## Installation

```bash
pip install "jaxamg[cuda12]"   # or jaxamg[cuda13]
```

JAX-AMG compiles a native extension against your CUDA toolkit and a source build of NVIDIA AmgX, so set `CUDA_HOME` and `AMGX_ROOT` first. See the [Installation](install.md) guide for full details (MPI, conda, building from source).

## Basic Usage

```python
import jax
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import jaxamg

# Create a sparse matrix
N = 100
rows, cols = ...  # (nnz,) row and column indices
data = ...        # (nnz,) nonzero values
A = jsp.BCOO((data, jnp.stack([rows, cols], axis=1)), shape=(N, N))
b = jnp.ones(N)

# Solve Ax = b
x, info = jaxamg.solve(A, b)

print(f"Solution: {x}")
print(f"Iterations: {info['iterations']}")
print(f"Residual: {info['residual']}")
print(f"Convergence curve: {info['residual_history']}")
```

`A` can be any sparse matrix (`jax.experimental.sparse` BCSR/BCOO, SciPy sparse), a dense array, or a matrix-free callable operator.

Refer to [Examples](examples.md) for additional usage examples.

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
