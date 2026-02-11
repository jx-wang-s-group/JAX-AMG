# JAX-AMG

**JAX-AMG** provides JAX bindings for the NVIDIA AmgX sparse linear solver, enabling high-performance linear solving on GPUs with automatic differentiation support.

## Features

- **High Performance**: Leverages NVIDIA AmgX for state-of-the-art algebraic multigrid solvers.
- **JAX Integration**: Fully compatible with JAX transformations (`jit`, `grad`, `vmap`).
- **Automatic Differentiation**: Compute gradients through the linear solve using implicit differentiation.
- **MPI Support**: Distributed solving across multiple GPUs and nodes.

## Dependencies

- CUDA Toolkit
- NVIDIA AmgX
- JAX (with CUDA support)
- mpi4py & mpi4jax (optional, for MPI support)


## Basic Usage

```python
import jax
import jax.numpy as jnp
import jax.scipy.sparse as jsp
import jaxamg

# Create a sparse matrix
N = 100
rows, cols = ...
data = ...
A = jsp.BCOO((data, (rows, cols)), shape=(N, N))
b = jnp.ones(N)

# Solve Ax = b
x, info = jaxamg.solve(A, b)

print(f"Solution: {x}")
print(f"Iterations: {info['iterations']}")
print(f"Residual: {info['residual']}")
```

## Environment Variables

| Variable | When Needed | Description |
|----------|------------|------------|
| `AMGX_ROOT` | Installation | Path to AmgX source directory (auto-detected if not set) |
| `AMGX_BUILD` | Installation | Path to AmgX build dir (defaults to `$AMGX_ROOT/build`) |
| `CUDA_HOME` | Installation | Path to CUDA toolkit (auto-detected if not set) |
| `MPI_HOME` | Installation | Path to custom MPI installation (optional) |
| `JAXAMG_ENABLE_MPI` | Installation | Force MPI linkage (optional, usually auto-detected) |
| `LD_LIBRARY_PATH` | Runtime | Need to include AmgX and CUDA library paths |
| `JAXAMG_CACHE_SIZE` | Runtime | Native AmgX resource cache size: `0` disables resource caching (isolated mode), `>0` (default is `1`) enables caching for performance improvement |
| `OMPI_MCA_opal_cuda_support` | Runtime | Set to `true` for GPU-aware MPI (when using OpenMPI) |
| `MPI4JAX_USE_CUDA_MPI` | Runtime | Set to `1` for GPU-aware MPI (for mpi4jax) |
