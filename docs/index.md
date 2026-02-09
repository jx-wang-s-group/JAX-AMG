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
