# Examples

This page collects common JAX-AMG usage patterns. For complete runnable scripts, see the `demo/` directory.

## Single-GPU mode

### Solving a matrix system

The input matrix can be a sparse matrix (from JAX or SciPy) of various types, or a dense matrix. Internally, all formats are converted to a `jax.experimental.sparse.BCSR` sparse matrix.

```python
import jax.numpy as jnp
import jaxamg
from jaxamg.matrices import rhs_one, tridiagonal_matrix

n = 100
A = tridiagonal_matrix(n, diagonal_value=4.0)
b = rhs_ones(100)

x, info = jaxamg.solve(A, b, solver="CG")
print("Solution:", x)
print("Iterations:", info["iterations"])
print("Residual:", info["residual"])
```

### Solving with an operator

You can also solve using a callable operator instead of an explicit matrix.

Note: The solve still requires an internal matrix representation, so this is not fully matrix-free backend execution.

```python
import jaxamg
from jaxamg.matrices import rhs_ones, tridiagonal_operator

n = 128
b = rhs_ones(n)

op = tridiagonal_operator(diagonal_value=4.0)
x, info = jaxamg.solve(op, b, solver="CG")
print("Solution:", x)
print("Iterations:", info["iterations"])
print("Residual:", info["residual"])
```

### Custom solver configuration

You can supply a custom solver configuration:

```python
import jax.numpy as jnp
import jaxamg
from jaxamg.matrices import poisson_matrix

n = 128
A = poisson_matrix(16, skew=0.5)
b = jnp.ones(16 * 16, dtype=jnp.float32)

x, info = jaxamg.solve(
    A,
    b,
    config={
        "solver": "PBICGSTAB",
        "preconditioner": {
            "solver": "AMG",
            "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.9},
            "presweeps": 2,
            "postsweeps": 2,
            "coarse_solver": "NOSOLVER",
            "max_levels": 100,
            "cycle": "V",
        },
        "tolerance": 1e-8,
        "max_iters": 100,
        "norm": "L2",
    }
)
```

See [Solver Configuration](config.md) for full details.

### Optimization via auto differentiation

```python
import jax
import jax.numpy as jnp
from jaxamg import solve
from jaxamg.matrices import rhs_ones, tridiagonal_matrix

n = 64
A = tridiagonal_matrix(n, diagonal_value=4.0)
b = rhs_ones(n)

def loss(b_vec):
    x, _ = solve(A, b_vec, solver="CG")
    return jnp.sum(x * x)

grad_fn = jax.jit(jax.grad(loss))
lr = 1e-2

for _ in range(10):
    g = grad_fn(b)
    b = b - lr * g
```


### Optimization with color caching for operator

For parameterized operators, compute coloring once and reuse it during optimization.

```python
import jax
import jax.numpy as jnp
import jaxamg
from jaxamg.matrices import rhs_ones, tridiagonal_operator

n = 64
b = rhs_ones(n)
true_diag = 4.0
x_target, _ = jaxamg.solve(tridiagonal_operator(true_diag), b, solver="CG")

# Cache coloring once using an operator with identical sparsity structure.
coloring = jaxamg.cache_coloring(tridiagonal_operator(4.5), shape=n)

@jax.jit
def loss(diag):
    op = tridiagonal_operator(diag)
    A = jaxamg.with_cache(op, coloring=coloring, is_symmetric=True)
    x_pred, _ = jaxamg.solve(A, b, solver="CG")
    return jnp.mean((x_pred - x_target) ** 2)

grad_fn = jax.jit(jax.grad(loss))
diag = 4.5
for _ in range(50):
    diag = diag - 2.0 * grad_fn(diag)
```


## MPI distributed mode

Launch scripts with MPI:

```bash
mpirun -n <num_procs> python your_script.py
```

### Solving a distributed matrix system

```python
from mpi4py import MPI
import jaxamg
from jaxamg.mpi_utils import partition_vector, gather_solution
from jaxamg.matrices import poisson_matrix_distributed, rhs_ones

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

n = 16
A_local, row_start, row_end = poisson_matrix_distributed(n, n, rank, nranks)
b_local, _, _ = partition_vector(rhs_ones(n * n), rank, nranks)

x_local, info = jaxamg.solve(
    A_local,
    b_local,
    comm=comm,
    nglobal=n * n,
    partition_info=(row_start, row_end),
    config={
        "solver": "CG",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
    },
)

x_global = gather_solution(x_local, comm, root=0)
if rank == 0:
    print("Solution:", x_global)

comm.Barrier()
jaxamg.finalize()
```

### Distributed optimization

Each rank computes local loss/gradient, then uses MPI reductions to form global metrics.

```python
import jax
import jax.numpy as jnp
from mpi4py import MPI
import jaxamg
from jaxamg.matrices import tridiagonal_matrix_distributed

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

n_global = 64
A_local, row_start, row_end = tridiagonal_matrix_distributed(
    n_global, rank, nranks, diagonal_value=4.0, dtype=jnp.float64
)
b_local = jnp.ones(row_end - row_start, dtype=jnp.float64)

config = {"solver": "CG", "communicator": "MPI_DIRECT"}
mpi_cache = jaxamg.cache_mpi_metadata(
    config, comm, n_global, (row_start, row_end), A_local
)

def loss_local(b_loc):
    A = jaxamg.with_cache(A_local, mpi=mpi_cache, is_symmetric=True)
    x_loc, _ = jaxamg.solve(A, b_loc)
    return jnp.sum(x_loc * x_loc)

loss_grad = jax.jit(jax.value_and_grad(loss_local))

for _ in range(10):
    l_loc, g_loc = loss_grad(b_local)
    l_global = comm.allreduce(float(l_loc), op=MPI.SUM)
    b_local = b_local - 1e-2 * g_loc
    if rank == 0:
        print("Global loss:", l_global)

comm.Barrier()
jaxamg.finalize()
```

