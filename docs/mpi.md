# MPI Guide

This page explains how to run JAX-AMG in distributed mode with MPI across multiple GPUs and nodes.

## Prerequisites

Before running MPI jobs, make sure you have:

- A working MPI installation (`mpirun` available).
- `mpi4py` and `mpi4jax` installed in the same environment as `jaxamg`.
- CUDA-enabled JAX and a working NVIDIA driver/toolkit.
- AmgX installed and visible through `LD_LIBRARY_PATH`.

See the [Installation Guide](install.md) for more details.

## Launching MPI Jobs

To run an MPI job, use:

```bash
mpirun -n 2 python demo/mpi_tridiagonal_matrix_optimization.py
```

You can set `CUDA_VISIBLE_DEVICES` to control which GPUs are visible to each process.

## Distributed Solve Pattern

JAX-AMG MPI solves follow this pattern:

1. Initialize communicator (`MPI.COMM_WORLD`).
2. Build a local matrix partition per rank.
3. Build local right-hand-side vector.
4. Call `jaxamg.solve(...)` with `comm`, `nglobal`, and `partition_info`.
5. Gather local solutions if you need a global vector on rank 0.

Example:

```python
from mpi4py import MPI
import jaxamg
from jaxamg.matrices import poisson_matrix_distributed, rhs_ones
from jaxamg.mpi_utils import partition_vector, gather_solution

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

# Build local matrix rows for each rank
n = 16
A_local, row_start, row_end = poisson_matrix_distributed(n, n, rank, nranks)
b_local, _, _ = partition_vector(rhs_ones(n * n), rank, nranks)

config = {
    "solver": "CG",
    "preconditioner": {"solver": "JACOBI_L1"},
    "communicator": "MPI_DIRECT",
}

# Solve
x_local, info = jaxamg.solve(
    A_local,
    b_local,
    comm=comm,
    nglobal=n * n,
    partition_info=(row_start, row_end),
    config=config,
)

# Gather solution
x_global = gather_solution(x_local, comm, root=0)
if rank == 0: print(f"Soluton: {x_global}")

# Finalization
comm.Barrier()
jaxamg.finalize()
```

## Caching

If you are solving many similar systems repeatedly, such as in an optimization loop, especially with JIT, compute and cache the metadata once (outside the JIT-compiled region) and reuse it for subsequent solves:

```python
# Compute once (outside JIT)
mpi_cache = jaxamg.cache_mpi_metadata(
    solver_config, comm, n_global, partition_info, A_local
)

# Use inside JIT if needed
A_cached = jaxamg.with_cache(A_local, mpi=mpi_cache, is_symmetric=True)
x, info = jaxamg.solve(A_cached, b_local)
```


In addition, you can tune the native resource cache behavior (which is separate from the metadata caching above) using:

```bash
export JAXAMG_CACHE_SIZE=2
```

Set `0` to disable resource caching, or use larger values to keep more solver resources alive between calls.

See the [Caching Guide](caching.md) for more details.



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

```python hl_lines="2"
config = {
    "communicator": "MPI_DIRECT",
    ...
}
```

Non-GPU-aware MPI still works, but communication may stage through host memory and be slower.
