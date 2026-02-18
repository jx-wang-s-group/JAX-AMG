# Caching Guide

`jaxamg` has multiple caching layers with different goals. This page focuses on the two caches users typically configure in scripts.

## Overview

1. **Metadata cache (Python, `jaxamg/cache.py`)**
    - Caches metadata, including sparsity/coloring info for operators and MPI-related data.
    - Main goal: avoid recomputing pre-processing work and make JIT usage easier.

2.  **AmgX resource cache (C++, `_amgx_*`)**
    - Controlled by `JAXAMG_CACHE_SIZE`.
    - Caches native AmgX handles (matrix/vector/solver/resources).
    - Main goal: avoid repeated native setup and improve solve throughput.

There is also an internal primitive cache in `jaxamg.py` used automatically by
the library. It usually does not require user tuning, so it is not a focus here.

## Metadata cache

### `with_cache(A, ...)`
- Main entry point for metadata caching: attach optional metadata to `A` once,
  then reuse `A` across repeated solves.
- A primary goal is to make JIT workflows easier by keeping static/precomputed
  metadata outside traced solve code.
- This is object-level metadata attachment, not native AmgX-handle caching.

When to use each option:

- `coloring=...`
    - For callable operators, this avoids recomputing sparsity and coloring on every
      solve.
    - It is especially helpful in iterative loops where the operator structure stays
      the same while values change.
    - In practice, pass the result of `cache_coloring(...)` into `with_cache(...)`.

- `mpi=...`
    - This reuses MPI metadata such as counts, displacements, communicator pointer,
      config string, and max nnz.
    - Use it when you run repeated MPI solves with the same communicator and
      partition layout.
    - In practice, pass the result of `cache_mpi_metadata(...)` into `with_cache(...)`.

- `is_symmetric=True`
    - This allows the backward pass to skip transpose-related work for symmetric systems.
    - Set it only when the matrix is truly symmetric and remains symmetric.
    - You can set it directly in `with_cache(...)`.

## Native AmgX resource cache

Set with environment variable:

```bash
export JAXAMG_CACHE_SIZE=2 # Defualt is 1
```

Behavior (two modes):

- `0`: isolated mode (no resource caching)
    - Create/destroy native resources every call.
    - Best for debugging behavior and cache isolation.
- Positive values (default: `1`): cache-enabled mode
    - Reuses native resources through an LRU cache for improved performance.
    - Larger values enable multi-entry reuse when alternating among multiple matrix
    structures/configs, including cases where the forward pass uses `A` and the
    gradient/backward pass uses a structurally different `A^T`.

### Solver setup reuse

When the cache hits (same sparsity structure and config as a previous solve), only the
matrix values are updated via `AMGX_matrix_replace_coefficients`. The internal solver
state built by `AMGX_solver_setup` during the first solve is **reused as-is** — setup
is not called again.

This is a common approach when the sparsity pattern stays fixed but coefficient values
change across repeated solves. The wall-time savings from skipping the expensive setup
phase often outweigh the small increase in iterations that may result from a slightly
stale internal state.


## Cache inspection

Use `jaxamg.get_solver_cache_info()` for inspecting current solver cache state, which includes:


- Current `size`/`capacity` for both native caches (`single_gpu`, `mpi`)
- Per-entry summaries (dimensions, mode, config, hashes)
- `isolated_mode` flag

## Clearing caches and cleanup

```python
import jaxamg

jaxamg.clear_solver_cache()  # Clears C++ AmgX handle cache
jaxamg.finalize()            # Clears caches/resources and tears down native state
```

### When to call `clear_solver_cache()`

Because solver setup state is reused on cache hit, force a rebuild when reuse quality
starts to degrade (for example, iteration counts grow noticeably or residual reduction
stalls).

For workloads with frequent large coefficient updates (including many optimization
workflows), a simple heuristic is to rebuild every `N` steps:

```python
for step in range(num_steps):
    if step % 10 == 0:
        jaxamg.clear_solver_cache()
    x, info = jaxamg.solve(A, b)
```

In many workloads with fixed sparsity and gradually changing coefficients, clearing the
cache is not necessary and setup reuse remains effective throughout the run.

### Notes

- `clear_solver_cache()` targets native C++ AmgX resources.
- Metadata attached via `with_cache(...)` remains on Python objects until those objects are replaced or discarded.
- For MPI mode, explicit `finalize()` during teardown can help avoid shutdown-time resource warnings.
