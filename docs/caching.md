# Caching Guide

`jaxamg` has multiple caching layers with different goals. This page focuses on the two caches users typically configure in scripts.

## Overview

1. **Metadata cache (Python, `jaxamg/cache.py`)**
   - Caches metadata, including sparsity/coloring info for operators and MPI-related data.
   - Main goal: avoid recomputing pre-processing work and make JIT usage easier.

2. **AmgX resource cache (C++, `_amgx_*`)**
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
- `>0` (default: `1`): cache-enabled mode
  - Reuses native resources through an LRU cache for improved performance.
  - `>1` enables multi-entry reuse when alternating among multiple matrix
    structures/configs, including cases where the forward pass uses `A` and the
    gradient/backward pass uses a structurally different `A^T`.


## Clearing caches and cleanup

```python
import jaxamg

jaxamg.clear_solver_cache()  # Clears C++ AmgX handle cache
jaxamg.finalize()            # Clears caches/resources and tears down native state
```

Notes:
- `clear_solver_cache()` targets native C++ AmgX resources.
- Metadata attached via `with_cache(...)` remains on Python objects until those objects are replaced or discarded.
- In MPI tests/demos, explicit `finalize()` during teardown can help avoid shutdown-time resource warnings.
