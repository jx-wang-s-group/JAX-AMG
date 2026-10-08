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

Before JIT, call `cache_coloring(..., dtype=...)` for each solve precision.
Recompute the colouring if new nonzero entries appear.

When to use each option:

- `coloring=...`
    - For callable operators, this avoids recomputing sparsity and coloring on every
      solve.
    - It is especially helpful in iterative loops where the operator structure stays
      the same while values change.
    - In practice, pass the result of `cache_coloring(...)` into `with_cache(...)`.
    - Under the hood, `cache_coloring(...)` detects the operator's sparsity pattern
      by **tracing** its jaxpr — propagating an index-set structure through each
      primitive to recover the exact pattern in a single trace — and falls back to
      exhaustive **probing** with basis vectors for operators it cannot trace
      (opaque calls, data-dependent indexing). The tracing method follows
      [Hill & Dalle (2025)](https://arxiv.org/abs/2501.17737); their Julia package
      is [SparseConnectivityTracer.jl](https://github.com/adrhill/SparseConnectivityTracer.jl).

- `pattern=...`
    - Declare all possible couplings with `jaxamg.pattern(rows, cols, shape)`,
      including entries that are zero initially but may appear later.
    - Pass the same declaration to new callable instances as parameters change.
      It covers all precisions used with those operators.
    - The declaration copies its inputs into immutable arrays.
      `pattern=` and `coloring=` are mutually exclusive. Attaching new colouring
      replaces the old declaration and any discovered colourings.

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
    - It also lets `transpose_nullspace` default to `nullspace`.

- `nullspace=...` / `transpose_nullspace=...`
    - Defaults for the arguments of the same name in `jaxamg.solve(...)`; see
      [Singular systems](examples.md#singular-systems).
    - In MPI mode also prepare the cached config with `cache_mpi_metadata(..., singular=True)`.

### Reusing a declared pattern

Create a pattern once outside JIT and reuse it as coefficients change:

```python
p = jaxamg.pattern(rows, cols, shape)

@jax.jit
def solve_at(coefficients, b):
    op = jaxamg.with_cache(lambda x: apply_operator(coefficients, x), pattern=p)
    return jaxamg.solve(op, b)[0]
```

Single-process solves reuse the pattern's layout while recomputing matrix values.
Reuse the same `Pattern` object; a new equivalent pattern has its own cache.

If your transpose callback materializes an operator, reuse a separate pattern for
it. MPI, sharded solves, and colouring tuples do not use this layout cache.

## Native AmgX resource cache

Set with environment variable:

```bash
export JAXAMG_CACHE_SIZE=2 # Default is 1
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

MPI cache capacity applies per communicator. Call `clear_solver_cache()` on
every rank together; inconsistent caches cause the solve to fail on all ranks.
Shared MPI resources are separate for each communicator, device, and transport
mode; `finalize()` releases them.

### Solver setup reuse

When the cache hits (same sparsity structure and config as a previous solve), the
matrix values are updated via `AMGX_matrix_replace_coefficients` and the solver setup
is repeated against the new values via `AMGX_solver_resetup`. `resetup` reuses the
cached AmgX solver/matrix objects, device allocations, and fine-level matrix
coloring established during the first solve, avoiding the resource-creation and
matrix-upload overhead of a cold start. The AMG hierarchy itself is rebuilt against
the new values by default; deeper reuse can be enabled via the
`structure_reuse_levels` AmgX config parameter.

This setup-reuse path is substantially cheaper than a cold start while remaining
correct for any change in coefficient values, making cache reuse safe for workloads
where coefficients change between solves (including optimization and time-stepping).

For repeated solves where reusing the existing AMG hierarchy is acceptable, pass
`reuse_setup=True` to `jaxamg.solve(...)`. This still refreshes the fine-level
matrix values, but skips `AMGX_solver_resetup` on cache hits. It can reduce the
per-solve cost further, but the reused hierarchy may be less effective for the
new values; if coefficients change enough, convergence may require more
iterations or become less robust. If that happens, call
`jaxamg.clear_solver_cache()` to force the next solve to build a fresh hierarchy.

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

In typical workloads — including optimization with changing coefficients — calling
`clear_solver_cache()` is **not** required. Cache hits automatically refresh the
solver against current values via `AMGX_solver_resetup`, so correctness is maintained
without any user intervention.

Reasons you might still want to call it explicitly:

- Free GPU memory between unrelated solve series (e.g. before moving on to a problem
  with a different shape or configuration).
- Force a fresh `AMGX_solver_setup` if `structure_reuse_levels > 0` is set in the
  AmgX config and the reused coarsening becomes a poor fit for the new values
  (not relevant with the default config, where `resetup` already rebuilds the
  hierarchy).
- Debugging or reproducing first-solve behavior.

Sparsity-pattern changes do **not** require an explicit clear: the cache key
includes a structural hash, so a different sparsity pattern produces a cache miss
and triggers a full setup automatically.

### Notes

- `clear_solver_cache()` targets native C++ AmgX resources.
- Metadata attached via `with_cache(...)` remains on Python objects until those objects are replaced or discarded.
- For MPI mode, explicit `finalize()` during teardown can help avoid shutdown-time resource warnings.
