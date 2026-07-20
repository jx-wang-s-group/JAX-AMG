# Solver Configuration

This page documents how to configure `jaxamg.solve(...)`, including solver/preconditioner choices and common AmgX options.

## Configuration input modes

`jaxamg.solve(...)` accepts configuration in three ways:

1. `config=<dict>`: a full configuration dictionary.
2. keyword arguments (`solver=...`, `max_iters=...`, etc.).
3. both together, where keyword arguments override values in `config`.

Example:

```python
x, info = jaxamg.solve(
    A,
    b,
    config={"solver": "CG", "max_iters": 100},
    max_iters=50,  # overrides config["max_iters"]
)
```

## Default config

If you do not provide a config, JAX-AMG uses:

```python
{
    "solver": "PBICGSTAB",
    "preconditioner": {
        "solver": "AMG",
        "algorithm": "CLASSICAL",
        "selector": "PMIS",
        "interpolator": "D2",
        "smoother": {
            "solver": "BLOCK_JACOBI",
            "relaxation_factor": 0.9,
        },
        "presweeps": 1,
        "postsweeps": 1,
        "max_levels": 100,
        "strength_threshold": 0.5,
        "dense_lu_num_rows": 1,
        "aggressive_levels": 0,
        "coarse_solver": "DENSE_LU_SOLVER",
        "max_iters": 1,
        "cycle": "V",
    },
    "convergence": "RELATIVE_INI",
    "tolerance": 1e-6,
    "max_iters": 1000,
    "norm": "L2",
    "exact_coarse_solve": 1,
}
```

User-supplied keys are merged into these defaults, so you only need
to specify what you want to change.  For example,
`config={"preconditioner": {"solver": "AMG"}}` inherits all the Classical AMG
settings above.

!!! note "Preconditioner default differs"

    The default above is for `jaxamg.solve(...)`, a full Krylov solve (`PBICGSTAB`) preconditioned by AMG. `jaxamg.make_preconditioner(...)` (and `make_lineax_preconditioner(...)`) instead default to a *single* AMG V-cycle (`solver="AMG"`, `max_iters=1`) used as an approximate inverse, since there the outer Krylov method owns the iteration. Override via their own `config`/`kwargs`.

JAX-AMG also enables residual tracking internally (`monitor_residual=1`,
`store_res_history=1`) so `info["residual"]` and `info["residual_history"]`
are always available. The history is the outer solver's convergence curve:
entry `i` is the residual norm after outer iteration `i`, with entry 0 the
initial residual. Outside `jit` it is trimmed to `iterations + 1` entries;
inside `jit` it has fixed length `max_iters + 1`, NaN-padded past entry
`iterations`.

### Block matrices

For coupled multi-component systems, pass `block_dim=k` to `jaxamg.solve(...)`
to treat the matrix as having square `k x k` blocks (unknowns interleaved
node-major: row `i*k + c` is component `c` of node `i`). `A` and `b` keep
their ordinary scalar CSR/vector form — the conversion to AmgX's BSR format
happens internally — and autodiff works unchanged. Rows must be divisible by
`block_dim` (each rank's local partition in MPI mode).

Because AmgX's classical AMG does not support block matrices, the AMG defaults
switch to aggregation AMG when `block_dim > 1`:

```python
{
    "solver": "AMG",
    "algorithm": "AGGREGATION",
    "selector": "SIZE_2",
    "smoother": {"solver": "BLOCK_JACOBI", "relaxation_factor": 0.9},
    "presweeps": 1,
    "postsweeps": 1,
    "max_levels": 100,
    "min_coarse_rows": 32,
    "dense_lu_num_rows": 64,
    "coarse_solver": "DENSE_LU_SOLVER",
    "max_iters": 1,
    "cycle": "V",
}
```

Explicitly configuring CLASSICAL AMG together with `block_dim > 1` raises a
`ValueError`. Block-aware preconditioning is often the entire benefit: on a
strongly coupled system, `BLOCK_JACOBI` under `block_dim=2` inverts true
2x2 diagonal blocks (instead of scalar diagonal entries) and can converge in
a handful of iterations where the scalar-preconditioned solve stalls.

### MPI config

The `communicator` key can be set to `MPI` for standard CPU-based MPI or `MPI_DIRECT` for GPU-aware MPI. The default is `MPI`. To use GPU-aware MPI, ensure that your MPI installation supports it. For more details, see the [MPI Guide](mpi.md).

```python hl_lines="4"
config = {
    "solver": "PBICGSTAB",
    "preconditioner": {"solver": "MULTICOLOR_DILU"},
    "communicator": "MPI_DIRECT",
    "max_iters": 200,
    "tolerance": 1e-8,
}
```

## Flat vs nested config

JAX-AMG supports both flat (simple) and nested config dict:

- **Flat config** (simple dict with top-level keys like `solver`, `tolerance`).
```python
config = {
    "solver": "CG",
    "preconditioner": {"solver": "JACOBI_L1"},
    "tolerance": 1e-8,
    "max_iters": 200,
}
```

- **Nested config** (`config_version: 2` with `solver: {...}` scope).
```python
config = {
    "config_version": 2,
    "solver": {
        "solver": "PCG",
        "preconditioner": {
            "solver": "AMG",
            "smoother": "JACOBI_L1",
        },
        "tolerance": 1e-6,
        "max_iters": 200,
    },
}
```

## Common configuration keys

These are commonly used and work in JAX-AMG:

| Key | Type | Meaning |
|---|---|---|
| `solver` | `str` | Main solver type, e.g. `CG`, `PBICGSTAB`, `FGMRES`, `AMG`. |
| `preconditioner` | `str` or `dict` | Optional preconditioner solver (simple or nested). |
| `tolerance` | `float` | Convergence tolerance. |
| `max_iters` | `int` | Max solver iterations. |
| `convergence` | `str` | Convergence criterion (see below). |
| `norm` | `str` | Residual norm type, commonly `L2`. |
| `monitor_residual` | `0` or `1` | Enable residual monitoring. |
| `store_res_history` | `0` or `1` | Store residual history. |
| `print_solve_stats` | `0` or `1` | Print per-solve stats from AmgX. |
| `obtain_timings` | `0` or `1` | Collect setup/solve timings in AmgX. |
| `communicator` | `str` | MPI communication mode (`MPI` or `MPI_DIRECT`). |

### Convergence criteria

| Value | Meaning |
|---|---|
| `ABSOLUTE` | Absolute residual threshold check. |
| `RELATIVE_INI_CORE` | Core variant of `RELATIVE_INI` (same criterion family, different internal residual handling). |
| `RELATIVE_MAX_CORE` | Core variant of `RELATIVE_MAX` (same criterion family, different internal residual handling). |
| `RELATIVE_INI` | True relative residual ‖r_k‖ / ‖r_0‖. Computes the explicit residual at every iteration (default). |
| `RELATIVE_MAX` | Max-norm relative residual. |
| `COMBINED_REL_INI_ABS` | Combined relative-initial and absolute criterion. |

### AMG-specific keys

These go inside the AMG solver/preconditioner scope:

| Key | Meaning |
|---|---|
| `algorithm` | AMG algorithm: `CLASSICAL` (default) or `AGGREGATION`. |
| `selector` | Coarsening selector. Classical: `PMIS` (default), `HMIS`. Aggregation: `SIZE_2`, `SIZE_4`, `SIZE_8`, `MULTI_PAIRWISE`. |
| `interpolator` | Interpolation strategy, e.g. `D2` (default for Classical). |
| `smoother` | Smoother solver, e.g. `BLOCK_JACOBI` (default), `JACOBI_L1`, `MULTICOLOR_GS`, `MULTICOLOR_DILU`. |
| `presweeps`, `postsweeps` | Number of smoothing sweeps (default: 1). |
| `coarse_solver` | Coarse-level solver: `DENSE_LU_SOLVER` (default) or `NOSOLVER`. |
| `strength_threshold` | Strength of connection threshold for coarsening (default: 0.5). |
| `max_levels` | Maximum multigrid levels (default: 100). |
| `cycle` | Cycle type: `V` (default), `W`, `F`, `CG`, `CGF`. |


## Supported Solvers and Preconditioners

AmgX supports a broader set of solvers, preconditioners, and smoothers:

| Value | Method |
|---|---|
| `AMG` | Algebraic multigrid (AMG) method. |
| `CG` | Conjugate gradient (CG) method for symmetric positive-definite systems. |
| `PCG` | Preconditioned CG method. |
| `PCGF` | Flexible PCG that supports changing/variable preconditioners across iterations. |
| `BICGSTAB` | Bi-conjugate gradient stabilized (BiCGStab) method for general nonsymmetric systems. |
| `PBICGSTAB` | Preconditioned BiCGSTAB method. |
| `GMRES` | Generalized minimal residual (GMRES) method for nonsymmetric systems. |
| `FGMRES` | Flexible GMRES that supports changing/variable preconditioners across iterations. |
| `IDR` | Induced dimension reduction (IDR) method for nonsymmetric systems. |
| `IDRMSYNC` | A variant of IDR designed for parallel performance optimization. |
| `JACOBI_L1` | L1-scaled Jacobi smoother/preconditioner. |
| `BLOCK_JACOBI` | Block Jacobi method. |
| `GS` | Gauss-Seidel preconditioner/smoother. |
| `MULTICOLOR_GS` | Parallelized Gauss-Seidel using graph coloring. |
| `MULTICOLOR_ILU` | Parallelized incomplete LU factorization (ILU) with graph coloring. |
| `MULTICOLOR_DILU` | Parallelized diagonal ILU variant with graph coloring. |
| `CHEBYSHEV` | Chebyshev iterative method. |
| `CHEBYSHEV_POLY` | Polynomial Chebyshev method. |
| `CF_JACOBI` | C/F-point Jacobi variant for AMG. |
| `DENSE_LU_SOLVER` | Dense LU direct solve. |
| `NOSOLVER` | No solver/preconditioner. |
