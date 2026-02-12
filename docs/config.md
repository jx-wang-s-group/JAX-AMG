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
        "smoother": {"solver": "JACOBI_L1", "relaxation_factor": 0.8},
        "presweeps": 1,
        "postsweeps": 1,
        "coarse_solver": "NOSOLVER",
        "max_levels": 50,
        "cycle": "V",
    },
    "tolerance": 1e-6,
    "max_iters": 1000,
    "print_solve_stats": 1,
    "norm": "L2",
}
```

JAX-AMG also enables residual tracking internally (`monitor_residual=1`, `store_res_history=1`) so `info["residual"]` is available.

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

- **Flat config** (simple dict with top-level keys like `solver`, `tolerance`), for example:
```python
config = {
    "solver": "CG",
    "preconditioner": {"solver": "JACOBI_L1"},
    "tolerance": 1e-8,
    "max_iters": 200,
}
```
- **Nested config** (`config_version: 2` with `solver: {...}` scope), for example:
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

For nested configs, JAX-AMG forwards your structure directly to AmgX and does not inject flat defaults into nested scopes.

### MPI config

## Common configuration keys

These are commonly used and work in JAX-AMG:

| Key | Type | Meaning |
|---|---|---|
| `solver` | `str` | Main solver type, e.g. `CG`, `PBICGSTAB`, `FGMRES`, `AMG`. |
| `preconditioner` | `str` or `dict` | Optional preconditioner solver (simple or nested). |
| `tolerance` | `float` | Convergence tolerance. |
| `max_iters` | `int` | Max solver iterations. |
| `norm` | `str` | Residual norm type, commonly `L2`. |
| `monitor_residual` | `0/1` | Enable residual monitoring. |
| `store_res_history` | `0/1` | Store residual history. |
| `print_solve_stats` | `0/1` | Print per-solve stats from AmgX. |
| `obtain_timings` | `0/1` | Collect setup/solve timings in AmgX. |
| `communicator` | `str` | MPI communication mode (`MPI` or `MPI_DIRECT`). |

AMG-specific keys (inside AMG solver/preconditioner scopes):

| Key | Meaning |
|---|---|
| `algorithm` | AMG algorithm family (for example `AGGREGATION`). |
| `smoother` | Smoother solver, e.g. `JACOBI_L1`, `MULTICOLOR_DILU`. |
| `presweeps`, `postsweeps` | Number of smoothing sweeps. |
| `coarse_solver` | Coarse-level solver, e.g. `DENSE_LU_SOLVER`, `NOSOLVER`. |
| `max_levels` | Maximum multigrid levels. |
| `cycle` | Cycle type (`V`, `W`, `F`, `CG`, `CGF`). |
| `selector` | Aggregation selector (e.g. `SIZE_2`, `SIZE_4`, `SIZE_8`, `MULTI_PAIRWISE`). |
| `interpolator` | Interpolation strategy (commonly `D2` in sample configs). |


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
| `MULTICOLOR_DILU` | Parallelized diagonal ILU varian with graph coloring. |
| `CHEBYSHEV` | Chebyshev iterative method. |
| `CHEBYSHEV_POLY` | Polynomial Chebyshev method. |
| `CF_JACOBI` | C/F-point Jacobi variant for AMG. |
| `DENSE_LU_SOLVER` | Dense LU direct solve. |
| `NOSOLVER` | No solver/preconditioner. |