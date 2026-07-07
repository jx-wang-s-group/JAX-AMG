# Examples

This page collects common JAX-AMG usage patterns. For complete runnable scripts, see the `demo/` directory.

## Single-GPU mode

### Solving a matrix system

The input matrix can be a sparse matrix (from JAX or SciPy) of various types, or a dense matrix. Internally, all formats are converted to a `jax.experimental.sparse.BCSR` sparse matrix.

=== "Python"

    ```python
    import jaxamg
    from jaxamg.matrices import rhs_ones, tridiagonal_matrix

    n = 100
    A = tridiagonal_matrix(n, diagonal_value=4.0)
    b = rhs_ones(100)

    x, info = jaxamg.solve(A, b, solver="CG")
    print("Solution:", x)
    print("Iterations:", info["iterations"])
    print("Residual:", info["residual"])
    ```

=== "Result"

    ```text
    Solution: [0.36602542 0.46410164 0.490381   0.4974226  0.49930936 0.49981493
    0.4999504  0.49998674 0.49999642 0.49999908 0.49999976 0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.5
    0.5        0.5        0.5        0.5        0.5        0.49999976
    0.49999908 0.49999642 0.49998674 0.4999504  0.49981493 0.49930936
    0.4974226  0.490381   0.46410164 0.36602542]
    Iterations: 12
    Residual: 3.281010094724479e-07
    ```

### Solving with an operator

You can also solve using a callable operator instead of an explicit matrix.

!!! note

    AmgX still needs an explicit matrix, so the operator is materialized internally — its sparsity pattern is detected automatically (by tracing the operator's jaxpr, with basis-vector probing as a fallback) and the values are assembled via graph-colored probing. This is cached, so it happens once. See [Caching](caching.md) for details.

=== "Python"

    ```python
    import jaxamg
    from jaxamg.matrices import rhs_ones, poisson_operator

    n = 16
    b = rhs_ones(n)

    op = poisson_operator()
    x, info = jaxamg.solve(op, b, solver="CG")
    print("Solution:", x)
    print("Iterations:", info["iterations"])
    print("Residual:", info["residual"])
    ```

=== "Result"

    ```text
    Solution: [0.8333334 1.1666667 1.1666667 0.8333334 1.1666667 1.6666667 1.6666667
    1.1666667 1.1666667 1.6666667 1.6666667 1.1666667 0.8333334 1.1666667
    1.1666667 0.8333334]
    Iterations: 3
    Residual: 4.600156700007574e-08
    ```



### Custom solver configuration

You can supply a custom solver configuration:

=== "Python"

    ```python
    import jaxamg
    from jaxamg.matrices import poisson_matrix, rhs_ones

    n = 4
    A = poisson_matrix(n, skew=0.5)
    b = rhs_ones(n * n)

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

    print("Solution:", x)
    print("Iterations:", info["iterations"])
    print("Residual:", info["residual"])
    ```

=== "Result"

    ```text
    Solution: [0.5736315  0.8630174  0.95168453 0.7778139  0.8630174  1.361689
    1.5261416  1.2288666  0.95168453 1.5261416  1.7215995  1.3806959
    0.7778139  1.2288666  1.3806959  1.112935  ]
    Iterations: 1
    Residual: 1.2169988968132022e-14
    ```

See [Solver Configuration](config.md) for full details.

### Using JAX-AMG as a preconditioner for native JAX solvers

You can also use JAX-AMG only for the preconditioner application, while a native JAX Krylov method owns the outer iterations.

!!! note

    `make_preconditioner` (and `make_lineax_preconditioner`, shown below) apply a single AMG V-cycle as an approximate inverse (not a full solve); the outer Krylov method (here `cg`) owns the iterations. See [Default config](config.md#default-config) for how this differs from `jaxamg.solve`.

=== "Python"

    ```python
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg

    import jaxamg
    from jaxamg.matrices import poisson_matrix, rhs_ones

    n = 32
    A = poisson_matrix(n)
    b = rhs_ones(n * n)
    M = jaxamg.make_preconditioner(A)

    x, _ = cg(A, b, M=M, tol=1e-6)

    residual = jnp.linalg.norm(b - A @ x) / jnp.linalg.norm(b)

    print(f"Solution: {x}")
    print(f"Residual: {residual:.3e}")
    ```

=== "Result"

    ```text
    Solution: [2.0437262 3.587453  4.8288155 ... 4.8288164 3.5874531 2.0437262]
    Residual: 1.867e-05
    ```

### Using JAX-AMG as a preconditioner for Lineax

If you use [Lineax](https://docs.kidger.site/lineax/), `jaxamg.make_lineax_preconditioner(...)` wraps a `lineax.AbstractLinearOperator` as an AMG preconditioner operator, ready to pass to `options={"preconditioner": ...}`.

=== "Python"

    ```python
    import jax
    import jax.numpy as jnp
    import lineax as lx

    import jaxamg
    from jaxamg.matrices import poisson_matrix, rhs_ones

    jax.config.update("jax_enable_x64", True)

    n = 32
    A = poisson_matrix(n, skew=2.0)
    b = rhs_ones(n * n)

    operator = lx.FunctionLinearOperator(lambda x: A @ x, b)
    preconditioner = jaxamg.make_lineax_preconditioner(operator)

    solution = lx.linear_solve(
        operator,
        b,
        solver=lx.BiCGStab(rtol=1e-6, atol=1e-6, max_steps=100),
        options={"preconditioner": preconditioner},
    )
    residual = jnp.linalg.norm(b - A @ solution.value) / jnp.linalg.norm(b)

    print(f"Solution: {solution.value}")
    print(f"Residual: {residual:.3e}")
    ```

=== "Result"

    ```text
    Solution: [ 0.25        0.375       0.4375     ... 13.88522106 14.16045232 14.41045187]
    Residual: 2.636e-07
    ```

### Optimization via auto differentiation

=== "Python"

    ```python
    import jax
    import jax.numpy as jnp
    import jaxamg
    from jaxamg.matrices import rhs_ones, rhs_linear, tridiagonal_matrix

    n = 64
    A = tridiagonal_matrix(n, diagonal_value=4.0)
    b_init = rhs_ones(n)
    x_target = rhs_linear(n)

    def loss_fn(b_vec):
        x, _ = jaxamg.solve(A, b_vec, solver="CG")
        return jnp.sum((x-x_target)**2)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    lr = 1.0
    eps = 0.001
    max_iters = 100

    b = b_init
    for _ in range(max_iters):
        loss, grad = grad_fn(b)
        b = b - lr * grad
        print(f"Iter {_:4d}: loss = {loss:8.4f}, grad_norm = {jnp.linalg.norm(grad):8.4f}")
        if jnp.linalg.norm(grad) < eps:
            print("Converged!")
            break
    ```

=== "Result"

    ```text
    Iter    0: loss =   5.5413, grad_norm =   2.2774
    Iter    1: loss =   1.5938, grad_norm =   1.1939
    Iter    2: loss =   0.5020, grad_norm =   0.6397
    Iter    3: loss =   0.1848, grad_norm =   0.3566
    Iter    4: loss =   0.0841, grad_norm =   0.2125
    Iter    5: loss =   0.0471, grad_norm =   0.1390
    Iter    6: loss =   0.0307, grad_norm =   0.1002
    Iter    7: loss =   0.0218, grad_norm =   0.0781
    Iter    8: loss =   0.0163, grad_norm =   0.0640
    Iter    9: loss =   0.0125, grad_norm =   0.0540
    Iter   10: loss =   0.0098, grad_norm =   0.0465
    Iter   11: loss =   0.0078, grad_norm =   0.0405
    Iter   12: loss =   0.0062, grad_norm =   0.0356
    Iter   13: loss =   0.0051, grad_norm =   0.0315
    Iter   14: loss =   0.0041, grad_norm =   0.0280
    Iter   15: loss =   0.0034, grad_norm =   0.0250
    Iter   16: loss =   0.0028, grad_norm =   0.0225
    Iter   17: loss =   0.0023, grad_norm =   0.0202
    Iter   18: loss =   0.0019, grad_norm =   0.0183
    Iter   19: loss =   0.0016, grad_norm =   0.0165
    Iter   20: loss =   0.0013, grad_norm =   0.0150
    Iter   21: loss =   0.0011, grad_norm =   0.0136
    Iter   22: loss =   0.0009, grad_norm =   0.0124
    Iter   23: loss =   0.0008, grad_norm =   0.0113
    Iter   24: loss =   0.0007, grad_norm =   0.0104
    Iter   25: loss =   0.0006, grad_norm =   0.0095
    Iter   26: loss =   0.0005, grad_norm =   0.0087
    Iter   27: loss =   0.0004, grad_norm =   0.0080
    Iter   28: loss =   0.0003, grad_norm =   0.0073
    Iter   29: loss =   0.0003, grad_norm =   0.0067
    Iter   30: loss =   0.0003, grad_norm =   0.0062
    Iter   31: loss =   0.0002, grad_norm =   0.0057
    Iter   32: loss =   0.0002, grad_norm =   0.0053
    Iter   33: loss =   0.0002, grad_norm =   0.0049
    Iter   34: loss =   0.0001, grad_norm =   0.0045
    Iter   35: loss =   0.0001, grad_norm =   0.0041
    Iter   36: loss =   0.0001, grad_norm =   0.0038
    Iter   37: loss =   0.0001, grad_norm =   0.0035
    Iter   38: loss =   0.0001, grad_norm =   0.0033
    Iter   39: loss =   0.0001, grad_norm =   0.0030
    Iter   40: loss =   0.0001, grad_norm =   0.0028
    Iter   41: loss =   0.0000, grad_norm =   0.0026
    Iter   42: loss =   0.0000, grad_norm =   0.0024
    Iter   43: loss =   0.0000, grad_norm =   0.0022
    Iter   44: loss =   0.0000, grad_norm =   0.0021
    Iter   45: loss =   0.0000, grad_norm =   0.0019
    Iter   46: loss =   0.0000, grad_norm =   0.0018
    Iter   47: loss =   0.0000, grad_norm =   0.0017
    Iter   48: loss =   0.0000, grad_norm =   0.0015
    Iter   49: loss =   0.0000, grad_norm =   0.0014
    Iter   50: loss =   0.0000, grad_norm =   0.0013
    Iter   51: loss =   0.0000, grad_norm =   0.0012
    Iter   52: loss =   0.0000, grad_norm =   0.0012
    Iter   53: loss =   0.0000, grad_norm =   0.0011
    Iter   54: loss =   0.0000, grad_norm =   0.0010
    Iter   55: loss =   0.0000, grad_norm =   0.0009
    Converged!
    ```

### Optimization with color caching for operator

For parameterized operators, compute coloring once and reuse it during optimization.

=== "Python"

    ```python
    import jax
    import jax.numpy as jnp
    import jaxamg
    from jaxamg.matrices import rhs_ones, tridiagonal_operator

    n = 64
    diag_true = 4.0
    diag_init = 8.0

    b = rhs_ones(n)
    x_target, _ = jaxamg.solve(tridiagonal_operator(diag_true), b, solver="CG")

    coloring = jaxamg.cache_coloring(tridiagonal_operator(diag_init), shape=n)

    def loss(diag):
        A = jaxamg.with_cache(tridiagonal_operator(diag), coloring=coloring, is_symmetric=True)
        x_pred, _ = jaxamg.solve(A, b, solver="CG")
        return jnp.mean((x_pred - x_target) ** 2)

    grad_fn = jax.jit(jax.value_and_grad(loss))

    lr = 2.0
    eps = 0.001
    max_iters = 100

    diag = diag_init
    for _ in range(max_iters):
        loss, grad = grad_fn(diag)
        diag = diag - lr * grad

        print(f"Iter {_:4d}: diag = {diag:8.4f}, loss = {loss:8.4f}, grad_norm = {jnp.linalg.norm(grad):8.4f}")
        if jnp.linalg.norm(grad) < eps:
            print("Converged!")
            break
    ```

=== "Result"

    ```text
    Iter    0: diag =   7.9637, loss =   0.1082, grad_norm =   0.0181
    Iter    1: diag =   7.9271, loss =   0.1076, grad_norm =   0.0183
    Iter    2: diag =   7.8902, loss =   0.1069, grad_norm =   0.0185
    Iter    3: diag =   7.8530, loss =   0.1062, grad_norm =   0.0186
    Iter    4: diag =   7.8153, loss =   0.1055, grad_norm =   0.0188
    Iter    5: diag =   7.7774, loss =   0.1048, grad_norm =   0.0190
    Iter    6: diag =   7.7390, loss =   0.1041, grad_norm =   0.0192
    Iter    7: diag =   7.7003, loss =   0.1033, grad_norm =   0.0194
    Iter    8: diag =   7.6612, loss =   0.1026, grad_norm =   0.0196
    Iter    9: diag =   7.6217, loss =   0.1018, grad_norm =   0.0197
    Iter   10: diag =   7.5818, loss =   0.1010, grad_norm =   0.0199
    Iter   11: diag =   7.5415, loss =   0.1002, grad_norm =   0.0202
    Iter   12: diag =   7.5008, loss =   0.0994, grad_norm =   0.0204
    Iter   13: diag =   7.4596, loss =   0.0986, grad_norm =   0.0206
    Iter   14: diag =   7.4180, loss =   0.0977, grad_norm =   0.0208
    Iter   15: diag =   7.3760, loss =   0.0969, grad_norm =   0.0210
    Iter   16: diag =   7.3335, loss =   0.0960, grad_norm =   0.0213
    Iter   17: diag =   7.2905, loss =   0.0951, grad_norm =   0.0215
    Iter   18: diag =   7.2470, loss =   0.0941, grad_norm =   0.0217
    Iter   19: diag =   7.2031, loss =   0.0932, grad_norm =   0.0220
    Iter   20: diag =   7.1586, loss =   0.0922, grad_norm =   0.0222
    Iter   21: diag =   7.1136, loss =   0.0912, grad_norm =   0.0225
    Iter   22: diag =   7.0681, loss =   0.0902, grad_norm =   0.0228
    Iter   23: diag =   7.0220, loss =   0.0892, grad_norm =   0.0230
    Iter   24: diag =   6.9754, loss =   0.0881, grad_norm =   0.0233
    Iter   25: diag =   6.9281, loss =   0.0870, grad_norm =   0.0236
    Iter   26: diag =   6.8803, loss =   0.0859, grad_norm =   0.0239
    Iter   27: diag =   6.8319, loss =   0.0847, grad_norm =   0.0242
    Iter   28: diag =   6.7828, loss =   0.0836, grad_norm =   0.0245
    Iter   29: diag =   6.7331, loss =   0.0823, grad_norm =   0.0249
    Iter   30: diag =   6.6828, loss =   0.0811, grad_norm =   0.0252
    Iter   31: diag =   6.6317, loss =   0.0798, grad_norm =   0.0255
    Iter   32: diag =   6.5800, loss =   0.0785, grad_norm =   0.0259
    Iter   33: diag =   6.5275, loss =   0.0772, grad_norm =   0.0262
    Iter   34: diag =   6.4743, loss =   0.0758, grad_norm =   0.0266
    Iter   35: diag =   6.4204, loss =   0.0744, grad_norm =   0.0270
    Iter   36: diag =   6.3657, loss =   0.0729, grad_norm =   0.0274
    Iter   37: diag =   6.3102, loss =   0.0714, grad_norm =   0.0278
    Iter   38: diag =   6.2538, loss =   0.0698, grad_norm =   0.0282
    Iter   39: diag =   6.1967, loss =   0.0682, grad_norm =   0.0286
    Iter   40: diag =   6.1387, loss =   0.0666, grad_norm =   0.0290
    Iter   41: diag =   6.0798, loss =   0.0649, grad_norm =   0.0294
    Iter   42: diag =   6.0201, loss =   0.0631, grad_norm =   0.0299
    Iter   43: diag =   5.9594, loss =   0.0613, grad_norm =   0.0303
    Iter   44: diag =   5.8979, loss =   0.0595, grad_norm =   0.0308
    Iter   45: diag =   5.8354, loss =   0.0576, grad_norm =   0.0312
    Iter   46: diag =   5.7720, loss =   0.0556, grad_norm =   0.0317
    Iter   47: diag =   5.7076, loss =   0.0536, grad_norm =   0.0322
    Iter   48: diag =   5.6423, loss =   0.0515, grad_norm =   0.0326
    Iter   49: diag =   5.5761, loss =   0.0494, grad_norm =   0.0331
    Iter   50: diag =   5.5090, loss =   0.0472, grad_norm =   0.0336
    Iter   51: diag =   5.4410, loss =   0.0449, grad_norm =   0.0340
    Iter   52: diag =   5.3721, loss =   0.0426, grad_norm =   0.0344
    Iter   53: diag =   5.3025, loss =   0.0402, grad_norm =   0.0348
    Iter   54: diag =   5.2321, loss =   0.0377, grad_norm =   0.0352
    Iter   55: diag =   5.1611, loss =   0.0352, grad_norm =   0.0355
    Iter   56: diag =   5.0896, loss =   0.0327, grad_norm =   0.0357
    Iter   57: diag =   5.0178, loss =   0.0302, grad_norm =   0.0359
    Iter   58: diag =   4.9458, loss =   0.0276, grad_norm =   0.0360
    Iter   59: diag =   4.8739, loss =   0.0250, grad_norm =   0.0359
    Iter   60: diag =   4.8024, loss =   0.0224, grad_norm =   0.0358
    Iter   61: diag =   4.7316, loss =   0.0199, grad_norm =   0.0354
    Iter   62: diag =   4.6620, loss =   0.0174, grad_norm =   0.0348
    Iter   63: diag =   4.5939, loss =   0.0150, grad_norm =   0.0340
    Iter   64: diag =   4.5279, loss =   0.0127, grad_norm =   0.0330
    Iter   65: diag =   4.4645, loss =   0.0106, grad_norm =   0.0317
    Iter   66: diag =   4.4044, loss =   0.0086, grad_norm =   0.0301
    Iter   67: diag =   4.3480, loss =   0.0068, grad_norm =   0.0282
    Iter   68: diag =   4.2960, loss =   0.0053, grad_norm =   0.0260
    Iter   69: diag =   4.2486, loss =   0.0040, grad_norm =   0.0237
    Iter   70: diag =   4.2063, loss =   0.0030, grad_norm =   0.0212
    Iter   71: diag =   4.1692, loss =   0.0021, grad_norm =   0.0186
    Iter   72: diag =   4.1371, loss =   0.0015, grad_norm =   0.0160
    Iter   73: diag =   4.1100, loss =   0.0010, grad_norm =   0.0136
    Iter   74: diag =   4.0873, loss =   0.0007, grad_norm =   0.0113
    Iter   75: diag =   4.0688, loss =   0.0004, grad_norm =   0.0093
    Iter   76: diag =   4.0538, loss =   0.0003, grad_norm =   0.0075
    Iter   77: diag =   4.0418, loss =   0.0002, grad_norm =   0.0060
    Iter   78: diag =   4.0323, loss =   0.0001, grad_norm =   0.0047
    Iter   79: diag =   4.0249, loss =   0.0001, grad_norm =   0.0037
    Iter   80: diag =   4.0191, loss =   0.0000, grad_norm =   0.0029
    Iter   81: diag =   4.0146, loss =   0.0000, grad_norm =   0.0022
    Iter   82: diag =   4.0112, loss =   0.0000, grad_norm =   0.0017
    Iter   83: diag =   4.0085, loss =   0.0000, grad_norm =   0.0013
    Iter   84: diag =   4.0065, loss =   0.0000, grad_norm =   0.0010
    Iter   85: diag =   4.0049, loss =   0.0000, grad_norm =   0.0008
    Converged!
    ```

### Batched solves with `vmap`

JAX-AMG natively supports batched solves using `jax.vmap`. This allows you to efficiently solve a system with multiple right-hand sides.

=== "Python"

    ```python
    import jax
    import jax.numpy as jnp
    import jaxamg
    from jaxamg.matrices import poisson_matrix, rhs_random

    grid_size = 32
    A = poisson_matrix(grid_size)
    n = grid_size**2

    batch_size = 5
    seeds = jnp.arange(batch_size)
    b_batched = jax.vmap(rhs_random, in_axes=(None, 0))(n, seeds)

    def solve_fn(matrix, rhs):
        return jaxamg.solve(matrix, rhs, solver="CG")

    vmap_solve = jax.vmap(solve_fn, in_axes=(None, 0))
    x_batched, infos = vmap_solve(A, b_batched)

    print(f"Batch Solution Shape: {x_batched.shape}")
    print(f"Batch Residuals: {infos['residual']}")
    ```

=== "Result"

    ```text
    Batch Solution Shape: (5, 1024)
    Batch Residuals: [2.8950488e-05 2.5211844e-05 2.8009201e-05 3.0575200e-05 2.9501451e-05]
    ```

## MPI distributed mode

Launch scripts with MPI:

```bash
mpirun -n <num_procs> python your_script.py
```

### Solving a distributed matrix system

=== "Python"

    ```python
    from mpi4py import MPI
    import jaxamg
    from jaxamg.mpi_utils import partition_vector, gather_vector
    from jaxamg.matrices import poisson_matrix_distributed, rhs_ones

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    n = 4
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

    x_global = gather_vector(x_local, comm, root=0)
    if rank == 0:
        print(f"Solution: {x_global}")
        print(f"Iterations: {info['iterations']}")
        print(f"Residual: {info['residual']}")

    comm.Barrier()
    jaxamg.finalize()
    ```

=== "Result"

    ```text
    Solution: [0.8333334 1.1666667 1.1666667 0.8333334 1.1666667 1.6666667 1.6666667
    1.1666667 1.1666667 1.6666667 1.6666667 1.1666667 0.8333334 1.1666667
    1.1666667 0.8333334]
    Iterations: 3
    Residual: 4.600157055278942e-08
    ```

### Distributed optimization

Each rank computes local loss/gradient, then uses MPI reductions to form global metrics.

=== "Python"

    ```python
    import jax
    import jax.numpy as jnp
    from mpi4py import MPI
    import jaxamg
    from jaxamg.matrices import tridiagonal_matrix_distributed, rhs_ones, rhs_linear
    from jaxamg.mpi_utils import gather_vector

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    n_global = 64
    x_target_global = rhs_linear(n_global)

    A_loc, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0
    )
    b_loc = rhs_ones(row_end - row_start)
    x_target_loc = x_target_global[row_start:row_end]

    config = {"solver": "CG", "communicator": "MPI_DIRECT"}

    mpi_cache = jaxamg.cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), A_loc
    )

    def loss_local(b_loc):
        A = jaxamg.with_cache(A_loc, mpi=mpi_cache, is_symmetric=True)
        x_loc, _ = jaxamg.solve(A, b_loc)
        return jnp.sum((x_loc - x_target_loc) ** 2)

    loss_grad = jax.jit(jax.value_and_grad(loss_local))

    lr = 0.5
    max_iters = 100
    eps = 0.01

    for _ in range(max_iters):
        loss_loc, grad_loc = loss_grad(b_loc)
        loss_global = comm.allreduce(loss_loc, op=MPI.SUM)
        b_loc = b_loc - lr * grad_loc
        if jnp.linalg.norm(grad_loc) < eps:
            print("Converged!")
            break
        if rank == 0:
            print(f"Iter {_:4d}: loss = {loss_global:8.4f}, grad_norm = {jnp.linalg.norm(grad_loc):8.4f}")

    comm.Barrier()
    x_loc, _ = jaxamg.solve(A_loc, b_loc)
    x_global = gather_vector(x_loc, comm, root=0)
    if rank == 0:
        print(f"Relative error: {jnp.linalg.norm(x_global - x_target_global) / jnp.linalg.norm(x_target_global)}")

    comm.Barrier()
    jaxamg.finalize()
    ```

=== "Result"

    ```text
    Iter    0: loss =   5.5413, grad_norm =   2.2774
    Iter    1: loss =   3.2578, grad_norm =   1.7341
    Iter    2: loss =   1.9327, grad_norm =   1.3232
    Iter    3: loss =   1.1603, grad_norm =   1.0124
    Iter    4: loss =   0.7075, grad_norm =   0.7773
    Iter    5: loss =   0.4401, grad_norm =   0.5995
    Iter    6: loss =   0.2807, grad_norm =   0.4651
    Iter    7: loss =   0.1844, grad_norm =   0.3635
    Iter    8: loss =   0.1254, grad_norm =   0.2868
    Iter    9: loss =   0.0885, grad_norm =   0.2288
    Iter   10: loss =   0.0648, grad_norm =   0.1851
    Iter   11: loss =   0.0493, grad_norm =   0.1521
    Iter   12: loss =   0.0387, grad_norm =   0.1271
    Iter   13: loss =   0.0312, grad_norm =   0.1080
    Iter   14: loss =   0.0258, grad_norm =   0.0934
    Iter   15: loss =   0.0217, grad_norm =   0.0820
    Iter   16: loss =   0.0185, grad_norm =   0.0730
    Iter   17: loss =   0.0160, grad_norm =   0.0657
    Iter   18: loss =   0.0139, grad_norm =   0.0597
    Iter   19: loss =   0.0122, grad_norm =   0.0547
    Iter   20: loss =   0.0108, grad_norm =   0.0504
    Iter   21: loss =   0.0096, grad_norm =   0.0467
    Iter   22: loss =   0.0085, grad_norm =   0.0434
    Iter   23: loss =   0.0076, grad_norm =   0.0405
    Iter   24: loss =   0.0068, grad_norm =   0.0379
    Iter   25: loss =   0.0061, grad_norm =   0.0356
    Iter   26: loss =   0.0055, grad_norm =   0.0334
    Iter   27: loss =   0.0050, grad_norm =   0.0315
    Iter   28: loss =   0.0045, grad_norm =   0.0297
    Iter   29: loss =   0.0041, grad_norm =   0.0280
    Iter   30: loss =   0.0037, grad_norm =   0.0265
    Iter   31: loss =   0.0033, grad_norm =   0.0250
    Iter   32: loss =   0.0030, grad_norm =   0.0237
    Iter   33: loss =   0.0028, grad_norm =   0.0225
    Iter   34: loss =   0.0025, grad_norm =   0.0213
    Iter   35: loss =   0.0023, grad_norm =   0.0203
    Iter   36: loss =   0.0021, grad_norm =   0.0193
    Iter   37: loss =   0.0019, grad_norm =   0.0183
    Iter   38: loss =   0.0017, grad_norm =   0.0174
    Iter   39: loss =   0.0016, grad_norm =   0.0166
    Iter   40: loss =   0.0015, grad_norm =   0.0158
    Iter   41: loss =   0.0013, grad_norm =   0.0151
    Iter   42: loss =   0.0012, grad_norm =   0.0144
    Iter   43: loss =   0.0011, grad_norm =   0.0137
    Iter   44: loss =   0.0010, grad_norm =   0.0131
    Iter   45: loss =   0.0009, grad_norm =   0.0125
    Iter   46: loss =   0.0009, grad_norm =   0.0120
    Iter   47: loss =   0.0008, grad_norm =   0.0114
    Iter   48: loss =   0.0007, grad_norm =   0.0109
    Iter   49: loss =   0.0007, grad_norm =   0.0105
    Iter   50: loss =   0.0006, grad_norm =   0.0100
    ```
