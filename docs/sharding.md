# JAX Sharding

JAX-AMG provides an additive sharding interface for applications that already
use JAX global arrays. It does not replace the [MPI interface](mpi.md): AmgX
still performs the distributed solve through MPI, while `jax.shard_map`
preserves the global sharding of the right-hand side and solution.

The interface targets `shard_map` and `NamedSharding` rather than adding a
separate `pmap` wrapper. This keeps inputs and outputs as JAX global arrays and
fits JAX's current explicit-sharding model.

The interface is experimental; expect the API to evolve.

## Current Scope

The initial interface supports:

- one MPI process and one mesh-local GPU per rank, with matching GPU models;
- a mesh in MPI rank order, with rows partitioned over `axis_name` (one axis
  or a tuple); all other axes must have size one;
- row-sharded vectors with equal or unequal local row counts;
- scalar and block matrices, provided every rank's true row count is divisible
  by `block_dim`;
- symmetric and nonsymmetric distributed matrices;
- singular systems with declared null spaces; and
- JIT compilation and differentiation with respect to matrix values, the RHS
  and operator parameters.

The CSR structure is fixed at creation. The communicator must include every
JAX process, with at least one row per rank.

## Process and Device Setup

Call `jax.distributed.initialize()` before querying JAX devices. On a single
node, keep all participating GPUs visible to every MPI process and select one
distinct device per process. Do not remap every rank's GPU to CUDA ordinal 0;
JAX/NCCL would then see duplicate devices. Because this interface uses
`mpi4py`, its cluster-detection method can derive the coordinator, process IDs,
and local device assignment from the MPI job:

```python
import jax

jax.distributed.initialize(
    cluster_detection_method="mpi4py",
)
```

Launchers recognized directly by JAX may also work with an argument-free
`jax.distributed.initialize()`. The explicit `mpi4py` method also covers MPI
environments whose launcher variables JAX does not recognize automatically.

## XLA Sharded Autotuning

Import `jaxamg` before querying JAX devices. It disables sharded autotuning
to prevent deadlocks when ranks compile different operator code, unless you
explicitly set the flag in `XLA_FLAGS`.

If JAX is already initialized, pass
`compiler_options={"xla_gpu_shard_autotuning": False}` to the outer `jax.jit`.

## Sharded Solve

Build a global RHS from each process's local partition, then create the solver
once and reuse it:

```python
import jaxamg

b = jaxamg.make_sharded_vector(
    b_local,
    global_size=n_global,
)

A = jaxamg.make_sharded_matrix(A_local, b)
solver = jaxamg.make_sharded_solver(
    A,
    b,
    config={"solver": "GMRES", "communicator": "MPI_DIRECT"},
)

x, info = solver(b)
```

The sharding helpers use `MPI.COMM_WORLD` and a one-dimensional mesh over all
JAX devices by default. Pass `comm=` or `mesh=` explicitly to override them;
the matrix otherwise infers the mesh from `b`, and the solver uses the matrix's
communicator and mesh.

On a multi-axis mesh, pass the mesh and the axes that partition the rows:

```python
mesh = jax.make_mesh((2, 2, 1), ("x", "y", "z"))  # the application's mesh
b = jaxamg.make_sharded_vector(b_local, mesh=mesh, axis_name=("x", "y"))
A = jaxamg.make_sharded_matrix(A_local, b, axis_name=("x", "y"))
```

`A` stores the CSR structure only on its owning rank and exposes its padded,
globally sharded values as `A.data`; it does not replicate the global matrix.

When `b_local` or the local matrix values are JAX device arrays, vector and
matrix packing stays on device. NumPy inputs are transferred to the target GPU
once during construction. Solver setup inspects only static CSR structure on
the host to build MPI partition, transpose, and halo metadata.

For a coupled block system, pass `block_dim=k` to `make_sharded_solver`. The
matrix and vectors retain their ordinary scalar CSR/vector representation;
AmgX performs the internal block conversion. Unequal rank partitions remain
supported as long as each partition ends on a block boundary.

To save detailed AmgX solver statistics, create the solver with
`save_stats=True` and pass a file path to a direct call:

```python
solver = jaxamg.make_sharded_solver(A, b, save_stats=True)
x, info = solver(b, save_stats_file="stats_sharded.txt")
```

As with `jaxamg.solve`, rank 0 writes the formatted file. Statistics can only
be saved from a direct call, not from inside `jax.jit` or another transform.

`x` is a global array with the same row sharding as `b`. For unequal row counts,
JAX's equal physical shards are padded to the largest local partition; the
solver ignores the padding and returns zeros there. Use `solver.local_vector(x)`
to access this rank's unpadded result. The values in `info` are global arrays,
with one entry per rank.

For multiple right-hand sides, construct each global sharded vector separately
and apply `jax.vmap` to the single-vector solver, just as with `jaxamg.solve`:

```python
import jax
import jax.numpy as jnp

batched_b = jnp.stack((b1, b2))
batched_x, batched_info = jax.vmap(
    lambda rhs: solver(rhs, A=A.data)
)(batched_b)
```

The underlying AmgX solves use JAX-AMG's sequential FFI batching path.

`A=` accepts either this rank's operator or matrix (next section) or the
packed global values `A.data`. Pass `A.data` to differentiate matrix entries;
the gradient has the same layout. Enter the mesh context for an outer
transformation:

```python
import jax.numpy as jnp

def loss(A_data, rhs):
    x, _ = solver(rhs, A=A_data)
    return jnp.sum(x**2)

compiled_solver = jax.jit(lambda A_data, rhs: solver(rhs, A=A_data))
compiled_gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))

with jax.set_mesh(b.sharding.mesh):
    x, info = compiled_solver(A.data, b)
    grad_A_data, grad_b = compiled_gradient(A.data, b)

grad_A_local = A.local_matrix(grad_A_data)
```

A direct `solver(b)` call uses the cached values. Under a JAX transformation
`A` is required, even for RHS-only differentiation, so the values stay a
dynamic operand rather than a constant baked into the executable.

## Singular Systems

Attach the null-space bases (local rows) to the matrix before packing it, as
for `solve`:

```python
A_local = jaxamg.with_cache(A_local, nullspace="constant", transpose_nullspace=V_local)
A = jaxamg.make_sharded_matrix(A_local, b)
solver = jaxamg.make_sharded_solver(A, b, config=config)
```

These bases are defaults for each solve. Override them with
`solver(b, A=values, nullspace=..., transpose_nullspace=..., labels=...)`,
using global arrays sharded like `b`, or `"constant"`.

As in [singular solves](examples.md#singular-systems), the RHS is projected
and the solution is pinned. Every rank must agree on the basis column counts;
matrix perturbations must preserve the declared null spaces.
`info["rhs_inconsistency"]` reports one value per rank.

For [disconnected domains](examples.md#disconnected-domains), use
`labels=(count, labels)` with `-1 <= label < count`, including under tracing.
Labels normally use global numbering. Custom linear `label_sum` functions map each
rank's `count` partial sums to totals in the same global sharded layout.

## Differentiating Operator Parameters

When the values depend on parameters, pass this rank's operator or matrix as
`A=`, exactly as `solve(A, b)` takes it in the MPI interface. The solver
materializes it with the sparsity fixed at construction and differentiates
through it. A matrix must carry exactly that CSR structure, with concrete
indices and row pointers; traced values with the fixed structure go through
the packed `A.data` layout.

```python
from jaxamg.matrices import poisson_operator
from jaxamg.mpi_utils import partition_operator

def local_operator(skew):
    operator, _, _ = partition_operator(
        poisson_operator(skew), n_global, rank, nranks
    )
    return operator

coloring = jaxamg.cache_coloring(local_operator(0.0), shape=(n_local, n_global))
A = jaxamg.make_sharded_matrix(
    jaxamg.with_cache(local_operator(0.0), coloring=coloring), b
)
solver = jaxamg.make_sharded_solver(A, b)

def loss(skew, rhs, x_target):
    x, _ = solver(rhs, A=local_operator(skew))
    return jnp.sum((x - x_target) ** 2) / n_global

with jax.set_mesh(A.mesh):
    value, grad_skew = jax.value_and_grad(loss)(skew, b, x_target)
```

Each rank materializes its own rows. Closed-over parameters must be identical
on every rank; their gradients are summed globally. Pass sharded matrix values
through `A.data`.

Wrap repeated solves in `jax.jit`. An untransformed call runs the rank-local
pipeline directly on this process's shard.

Run the complete single-node examples with:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMPI_MCA_opal_cuda_support=true \
mpirun -n 2 python demo/sharded_poisson_problem.py

CUDA_VISIBLE_DEVICES=0,1 \
OMPI_MCA_opal_cuda_support=true \
mpirun -n 2 python demo/sharded_poisson_operator_optimization.py
```

The second example is the sharded counterpart of
`demo/mpi_poisson_operator_optimization.py`: it recovers an operator parameter
by gradient descent through `solver(rhs, A=...)`. The loss is a reduction over
global arrays, so its gradient is already global and no `comm.allreduce` is
needed.

Matrix gradients exchange halo and transpose values with neighbouring ranks
only. `MPI4JAX_USE_CUDA_MPI=1` passes device buffers to MPI directly.

## Halo-form and global-view operators

`make_sharded_matrix` and `solver(rhs, A=...)` also accept:

- [`jaxamg.halo_operator(...)`](mpi.md#halo-form-operators);
- `jaxamg.global_operator(fn, indices, indptr, comm=comm, mesh=mesh)`: `fn`
  acts on global row-sharded vectors and does its own communication;
  `indices` and `indptr` give this rank's rows (global columns). Row
  partitions must be equal.

Update parameters with `A=op.with_fn(new_fn)`, keeping the same sparsity.
See `demo/sharded_global_operator_optimization.py`.

For programs with their own collectives, use `global_operator` or `A.data`
so ranks compile the same program. With explicit mesh axes, pass traced
parameters as `shard_map` arguments rather than capturing them in closures.
