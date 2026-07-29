# JAX Sharding

JAX-AMG provides an additive sharding interface for applications that already
use JAX global arrays. It does not replace the [MPI interface](mpi.md): AmgX
still performs the distributed solve through MPI, while `jax.shard_map`
preserves the global sharding of the right-hand side and solution.

The interface targets `shard_map` and `NamedSharding` rather than adding a
separate `pmap` wrapper. This keeps inputs and outputs as JAX global arrays and
fits JAX's current explicit-sharding model.

## Current Scope

The initial interface supports:

- one MPI process and one mesh-local GPU per rank;
- a one-dimensional JAX mesh where device position `i` belongs to MPI rank `i`;
- row-sharded vector and batched right-hand sides with equal or unequal local
  row counts;
- scalar and block matrices, provided every rank's true row count is divisible
  by `block_dim`;
- symmetric and nonsymmetric distributed matrices; and
- JIT compilation and reverse-mode differentiation with respect to matrix
  values and the RHS.

The local CSR structure is fixed when the solver is created. Multiple local GPUs
per MPI process are not supported yet.

## Process and Device Setup

Call `jax.distributed.initialize()` before querying JAX devices. On a single
node, keep all participating GPUs visible to every MPI process and select one
distinct device per process. Do not remap every rank's GPU to CUDA ordinal 0;
JAX/NCCL would then see duplicate devices. Because JAX-AMG already requires
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
the matrix and solver otherwise infer the mesh from `b`.

`A` stores the CSR structure only on its owning rank and exposes its padded,
globally sharded values as `A.data`; it does not replicate the global matrix.
The original local-matrix form remains supported by
`make_sharded_solver(A_local, b)` as a convenience.

When `b_local` or the local matrix values are JAX device arrays, vector and
matrix packing stays on device. NumPy inputs are transferred to the target GPU
once during construction. Solver setup inspects only static CSR structure on
the host to build MPI partition, transpose, and halo metadata.

For a coupled block system, pass `block_dim=k` to `make_sharded_solver`. The
matrix and vectors retain their ordinary scalar CSR/vector representation;
AmgX performs the internal block conversion. Unequal rank partitions remain
supported as long as each partition ends on a block boundary.

`x` is a global array with the same row sharding as `b`. For unequal row counts,
JAX's equal physical shards are padded to the largest local partition; the
solver ignores the padding and returns zeros there. Use `solver.local_vector(x)`
to access this rank's unpadded result. The values in `info` are global arrays,
with one entry per rank.

For multiple right-hand sides sharing the same matrix, pass rank-local values
with shape `(n_local, nrhs)` to `make_sharded_vector`. The returned array and
solution use `PartitionSpec("rank", None)`. AmgX solves the columns sequentially
in a fixed cross-rank order. Matrix gradients are summed over all RHS columns;
RHS gradients retain the padded shape and sharding of the RHS. The scalar info
values have shape `(nranks, nrhs)`, and residual history has shape
`(nranks, nrhs, max_iters + 1)`.

Pass `A.data` to the solve when differentiating matrix values. The
backward-compatible `solver.A_data` attribute aliases the same array. Enter the
mesh context for an outer transformation:

```python
import jax.numpy as jnp

def loss(A_data, rhs):
    x, _ = solver(rhs, A_data=A_data)
    return jnp.sum(x**2)

compiled_solver = jax.jit(solver)
compiled_gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))

with jax.set_mesh(b.sharding.mesh):
    x, info = compiled_solver(b)
    grad_A_data, grad_b = compiled_gradient(A.data, b)

grad_A_local = A.local_matrix(grad_A_data)
```

The solver is compiled internally and also composes with an enclosing
`jax.jit`, including JIT-compiled reverse-mode differentiation.

For a one-off solve, `jaxamg.solve_sharded(...)` accepts the same setup arguments
and immediately invokes the resulting solver. Prefer `make_sharded_solver` in
loops so metadata and compiled executables are reused.

Run the complete single-node example with:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMPI_MCA_opal_cuda_support=true \
mpirun -n 2 python demo/sharded_poisson_problem.py
```

`MPI4JAX_USE_CUDA_MPI` remains relevant to the existing MPI autodiff path but
is not required for the sharding path. Matrix gradients use a sparse JAX
all-to-all halo exchange. For a nonsymmetric matrix, the transpose structure is
cached during solver creation and its values are updated for each adjoint solve;
the forward and adjoint solves are both performed by AmgX.
