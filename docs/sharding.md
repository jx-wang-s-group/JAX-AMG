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
- row-sharded vectors with equal or unequal local row counts;
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
    lambda rhs: solver(rhs, A_data=A.data)
)(batched_b)
```

The underlying AmgX solves use JAX-AMG's sequential FFI batching path.

Pass `A.data` to the solve when differentiating matrix values. Enter the mesh
context for an outer transformation:

```python
import jax.numpy as jnp

def loss(A_data, rhs):
    x, _ = solver(rhs, A_data=A_data)
    return jnp.sum(x**2)

compiled_solver = jax.jit(
    lambda A_data, rhs: solver(rhs, A_data=A_data)
)
compiled_gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))

with jax.set_mesh(b.sharding.mesh):
    x, info = compiled_solver(A.data, b)
    grad_A_data, grad_b = compiled_gradient(A.data, b)

grad_A_local = A.local_matrix(grad_A_data)
```

The solver is compiled internally, so a direct `solver(b)` call can use the
matrix's cached values. Pass `A.data` explicitly under an enclosing JAX
transformation, including RHS-only differentiation. This keeps the distributed
matrix values as a dynamic operand instead of embedding a full local value
buffer in the compiled executable.

Run the complete single-node example with:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMPI_MCA_opal_cuda_support=true \
mpirun -n 2 python demo/sharded_poisson_problem.py
```

`MPI4JAX_USE_CUDA_MPI` remains relevant to the existing MPI autodiff path but
is not required for the sharding path. Matrix gradients use a sparse JAX
all-to-all halo exchange. For a nonsymmetric matrix, the transpose structure
and a sparse value-exchange plan are cached during solver creation. Only
off-rank transpose values are exchanged, once per backward pass, rather than
replicating all matrix values on every GPU. The forward and adjoint solves are
both performed by AmgX.
