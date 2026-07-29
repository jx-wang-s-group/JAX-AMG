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
- a row-sharded, one-dimensional right-hand side with equal or unequal local
  row counts;
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
import numpy as np
import jaxamg

mesh = jax.make_mesh((nranks,), ("rank",))
b = jaxamg.make_sharded_vector(
    np.asarray(b_local),
    comm=comm,
    mesh=mesh,
    global_size=n_global,
)

solver = jaxamg.make_sharded_solver(
    A_local,
    b,
    comm=comm,
    mesh=mesh,
    config={"solver": "GMRES", "communicator": "MPI_DIRECT"},
)

x, info = solver(b)
```

`x` is a global array with the same row sharding as `b`. For unequal row counts,
JAX's equal physical shards are padded to the largest local partition; the
solver ignores the padding and returns zeros there. Use `solver.local_vector(x)`
to access this rank's unpadded result. The values in `info` are global arrays,
with one entry per rank.

The solver exposes the packed, sharded CSR values as `solver.A_data`. Pass them
to the solve when differentiating matrix values. Enter the mesh context for an
outer transformation:

```python
import jax.numpy as jnp

def loss(A_data, rhs):
    x, _ = solver(rhs, A_data=A_data)
    return jnp.sum(x**2)

compiled_solver = jax.jit(solver)
compiled_gradient = jax.jit(jax.grad(loss, argnums=(0, 1)))

with jax.set_mesh(mesh):
    x, info = compiled_solver(b)
    grad_A_data, grad_b = compiled_gradient(solver.A_data, b)

grad_A_local = solver.local_matrix_gradient(grad_A_data)
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
