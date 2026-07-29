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

## Required XLA Flag for Multi-Process Jobs

A sharded solver compiles each rank's local CSR structure and communication
plans into that process's program as constants, so the processes compile
programs that are not identical. XLA's cross-process *sharded autotuning*
assumes identical programs and deadlocks during compilation as soon as such a
program also contains an automatically partitioned collective. A loss that
reduces the solution across ranks inside `jax.jit`, such as
`jnp.sum(x**2)`, produces exactly that collective, so disable the autotuning:

```bash
XLA_FLAGS=--xla_gpu_shard_autotuning=false \
  mpirun -n 2 python your_script.py
```

The flag is read when JAX initializes its backend, so set it in the environment
or via `os.environ` before the first JAX device call. Without it, a compiled
loss that reduces across ranks hangs in compilation rather than failing;
uncompiled calls and `jax.jit(jax.grad(...))` are unaffected, because neither
emits that collective.

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

A direct `solver(b)` call can use the matrix's cached values. Pass `A.data`
explicitly under an enclosing JAX transformation, including RHS-only
differentiation. This keeps the distributed matrix values as a dynamic operand
instead of embedding a full local value buffer in the compiled executable.

Compilation is entirely the caller's decision: the solver adds no `jax.jit` of
its own. `jax.value_and_grad(loss)` runs the whole pipeline eagerly, while
`jax.jit(jax.value_and_grad(loss))` compiles it as one program. Both give the
same results and both match the MPI interface's speed: a compiled caller runs
the solver's `shard_map` regions inside its one program, while an untransformed
call executes the rank-local pipeline directly on this process's shard — the
same code path as `solve(..., comm=...)` — and reassembles the global arrays,
instead of paying for JAX's much slower per-primitive eager `shard_map`
execution.

Run the complete single-node examples with:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
OMPI_MCA_opal_cuda_support=true \
mpirun -n 2 python demo/sharded_poisson_problem.py

CUDA_VISIBLE_DEVICES=0,1 \
OMPI_MCA_opal_cuda_support=true \
mpirun -n 2 python demo/sharded_poisson_operator_optimization.py
```

The second example recovers an operator parameter by gradient descent, the
sharded counterpart of `demo/mpi_poisson_operator_optimization.py`. Because the
loss is a reduction over global arrays, JAX emits the cross-rank collectives and
the gradient it returns is already the complete global gradient, so no
`comm.allreduce` of losses or gradients is needed.

Matrix gradients use a sparse all-to-all halo exchange: `jax.lax.all_to_all`
inside a compiled program, and the MPI interface's mpi4jax exchange for
untransformed calls — so `MPI4JAX_USE_CUDA_MPI` applies to eager sharded
gradients exactly as it does to the MPI autodiff path. For a nonsymmetric
matrix, the transpose structure and a sparse value-exchange plan are cached
during solver creation. Only off-rank transpose values are exchanged, once per
backward pass, rather than replicating all matrix values on every GPU. The
forward and adjoint solves are both performed by AmgX.
