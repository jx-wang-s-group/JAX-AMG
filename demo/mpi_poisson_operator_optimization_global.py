"""
Demo: MPI-distributed optimization of a Poisson operator skew parameter.

Demonstrates end-to-end JIT compilation and differentiation of a custom JAX operator
in an MPI setting, with loss defined on the assembled global solution.

Usage:
    mpirun -n 4 python demo/mpi_poisson_operator_optimization_global.py
"""

from mpi4py import MPI
import jax
import jax.numpy as jnp

import jaxamg
from jaxamg.matrices import (
    poisson_operator,
    poisson_operator_distributed,
    rhs_ones,
)
from jaxamg.mpi_utils import get_partition_info

jax.config.update("jax_enable_x64", True)


def make_gather_solution(partition_info, comm, total_size, arr_dtype):
    """Build a differentiable MPI Allgatherv callable.

    Precomputes partition layout once and registers a custom VJP so that
    jax.grad can backpropagate through the gather: the backward pass simply
    slices the incoming global gradient back to each rank's local segment.
    """
    import numpy as np
    from mpi4py import MPI

    _rank = comm.Get_rank()
    _local_size = np.array(partition_info['row_end'] - partition_info['row_start'], dtype=np.int32)
    _all_sizes = np.empty(comm.Get_size(), dtype=np.int32)
    comm.Allgather(_local_size, _all_sizes)
    _displacements = np.insert(np.cumsum(_all_sizes[:-1]), 0, 0).astype(np.int32)
    _total = int(total_size)
    _mpi_dtype = MPI.FLOAT if arr_dtype == jnp.float32 else MPI.DOUBLE
    _np_dtype = np.float32 if _mpi_dtype == MPI.FLOAT else np.float64
    _disp = int(_displacements[_rank])
    _lsize = int(_all_sizes[_rank])

    @jax.custom_vjp
    def gather_solution(x_local):
        def _allgatherv(x_np):
            # pure_callback passes arrays with byte-order prefix (e.g. '=f8');
            # ascontiguousarray with explicit dtype strips it so mpi4py can resolve it
            x_np = np.ascontiguousarray(x_np, dtype=_np_dtype)
            recvbuf = np.empty(_total, dtype=_np_dtype)
            comm.Allgatherv(x_np, [recvbuf, _all_sizes, _displacements, _mpi_dtype])
            return recvbuf

        result_shape = jax.ShapeDtypeStruct((total_size,), x_local.dtype)
        return jax.pure_callback(_allgatherv, result_shape, x_local)

    def _gather_fwd(x_local):
        return gather_solution(x_local), None

    def _gather_bwd(_, g_global):
        # Scatter: each rank takes its slice of the incoming global gradient
        return (g_global[_disp : _disp + _lsize],)

    gather_solution.defvjp(_gather_fwd, _gather_bwd)
    return gather_solution


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    # Pin each rank to a distinct GPU. CUDA_VISIBLE_DEVICES is ineffective here
    # because OpenMPI initialises CUDA before Python code runs. Instead we use
    # jax.default_device() so all JAX operations on this process land on GPU #rank.
    _gpus = jax.devices()
    jax.config.update("jax_default_device", _gpus[rank % len(_gpus)])

    # Problem size
    grid_size = 16
    n_global = grid_size * grid_size

    if rank == 0:
        print(f"Setting up MPI Poisson Optimization on {grid_size}x{grid_size} grid...")
        print(f"MPI ranks: {nranks}")
        print()

    # Ground truth
    true_skew = 5.0

    # Generate ground truth solution using single-GPU on rank 0
    if rank == 0:
        b_global = rhs_ones(n_global)
        A_true = poisson_operator(true_skew)

        x_target_global, info = jaxamg.solve(
            A_true,
            b_global,
            solver="PBICGSTAB",
            preconditioner={"solver": "JACOBI_L1"},
            tolerance=1e-8,
        )
    else:
        x_target_global = None

    # Broadcast ground truth to all ranks
    x_target_global = comm.bcast(x_target_global, root=0)

    # Partition
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)

    b_local = rhs_ones(n_local)

    comm.Barrier()
    print(f"  Rank {rank}: {n_local} rows [{row_start}:{row_end})")
    comm.Barrier()

    # Configuration for solver
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "max_iters": 50,
        "tolerance": 1e-6,
    }

    # Create dummy operator for caching
    dummy_op = poisson_operator_distributed(grid_size, row_start, row_end, skew=1.0)

    # Cache MPI metadata
    mpi_cache = jaxamg.cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), dummy_op
    )

    # Cache coloring
    coloring_cache = jaxamg.cache_coloring(dummy_op, shape=(n_local, n_global))

    # Precompute the differentiable gather once (not inside the loss)
    gather_solution = make_gather_solution(
        {'row_start': row_start, 'row_end': row_end},
        comm, n_global, jnp.float64,
    )

    if rank == 0:
        print("\nStarting optimization...")

    # Loss on the global solution.
    # Forward:  solve locally → gather globally → MSE against global target.
    # Backward: gradient w.r.t. skew flows back through the gather (VJP slices
    #           the global gradient to this rank's segment) then through the
    #           distributed solve.  Each rank produces its own local contribution;
    #           allreduce(SUM) in the loop assembles the full gradient.
    def loss_global(skew, b_loc):
        op = poisson_operator_distributed(grid_size, row_start, row_end, skew=skew)
        A = jaxamg.with_cache(op, coloring=coloring_cache, mpi=mpi_cache)
        x_pred_loc, info = jaxamg.solve(A, b_loc)
        x_pred_global = gather_solution(x_pred_loc)
        return jnp.sum((x_pred_global - x_target_global) ** 2) / n_global

    grad_fn = jax.jit(jax.grad(loss_global))

    # Optimization loop
    skew_init = 0.0
    lr = 0.1

    if rank == 0:
        print(f"{'Epoch':<6} {'Skew':<12} {'Global Loss':<15} {'Gradient':<12}")
        print("-" * 50)

    for epoch in range(200):
        # Force solver rebuild periodically
        if epoch % 20 == 0:
            jaxamg.clear_solver_cache()

        # All ranks compute the same global loss (x_pred_global is identical on
        # every rank after the gather), so no allreduce is needed for the loss.
        l_global = float(loss_global(skew_init, b_local))

        # Each rank's gradient is only its local contribution (VJP slices back);
        # sum across ranks to get the full gradient of the global loss.
        g_loc = grad_fn(skew_init, b_local)
        g_global = comm.allreduce(float(g_loc), op=MPI.SUM)

        if rank == 0:
            print(f"{epoch:<6} {skew_init:<12.4f} {l_global:<15.6f} {g_global:<12.6f}")

        # All ranks apply the same update
        skew_init -= lr * g_global

        if l_global < 1e-6:
            if rank == 0:
                print("\nConverged!")
            break

    if rank == 0:
        print(f"\nFinal skew: {skew_init:.4f}, True skew: {true_skew:.4f}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
