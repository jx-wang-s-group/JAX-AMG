"""
Demo: Benchmark MPI AmgX resource cache performance.

This demo compares:
1) Baseline: force AmgX resource cache misses each iteration
2) Cached: reuse AmgX solver cache across iterations

Usage:
    mpirun -n 4 python demo/mpi_solver_cache_benchmark.py
"""

import time

import jax
import jax.experimental.sparse as jsp
from mpi4py import MPI

import jaxamg
from jaxamg.matrices import rhs_ones, tridiagonal_matrix_distributed


def _summarize_times(comm, rank, label, local_times):
    all_times = comm.gather(local_times, root=0)
    if rank != 0:
        return None

    flat_times = [t for rank_times in all_times for t in rank_times]
    avg_t = sum(flat_times) / len(flat_times)
    min_t = min(flat_times)
    max_t = max(flat_times)
    print(f"{label} Average: {avg_t:.2f} ms")
    print(f"{label} Min: {min_t:.2f} ms")
    print(f"{label} Max: {max_t:.2f} ms")
    return avg_t


def run_benchmark(n_global, n_runs=5):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    _gpus = jax.devices()
    jax.config.update("jax_default_device", _gpus[rank % len(_gpus)])

    if rank == 0:
        print(
            f"\nBenchmarking MPI tridiagonal matrix of size {n_global} x {n_global}..."
        )
        print(f"MPI ranks: {nranks}")

    A_local, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0
    )
    n_local = row_end - row_start
    b_local = rhs_ones(n_local, dtype=A_local.data.dtype)
    partition_info = (row_start, row_end)

    comm.Barrier()
    if rank == 0:
        print(f"\nPartitioning matrix across {nranks} ranks...")
    comm.Barrier()
    print(f"  Rank {rank}: {n_local} rows [{row_start}:{row_end})")
    comm.Barrier()

    # Keep solve work light so resource-cache effects are easier to observe.
    solver_config = {
        "solver": "JACOBI_L1",
        "max_iters": 1,
        "monitor_residual": 0,
        "obtain_timings": 0,
        "print_solve_stats": 0,
        "communicator": "MPI_DIRECT",
    }

    def step(A_template, b_loc, key_, mpi_cache_):
        perturbation = jax.random.uniform(
            key_,
            A_template.data.shape,
            minval=-0.01,
            maxval=0.01,
            dtype=A_template.data.dtype,
        )
        A_new = jsp.BCSR(
            (A_template.data + perturbation, A_template.indices, A_template.indptr),
            shape=A_template.shape,
        )
        A_cached = jaxamg.with_cache(A_new, mpi=mpi_cache_, is_symmetric=True)
        x, _ = jaxamg.solve(A_cached, b_loc)
        return x

    baseline_times = []
    key = jax.random.PRNGKey(0)

    mpi_cache = jaxamg.cache_mpi_metadata(
        solver_config, comm, n_global, partition_info, A_local, is_symmetric=True
    )

    if rank == 0:
        print("\nRunning baseline...")

    # Warmup baseline
    key, warmup_key = jax.random.split(key)
    step(A_local, b_local, warmup_key, mpi_cache).block_until_ready()
    comm.Barrier()

    for _ in range(n_runs):
        key, subkey = jax.random.split(key)
        # Drop the cached AmgX resources (all ranks in lockstep) so every
        # timed solve pays the full miss cost: resource creation, matrix
        # upload, and solver setup. The clear itself is not timed.
        jaxamg.clear_solver_cache()
        comm.Barrier()
        t0 = time.time()
        x = step(A_local, b_local, subkey, mpi_cache)
        x.block_until_ready()
        comm.Barrier()
        baseline_times.append((time.time() - t0) * 1000)

    avg_baseline = _summarize_times(comm, rank, "Baseline", baseline_times)

    cached_times = []
    key = jax.random.PRNGKey(0)

    # Start the cached branch from a miss as well: its warmup repopulates the
    # cache, then every timed solve is a hit that reuses the AmgX resources.
    jaxamg.clear_solver_cache()
    comm.Barrier()

    if rank == 0:
        print("\nRunning cached...")

    # Warmup cached
    key, warmup_key = jax.random.split(key)
    step(A_local, b_local, warmup_key, mpi_cache).block_until_ready()
    comm.Barrier()

    for _ in range(n_runs):
        key, subkey = jax.random.split(key)
        comm.Barrier()
        t0 = time.time()
        x = step(A_local, b_local, subkey, mpi_cache)
        x.block_until_ready()
        comm.Barrier()
        cached_times.append((time.time() - t0) * 1000)

    avg_cached = _summarize_times(comm, rank, "Cached", cached_times)

    if rank == 0 and avg_cached is not None and avg_baseline is not None:
        print(f"\nSpeedup: {avg_baseline / avg_cached:.2f}x")

    comm.Barrier()
    jaxamg.finalize()
    return avg_cached if rank == 0 else None


if __name__ == "__main__":

    run_benchmark(n_global=5000, n_runs=5)
