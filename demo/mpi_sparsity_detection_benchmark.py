"""
Benchmark: sparsity-detection cost -- tracing vs probing -- in an MPI setting.

Before jaxamg can hand a matrix-free operator to AmgX it must recover the
operator's sparsity pattern. ``jaxamg.sparsity`` does this two ways:

  * TRACING (default): interpret the operator's jaxpr and propagate a sparse
    connectivity structure -- the exact pattern in one host-side pass, in O(nnz)
    memory, with no operator evaluations.
  * PROBING (fallback): apply the operator to ``n_global`` one-hot basis vectors
    and read off the nonzeros -- correct for any operator, but it materialises
    large dense probe batches on the GPU and costs O(n_global) evaluations.

This benchmark partitions a global 32^3 Poisson operator across 2 ranks and, on
each rank, runs both detectors on the same local ``(n_local, n_global)`` block,
reporting the wall time and GPU peak memory each drives. Both yield the same
pattern; tracing is far faster at negligible GPU cost. (Much larger grids make
probing's dense one-hot batch exceed XLA's scheduler limit and abort, which is
exactly the regime where the tracing default earns its keep.)

Usage:
    mpirun -n 2 python demo/mpi_sparsity_detection_benchmark.py
"""

from time import perf_counter

import jax
from mpi4py import MPI

from jaxamg.matrices import poisson3d_operator
from jaxamg.mpi_utils import get_partition_info, partition_operator
from jaxamg.sparsity import probe_sparsity_pattern, trace_sparsity_pattern

jax.config.update("jax_enable_x64", True)


def _pattern_set(res):
    return None if res is None else set(zip(res[0].tolist(), res[1].tolist()))


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    gpus = jax.devices()
    dev = gpus[rank % len(gpus)]
    jax.config.update("jax_default_device", dev)

    grid = 32
    n_global = grid**3
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)

    local_op, _, _ = partition_operator(
        poisson3d_operator(robin=2.0), n_global, rank, nranks
    )
    shape = (n_local, n_global)

    def gpu_peak_gb():
        return dev.memory_stats().get("peak_bytes_in_use", 0) / 1e9

    # Tracing: exact pattern from the jaxpr (host-side, ~no GPU).
    base = gpu_peak_gb()
    t0 = perf_counter()
    traced = trace_sparsity_pattern(local_op, shape)
    t_trace = perf_counter() - t0
    gpu_trace = max(gpu_peak_gb() - base, 0.0)

    # Probing: one-hot basis vectors on the GPU (the fallback).
    mid = gpu_peak_gb()
    t0 = perf_counter()
    probed = probe_sparsity_pattern(local_op, shape)
    t_probe = perf_counter() - t0
    gpu_probe = max(gpu_peak_gb() - mid, 0.0)

    match = _pattern_set(traced) == _pattern_set(probed)
    speedup = t_probe / t_trace if t_trace > 0 else float("inf")

    if rank == 0:
        print(f"3D Poisson, grid {grid}^3 = {n_global:,} unknowns, {nranks} ranks\n")
        print(
            f"{'rank':<5}{'n_local':>10}{'trace (s)':>11}{'probe (s)':>11}"
            f"{'speedup':>9}{'trace GPU':>11}{'probe GPU':>11}{'match':>7}"
        )
        print("-" * 74)
    comm.Barrier()

    for r in range(nranks):
        if r == rank:
            print(
                f"{rank:<5}{n_local:>10}{t_trace:>11.3f}{t_probe:>11.3f}"
                f"{speedup:>8.0f}x{gpu_trace:>10.2f}G{gpu_probe:>10.2f}G{str(match):>7}"
            )
        comm.Barrier()


if __name__ == "__main__":
    main()
