"""Benchmark the MPI and JAX-sharding distributed interfaces.

The benchmark uses the same nonsymmetric tridiagonal system, AmgX
configuration, MPI communicator, and one-GPU-per-rank placement for both
interfaces. It separates JAX compilation, the first execution after clearing
the AmgX resource cache, and steady-state execution for a forward solve,
``dL/db``, and ``dL/dA``, where ``L = 0.5 * ||x||**2``.

Single-node usage:
    CUDA_VISIBLE_DEVICES=0,1 \
      OMPI_MCA_opal_cuda_support=true MPI4JAX_USE_CUDA_MPI=1 \
      mpirun -n 2 python demo/sharding_mpi_benchmark.py
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import jax
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

# Multi-process JAX must initialize before any operation that can inspect the
# backend. A one-rank run needs no coordination service.
if nranks > 1:
    jax.distributed.initialize(cluster_detection_method="mpi4py")
jax.config.update("jax_enable_x64", True)

import jax.experimental.sparse as jsp
import jax.numpy as jnp

import jaxamg
from jaxamg.matrices import tridiagonal_matrix_distributed
from jaxamg.sharding import ShardedSolve


@dataclass
class Timing:
    compile_ms: float
    first_ms: float
    steady_ms: list[float]
    output: jax.Array

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.steady_ms))


def _global_max_time(start: float) -> float:
    elapsed_ms = (perf_counter() - start) * 1000.0
    return float(comm.allreduce(elapsed_ms, op=MPI.MAX))


def _execute_once(compiled: Any, arguments: tuple[jax.Array, ...]):
    comm.Barrier()
    start = perf_counter()
    output = compiled(*arguments)
    output.block_until_ready()
    elapsed_ms = _global_max_time(start)
    return output, elapsed_ms


def _benchmark_callable(
    function: Any,
    arguments: tuple[jax.Array, ...],
    n_runs: int,
) -> Timing:
    comm.Barrier()
    start = perf_counter()
    compiled = function.lower(*arguments).compile()
    compile_ms = _global_max_time(start)

    # Measure the first execution from an empty AmgX resource cache. Compilation
    # is already complete, so this isolates resource creation and execution.
    jaxamg.clear_solver_cache()
    first_output, first_ms = _execute_once(compiled, arguments)

    # Populate the ordinary jit dispatch cache before steady-state timing. This
    # extra invocation is intentionally untimed: mpi4jax threads a hidden token
    # through its compiled gradient, so a directly invoked Lowered executable
    # is valid for the cold call but cannot be reused with the same public
    # argument list. The normal jit callable manages that token across calls.
    function(*arguments).block_until_ready()

    steady_ms = []
    for _ in range(n_runs):
        _, elapsed_ms = _execute_once(function, arguments)
        steady_ms.append(elapsed_ms)

    return Timing(compile_ms, first_ms, steady_ms, first_output)


def _nonsymmetric_tridiagonal(n_global: int) -> tuple[jsp.BCSR, int, int]:
    template, row_start, row_end = tridiagonal_matrix_distributed(
        n_global,
        rank,
        nranks,
        diagonal_value=4.0,
        dtype=jnp.float32,
    )
    global_rows = np.repeat(
        np.arange(row_start, row_end), np.diff(np.asarray(template.indptr))
    )
    columns = np.asarray(template.indices)
    values = np.where(
        columns < global_rows,
        -0.75,
        np.where(columns > global_rows, -1.25, 4.0),
    ).astype(np.float32)
    matrix = jsp.BCSR(
        (jnp.asarray(values), template.indices, template.indptr),
        shape=template.shape,
    )
    return matrix, row_start, row_end


def _initialize_amgx(config: dict[str, Any]) -> None:
    """Remove one-time AmgX initialization from the first measured interface."""
    warmup_size = 4 * nranks
    matrix, row_start, row_end = _nonsymmetric_tridiagonal(warmup_size)
    cache = jaxamg.cache_mpi_metadata(
        config,
        comm,
        warmup_size,
        (row_start, row_end),
        matrix,
        is_symmetric=False,
    )
    matrix = jaxamg.with_cache(matrix, mpi=cache, is_symmetric=False)
    rhs = jnp.ones(row_end - row_start, dtype=matrix.data.dtype)
    jaxamg.solve(matrix, rhs, reuse_setup=True)[0].block_until_ready()
    comm.Barrier()
    jaxamg.clear_solver_cache()
    comm.Barrier()


def _mpi_functions(
    A_local: jsp.BCSR,
    mpi_cache: dict[str, Any],
) -> dict[str, Any]:
    def solution(matrix_data, rhs):
        matrix = jsp.BCSR(
            (matrix_data, A_local.indices, A_local.indptr), shape=A_local.shape
        )
        matrix = jaxamg.with_cache(matrix, mpi=mpi_cache, is_symmetric=False)
        return jaxamg.solve(matrix, rhs, reuse_setup=True)[0]

    def rhs_gradient(matrix_data, rhs):
        x, pullback = jax.vjp(lambda value: solution(matrix_data, value), rhs)
        return pullback(x)[0]

    def matrix_gradient(matrix_data, rhs):
        x, pullback = jax.vjp(lambda value: solution(value, rhs), matrix_data)
        return pullback(x)[0]

    return {
        "forward": jax.jit(solution),
        "dL/db": jax.jit(rhs_gradient),
        "dL/dA": jax.jit(matrix_gradient),
    }


def _sharding_functions(solver: ShardedSolve) -> dict[str, Any]:
    def solution(matrix_data, rhs):
        return solver(rhs, A_data=matrix_data)[0]

    def rhs_gradient(matrix_data, rhs):
        x, pullback = jax.vjp(lambda value: solution(matrix_data, value), rhs)
        return pullback(x)[0]

    def matrix_gradient(matrix_data, rhs):
        x, pullback = jax.vjp(lambda value: solution(value, rhs), matrix_data)
        return pullback(x)[0]

    return {
        "forward": jax.jit(solution),
        "dL/db": jax.jit(rhs_gradient),
        "dL/dA": jax.jit(matrix_gradient),
    }


def _run_interface(
    name: str,
    functions: dict[str, Any],
    arguments: tuple[jax.Array, jax.Array],
    n_runs: int,
) -> dict[str, Timing]:
    timings = {}
    for metric, function in functions.items():
        if rank == 0:
            print(f"Benchmarking {name} {metric}...", flush=True)
        timings[metric] = _benchmark_callable(function, arguments, n_runs)
    return timings


def _relative_error(left: jax.Array, right: jax.Array) -> float:
    left_np = np.asarray(left, dtype=np.float64)
    right_np = np.asarray(right, dtype=np.float64)
    error_sq = comm.allreduce(float(np.sum((left_np - right_np) ** 2)), op=MPI.SUM)
    reference_sq = comm.allreduce(float(np.sum(right_np**2)), op=MPI.SUM)
    return float(np.sqrt(error_sq / reference_sq))


def _print_results(
    mpi_timings: dict[str, Timing],
    sharding_timings: dict[str, Timing],
) -> None:
    if rank != 0:
        return

    print("\nTimes are maximum wall-clock times across MPI ranks (milliseconds).")
    print(
        f"{'Interface':<10} {'Metric':<8} {'Compile':>10} {'First':>10} "
        f"{'Steady avg':>12} {'Steady min':>12} {'Steady max':>12}"
    )
    print("-" * 80)
    for name, timings in (("MPI", mpi_timings), ("Sharding", sharding_timings)):
        for metric, timing in timings.items():
            print(
                f"{name:<10} {metric:<8} {timing.compile_ms:10.2f} "
                f"{timing.first_ms:10.2f} {timing.mean_ms:12.2f} "
                f"{min(timing.steady_ms):12.2f} {max(timing.steady_ms):12.2f}"
            )

    print("\nSteady-state sharding / MPI ratio:")
    for metric in mpi_timings:
        ratio = sharding_timings[metric].mean_ms / mpi_timings[metric].mean_ms
        print(f"  {metric:<8} {ratio:.3f}x")


def main() -> None:
    n_global = 1000000
    n_runs = 5

    if n_global % nranks:
        raise ValueError("n_global must be divisible by the number of MPI ranks")

    jax.config.update("jax_logging_level", "ERROR")
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "max_iters": 100,
        "tolerance": 1e-6,
        "monitor_residual": 0,
        "obtain_timings": 0,
        "print_solve_stats": 0,
    }
    _initialize_amgx(config)

    A_local, row_start, row_end = _nonsymmetric_tridiagonal(n_global)
    b_local_np = (
        1.0 + 0.1 * np.sin(np.arange(row_start, row_end, dtype=np.float32) * 0.001)
    ).astype(np.float32)
    b_local = jnp.asarray(b_local_np)

    mpi_matrix = jsp.BCSR(
        (A_local.data, A_local.indices, A_local.indptr), shape=A_local.shape
    )
    mpi_cache = jaxamg.cache_mpi_metadata(
        config,
        comm,
        n_global,
        (row_start, row_end),
        mpi_matrix,
        is_symmetric=False,
    )

    mesh = jax.make_mesh((nranks,), ("rank",))
    sharding = jax.NamedSharding(mesh, jax.P("rank"))
    b_sharded = jax.make_array_from_process_local_data(
        sharding, b_local_np, global_shape=(n_global,)
    )
    sharded_matrix = jaxamg.make_sharded_matrix(
        A_local, b_sharded, comm=comm, mesh=mesh
    )
    sharded_solver = jaxamg.make_sharded_solver(
        sharded_matrix,
        b_sharded,
        config=config,
        is_symmetric=False,
        reuse_setup=True,
    )

    if rank == 0:
        print(
            f"Nonsymmetric tridiagonal system: n={n_global:,}, "
            f"ranks/GPUs={nranks}, steady runs={n_runs}\n"
        )

    mpi_timings = _run_interface(
        "MPI", _mpi_functions(mpi_matrix, mpi_cache), (mpi_matrix.data, b_local), n_runs
    )
    with jax.set_mesh(mesh):
        sharding_timings = _run_interface(
            "Sharding",
            _sharding_functions(sharded_solver),
            (sharded_matrix.data, b_sharded),
            n_runs,
        )

    sharded_x_local = sharding_timings["forward"].output.addressable_shards[0].data
    sharded_db_local = sharding_timings["dL/db"].output.addressable_shards[0].data
    sharded_dA_local = sharded_matrix.local_matrix(
        sharding_timings["dL/dA"].output
    ).data
    forward_error = _relative_error(sharded_x_local, mpi_timings["forward"].output)
    rhs_gradient_error = _relative_error(sharded_db_local, mpi_timings["dL/db"].output)
    matrix_gradient_error = _relative_error(
        sharded_dA_local, mpi_timings["dL/dA"].output
    )

    _print_results(mpi_timings, sharding_timings)
    if rank == 0:
        print("\nRelative differences (sharding versus MPI):")
        print(f"  forward  {forward_error:.3e}")
        print(f"  dL/db    {rhs_gradient_error:.3e}")
        print(f"  dL/dA    {matrix_gradient_error:.3e}")

    comm.Barrier()
    jaxamg.finalize()
    comm.Barrier()
    if nranks > 1:
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
