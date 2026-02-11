"""
Demo: Benchmark solver cache performance.

This demo benchmarks the performance of the solver cache.
Run with different `JAXAMG_CACHE_SIZE` values to see the impact on performance.
"""

import time

import jax
import jax.experimental.sparse as jsp

import jaxamg
from jaxamg.matrices import poisson_matrix, rhs_ones


def run_benchmark(n_rows, n_runs=5):
    print(f"\nBenchmarking Poisson matrix of size {n_rows} x {n_rows}...")

    # Setup Poisson matrix
    A = poisson_matrix(n_rows)
    b = rhs_ones(n_rows**2)

    solver_config = {
        "config_version": 2,
        "solver": {
            "solver": "JACOBI_L1",
            "max_iters": 1,
            "monitor_residual": 0,
            "obtain_timings": 0,
        },
    }

    # Define step function with perturbation
    @jax.jit
    def step_fn(A, b, key):
        perturbation = jax.random.uniform(key, A.data.shape, minval=-0.01, maxval=0.01)
        new_data = A.data + perturbation
        A_new = jsp.BCSR((new_data, A.indices, A.indptr), shape=A.shape)
        x, info = jaxamg.solve(A_new, b, config=solver_config)
        return x.sum()

    # Run baseline
    print("Running baseline...")
    baseline_times = []
    key = jax.random.PRNGKey(0)
    jaxamg.clear_solver_cache()

    # Warmup
    step_fn(A, b, key).block_until_ready()

    for i in range(n_runs):
        key, subkey = jax.random.split(key)
        jaxamg.clear_solver_cache()  # Clear cache to force new resources
        t0 = time.time()
        x = step_fn(A, b, subkey)
        x.block_until_ready()
        t1 = time.time()
        baseline_times.append((t1 - t0) * 1000)

    avg_baseline = sum(baseline_times) / len(baseline_times)
    print(f"Average Baseline Time: {avg_baseline:.2f} ms")
    print(f"Min Baseline Time: {min(baseline_times):.2f} ms")
    print(f"Max Baseline Time: {max(baseline_times):.2f} ms")

    # Run cached
    print("\nRunning cached...")
    cached_times = []
    key = jax.random.PRNGKey(0)
    jaxamg.clear_solver_cache()

    # Warmup
    step_fn(A, b, key).block_until_ready()

    for i in range(n_runs):
        key, subkey = jax.random.split(key)
        t0 = time.time()
        x = step_fn(A, b, subkey)
        x.block_until_ready()
        t1 = time.time()
        cached_times.append((t1 - t0) * 1000)

    avg_cached = sum(cached_times) / len(cached_times)
    print(f"Average Cached Time: {avg_cached:.2f} ms")
    print(f"Min Cached Time: {min(cached_times):.2f} ms")
    print(f"Max Cached Time: {max(cached_times):.2f} ms")

    print(f"\nSpeedup:  {avg_baseline / avg_cached:.1f}x")

    return avg_cached


if __name__ == "__main__":

    n_rows = 5000
    run_benchmark(n_rows)
