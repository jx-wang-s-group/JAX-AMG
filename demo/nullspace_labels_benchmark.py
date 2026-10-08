"""
Benchmark: a disconnected domain's null space as labels or as dense columns.

A 2D Poisson matrix (symmetric flux form, Neumann) on ``side × side`` cells cut
into ``tiles × tiles`` uncoupled tiles: one constant null vector per tile.
Declared as ``nullspace="constant"`` with one label per row, or as one dense
indicator column per tile (``nullspace=N``, ``N`` of shape ``(n, L)``), attached
with ``with_cache`` (constants of the jitted solve). One case per process, so
the device's peak memory is the case's own: setup time (the first jitted solve:
compilation, the checks and the AmgX setup), per-solve time (median of
repeated solves, each a fixed 20 iterations), iterations and peak memory,
appended as a JSON line.

Single GPU (``CASE`` = ``tiles:form``, form ``labels``, ``constant`` or
``dense``)::

    CASE=16:labels CUDA_VISIBLE_DEVICES=0 python demo/nullspace_labels_benchmark.py out.jsonl

MPI mode (rows split over the ranks, labels numbered globally)::

    CASE=16:labels CUDA_VISIBLE_DEVICES=0,1 mpirun -n 2 python demo/nullspace_labels_benchmark.py out.jsonl
"""

import json
import os
import statistics
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import jaxamg

jax.config.update("jax_enable_x64", True)

side = int(os.environ.get("SIDE", 1024))
# A fixed number of iterations (an unreachable tolerance), so every form does the
# same solver work and the times compare the null-space handling. AmgX's own
# convergence on these singular systems varies run to run with either form.
cfg = {"solver": "PCG", "tolerance": 1e-300, "max_iters": 20}
repeats = 10
dense_limit = 4 * 2**30  # bytes of dense columns per rank, at most


def tiled_poisson(side, tiles):
    """The tiles' Neumann Laplacians, rows ordered tile by tile, and the
    tile of every row."""
    m = side // tiles
    line = sp.diags(
        [-np.ones(m - 1), np.r_[1.0, 2 * np.ones(m - 2), 1.0], -np.ones(m - 1)],
        [-1, 0, 1],
    )
    tile = (sp.kron(line, sp.eye(m)) + sp.kron(sp.eye(m), line)).tocsr()
    A = sp.kron(sp.eye(tiles * tiles), tile, format="csr")
    labels = np.repeat(np.arange(tiles * tiles), m * m)
    return A, labels


def peak_bytes():
    stats = jax.devices()[0].memory_stats() or {}
    return stats.get("peak_bytes_in_use", 0)


def run(A_local, b, declared, mpi):
    """(setup seconds, median solve seconds, iterations, peak bytes)."""
    # A fresh matrix object per case: with_cache attaches to the object.
    A_local = jax.experimental.sparse.BCSR(
        (A_local.data, A_local.indices, A_local.indptr), shape=A_local.shape
    )
    A_local = jaxamg.with_cache(A_local, is_symmetric=True, mpi=mpi, **declared)
    solve = jax.jit(
        lambda b_: jaxamg.solve(A_local, b_, **({} if mpi else {"config": cfg}))
    )
    start = time.perf_counter()
    x, info = solve(b)
    x.block_until_ready()
    setup = time.perf_counter() - start
    times = []
    for i in range(repeats):
        rhs = b * (1.0 + 0.01 * i)
        start = time.perf_counter()
        solve(rhs)[0].block_until_ready()
        times.append(time.perf_counter() - start)
    return setup, statistics.median(times), int(info["iterations"]), peak_bytes()


def main():
    comm = None
    if "OMPI_COMM_WORLD_SIZE" in os.environ:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    rank, nranks = (0, 1) if comm is None else (comm.Get_rank(), comm.Get_size())
    tiles, form = os.environ.get("CASE", "16:labels").split(":")
    tiles = int(tiles)
    A, labels = tiled_poisson(side, tiles)
    n, count = A.shape[0], tiles * tiles
    lo, hi = rank * n // nranks, (rank + 1) * n // nranks
    A_local = A[lo:hi]
    rows = labels[lo:hi]
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    b -= (
        np.bincount(labels, b, count)[labels]
        / np.bincount(labels, minlength=count)[labels]
    )
    b = jnp.asarray(b[lo:hi])
    if comm is None:
        A_jax = jax.experimental.sparse.BCSR.from_scipy_sparse(A_local)
        mpi = None
    else:
        A_jax = jax.experimental.sparse.BCSR(
            (
                jnp.asarray(A_local.data),
                jnp.asarray(A_local.indices, jnp.int64),
                jnp.asarray(A_local.indptr, jnp.int64),
            ),
            shape=A_local.shape,
        )
        mpi = jaxamg.cache_mpi_metadata(cfg, comm, n, (lo, hi), A_jax, singular=True)
    if form == "labels":
        declared = dict(nullspace="constant", labels=(count, rows))
    elif form == "constant":
        declared = dict(nullspace="constant")
    else:
        if (hi - lo) * count * 8 > dense_limit:
            if rank == 0:
                print(
                    f"{count:6d} components  dense columns  not run: "
                    f"{(hi - lo) * count * 8 / 2**30:.0f} GiB of columns per rank"
                )
            return
        dense = np.zeros((hi - lo, count))
        dense[np.arange(hi - lo), rows] = 1.0
        declared = dict(nullspace=jnp.asarray(dense))
    setup, solve, iterations, peak = run(A_jax, b, declared, mpi)
    if comm is not None:
        setup, solve = comm.allreduce(setup, op=MPI.MAX), comm.allreduce(
            solve, op=MPI.MAX
        )
        peak = comm.allreduce(peak, op=MPI.MAX)
    if rank == 0:
        print(
            f"{count:6d} components  {form:9s} setup {setup:7.2f} s  solve {1e3 * solve:8.2f} ms  "
            f"{iterations:3d} iterations  peak {peak / 2**20:9.1f} MiB",
            flush=True,
        )
        if len(sys.argv) > 1:
            with open(sys.argv[1], "a") as f:
                f.write(
                    json.dumps(
                        dict(
                            side=side,
                            ranks=nranks,
                            device=jax.devices()[0].device_kind,
                            components=count,
                            form=form,
                            setup_s=setup,
                            solve_s=solve,
                            iterations=iterations,
                            peak_bytes=peak,
                        )
                    )
                    + "\n"
                )


if __name__ == "__main__":
    main()
