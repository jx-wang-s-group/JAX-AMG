"""
Demo: MPI-distributed version of singular_poisson_stretched_grid.py.

Usage:
    mpirun -n 2 python demo/mpi_singular_poisson_stretched_grid.py
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

import jaxamg
from jaxamg import NullSpaceWarning
from jaxamg.matrices import poisson_matrix_stretched
from jaxamg.mpi_utils import gather_vector, partition_csr_matrix
from jaxamg.utils import to_scipy

jax.config.update("jax_enable_x64", True)

nx, ny = 64, 48
stretches = [1.00, 1.02, 1.04, 1.08, 1.10]
cfg = {"tolerance": 1e-10, "max_iters": 200}


def run_case(stretch, comm):
    rank, nranks = comm.Get_rank(), comm.Get_size()
    A_global, V = poisson_matrix_stretched(nx, ny, stretch, dtype=jnp.float64)
    n = A_global.shape[0]
    V_np = np.asarray(V)
    rng = np.random.default_rng(0)
    b = rng.standard_normal(n)
    b -= (V_np @ b) / V_np.sum()  # compatible RHS
    w = rng.standard_normal(n) + 0.5

    A_sp = to_scipy(A_global)
    A_local, row_start, row_end = partition_csr_matrix(A_global, rank, nranks)
    A_T_local, _, _ = partition_csr_matrix(A_sp.T.tocsr(), rank, nranks)
    rows = slice(row_start, row_end)
    V_local, b_local, w_local = (jnp.asarray(v[rows]) for v in (V_np, b, w))
    mpi = {"comm": comm, "nglobal": n, "partition_info": (row_start, row_end)}

    if rank == 0:
        pinv = np.linalg.pinv(A_sp.toarray())
        x_ref, g_ref = pinv @ b, pinv.T @ w

    def err(v_local, ref, null):
        # plain solves are compared up to the null vector (1 forward, V adjoint)
        v = gather_vector(v_local, comm)
        if rank != 0:
            return None
        v = np.asarray(v)
        if null is not None:
            v = v - null * ((v - ref) @ null) / (null @ null)
        return np.linalg.norm(v - ref) / np.linalg.norm(ref)

    out = []
    ones = np.ones(n)
    for kwargs, kwargs_T, null_x, null_g in [
        ({}, {}, ones, V_np),
        (
            {"nullspace": "constant", "transpose_nullspace": V_local},
            {"nullspace": V_local, "transpose_nullspace": "constant"},
            None,
            None,
        ),
    ]:
        x, info = jaxamg.solve(A_local, b_local, **kwargs, **mpi, **cfg)
        # adjoint system Aᵀλ = g solved by jax.grad (explicitly, to read its iteration count)
        _, info_adj = jaxamg.solve(A_T_local, w_local, **kwargs_T, **mpi, **cfg)
        # local loss w_local·x_local: summed over ranks it is w·x, so each
        # rank's gradient is its slice of the global one
        g = jax.grad(
            lambda b_: jnp.dot(
                w_local, jaxamg.solve(A_local, b_, **kwargs, **mpi, **cfg)[0]
            )
        )(b_local)
        ex = err(x, x_ref if rank == 0 else None, null_x)
        eg = err(g, g_ref if rank == 0 else None, null_g)
        if rank == 0:
            out.append(
                f"{info['iterations']:4d} {ex:8.1e}   {info_adj['iterations']:4d} {eg:8.1e}"
            )
    return out


def main():
    warnings.simplefilter("ignore", NullSpaceWarning)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    devices = jax.devices()
    jax.config.update("jax_default_device", devices[rank % len(devices)])
    if rank == 0:
        print(
            f"{nx}x{ny} cells, {comm.Get_size()} ranks, float64, {cfg} (iterations and relative error)\n"
        )
        print(
            f"{'':8s}{'------- plain solve -------':^30s}   {'----- with nullspace -----':^30s}"
        )
        print(
            f"{'':8s}{'forward':^15s}{'adjoint':^15s}   {'forward':^15s}{'adjoint':^15s}"
        )
        print(
            f"{'stretch':8s}{'iters':>5s}{'error':>9s}   {'iters':>5s}{'error':>9s}   "
            f"{'iters':>5s}{'error':>9s}   {'iters':>5s}{'error':>9s}"
        )
    for stretch in stretches:
        rows = run_case(stretch, comm)
        if rank == 0:
            print(f"{stretch:<8.2f}{rows[0]}   {rows[1]}")
    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
