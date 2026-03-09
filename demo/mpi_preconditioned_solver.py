"""Demo: use `jaxamg.make_preconditioner(...)` with native JAX `bicgstab` in MPI mode.

This example solves a distributed non-symmetric Poisson problem by combining
a JAX-style BiCGSTAB outer iteration with an MPI-enabled
`jaxamg.make_preconditioner(...)`, then validates the MPI result against a
single-GPU reference solve.

Usage:
    mpirun -n 2 python demo/mpi_preconditioned_solver.py
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.sparse.linalg import bicgstab
from mpi4py import MPI

import jaxamg
from jaxamg.matrices import (
    poisson_matrix,
    poisson_matrix_distributed,
    rhs_ones,
)
from jaxamg.mpi_utils import _mpi4jax_allgatherv, partition_vector


def mpi_dot(x_local, y_local, comm):
    local_value = np.vdot(np.asarray(x_local), np.asarray(y_local))
    global_value = comm.allreduce(local_value, op=MPI.SUM)
    return jnp.asarray(global_value, dtype=jnp.result_type(x_local, y_local))


def mpi_norm2(x_local, comm):
    return jnp.real(mpi_dot(x_local, x_local, comm))


def mpi_bicgstab(
    A_local,
    b_local,
    preconditioner_local,
    comm,
    recvcounts_tuple,
    *,
    tol,
    maxiter,
):
    def matvec_local(x_local):
        x_global = _mpi4jax_allgatherv(x_local, recvcounts_tuple, comm)
        return A_local @ x_global

    dot = lambda x_local, y_local: mpi_dot(x_local, y_local, comm)
    norm2 = lambda x_local: mpi_norm2(x_local, comm)

    x_local = jnp.zeros_like(b_local)
    r_local = b_local - matvec_local(x_local)
    rhat_local = r_local
    p_local = r_local
    v_local = jnp.zeros_like(r_local)

    one = jnp.asarray(1.0, dtype=b_local.dtype)
    rho = one
    alpha = one
    omega = one

    tol2 = (tol**2) * norm2(b_local)

    for iteration in range(maxiter):
        if float(norm2(r_local)) <= float(tol2):
            return x_local

        rho_next = dot(rhat_local, r_local)
        if float(jnp.abs(rho_next)) == 0.0:
            return x_local

        if iteration > 0:
            beta = (rho_next / rho) * (alpha / omega)
            p_local = r_local + beta * (p_local - omega * v_local)

        phat_local = preconditioner_local(p_local)
        v_local = matvec_local(phat_local)

        denom_alpha = dot(rhat_local, v_local)
        if float(jnp.abs(denom_alpha)) == 0.0:
            return x_local

        alpha = rho_next / denom_alpha
        s_local = r_local - alpha * v_local

        if float(norm2(s_local)) <= float(tol2):
            x_local = x_local + alpha * phat_local
            return x_local

        shat_local = preconditioner_local(s_local)
        t_local = matvec_local(shat_local)

        denom_omega = dot(t_local, t_local)
        if float(jnp.abs(denom_omega)) == 0.0:
            return x_local

        omega = dot(t_local, s_local) / denom_omega
        if float(jnp.abs(omega)) == 0.0:
            return x_local

        x_local = x_local + alpha * phat_local + omega * shat_local
        r_local = s_local - omega * t_local

        rho = rho_next

    return x_local


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    grid_size = 8
    skew = 2.0
    n = grid_size**2
    tol = 1e-6
    maxiter = 100

    if rank == 0:
        print(f"MPI ranks: {nranks}")
        print(
            f"Setting up distributed non-symmetric Poisson system ({grid_size}x{grid_size}, skew={skew:g})..."
        )

    A_global = poisson_matrix(grid_size, skew=skew)
    b_global = rhs_ones(n)

    # Distribute the matrix and vector across MPI processes
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks, skew=skew
    )
    b_local, _, _ = partition_vector(b_global, rank, nranks)
    recvcounts_tuple = tuple(comm.allgather(int(b_local.shape[0])))
    partition_info = (row_start, row_end)

    # Create the AMG preconditioner for the local matrix
    preconditioner_local = jaxamg.make_preconditioner(
        A_local,
        comm=comm,
        nglobal=n,
        partition_info=partition_info,
    )

    # Run distributed BiCGSTAB with the AMG preconditioner
    x_local = mpi_bicgstab(
        A_local,
        b_local,
        preconditioner_local,
        comm,
        recvcounts_tuple,
        tol=tol,
        maxiter=maxiter,
    )
    x_local = jax.block_until_ready(x_local)

    # Gather the local solution parts to form the global solution vector
    x_global_parts = comm.allgather(np.asarray(x_local))
    x_mpi = jnp.concatenate([jnp.asarray(part) for part in x_global_parts])
    x_mpi_global = _mpi4jax_allgatherv(x_local, recvcounts_tuple, comm)

    if rank == 0:
        # Run a single-GPU reference solve using the same AMG preconditioner
        M_ref = jaxamg.make_preconditioner(A_global)
        x_ref, info_ref = bicgstab(
            A_global,
            b_global,
            M=M_ref,
            tol=tol,
            maxiter=maxiter,
        )
        x_ref = jax.block_until_ready(x_ref)

        # Compute residuals and relative solution difference
        residual_mpi = jnp.linalg.norm(b_global - A_global @ x_mpi) / jnp.linalg.norm(
            b_global
        )
        residual_ref = jnp.linalg.norm(b_global - A_global @ x_ref) / jnp.linalg.norm(
            b_global
        )
        rel_diff = jnp.linalg.norm(x_mpi - x_ref) / jnp.linalg.norm(x_ref)

        print(f"MPI mode residual={residual_mpi:.3e}")
        print(f"Single-GPU mode residual={residual_ref:.3e}")
        print(f"Relative solution difference={rel_diff:.3e}")

    comm.Barrier()
    jaxamg.finalize()


if __name__ == "__main__":
    main()
