"""Multi-GPU integration tests for the additive JAX sharding interface.

Run this module separately from the ordinary MPI suite because JAX distributed
must see every participating GPU in every process::

    CUDA_VISIBLE_DEVICES=0,1 \
      mpirun -n 2 python -m pytest --only-mpi tests/test_sharding_mpi.py
"""

from __future__ import annotations

import sys

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

_SHARDING_TEST = any("test_sharding_mpi.py" in arg for arg in sys.argv[1:])
if _SHARDING_TEST:
    from mpi4py import MPI

    _SHARDING_TEST = MPI.COMM_WORLD.Get_size() > 1
    if _SHARDING_TEST:
        # This must precede calls that can initialize the XLA backend.
        jax.distributed.initialize(cluster_detection_method="mpi4py")

import jaxamg  # noqa: E402
from jaxamg.matrices import tridiagonal_matrix_distributed  # noqa: E402

pytestmark = [
    pytest.mark.mpi(min_size=2),
    pytest.mark.sharding,
    pytest.mark.skipif(
        not _SHARDING_TEST,
        reason="run this module separately under mpirun with at least two ranks",
    ),
]


@pytest.fixture(scope="module")
def sharding_context():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    mesh = jax.make_mesh((nranks,), ("rank",))

    yield comm, rank, nranks, mesh

    comm.Barrier()
    jaxamg.finalize()
    comm.Barrier()
    jax.config.update("jax_logging_level", "ERROR")
    jax.distributed.shutdown()


def _global_vector(
    local_values: np.ndarray, global_size: int, mesh: jax.sharding.Mesh
) -> jax.Array:
    sharding = jax.NamedSharding(mesh, jax.P("rank"))
    return jax.make_array_from_process_local_data(
        sharding, local_values, global_shape=(global_size,)
    )


def _gather_global(array: jax.Array, comm: MPI.Comm) -> np.ndarray:
    local = np.asarray(array.addressable_shards[0].data)
    return np.concatenate(comm.allgather(local))


def _local_status(info: dict[str, jax.Array]) -> int:
    status = np.asarray(info["status"].addressable_shards[0].data)
    return int(status.item())


def test_sharded_nonsymmetric_matrix_and_rhs_gradients(sharding_context):
    """Exercise dynamic values and unequal local nnz across A and A transpose."""
    comm, rank, nranks, mesh = sharding_context
    n_local = 4
    n_global = n_local * nranks
    row_start = rank * n_local

    # Structurally nonsymmetric: diagonal plus a dense first column. Rank zero
    # has one fewer entry, exercising packed-value padding as well.
    data: list[float] = []
    indices: list[int] = []
    indptr = [0]
    for global_row in range(row_start, row_start + n_local):
        if global_row:
            data.append(-0.25)
            indices.append(0)
        data.append(4.0)
        indices.append(global_row)
        indptr.append(len(data))

    A_local = jsp.BCSR(
        (
            jnp.asarray(data, dtype=jnp.float32),
            jnp.asarray(indices, dtype=jnp.int32),
            jnp.asarray(indptr, dtype=jnp.int32),
        ),
        shape=(n_local, n_global),
    )
    b_local = np.arange(row_start + 1, row_start + n_local + 1, dtype=np.float32)
    b = _global_vector(b_local, n_global, mesh)
    solver = jaxamg.make_sharded_solver(
        A_local,
        b,
        comm=comm,
        mesh=mesh,
        config={
            "solver": "GMRES",
            "preconditioner": {"solver": "JACOBI_L1"},
            "communicator": "MPI_DIRECT",
            "max_iters": 100,
            "tolerance": 1e-8,
        },
    )

    # Change every real matrix value after setup. Padding is also changed but
    # must remain disconnected from both the solve and its gradient.
    A_data = solver.A_data + jnp.asarray(0.1, dtype=solver.A_data.dtype)

    def loss(matrix_data, rhs):
        x, _ = solver(rhs, A_data=matrix_data)
        return jnp.sum(x**2)

    x, info = solver(b, A_data=A_data)
    with jax.set_mesh(mesh):
        grad_A_data, grad_b = jax.grad(loss, argnums=(0, 1))(A_data, b)
    x.block_until_ready()
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()

    x_global = _gather_global(x, comm)
    grad_b_global = _gather_global(grad_b, comm)
    A_global = 4.1 * np.eye(n_global, dtype=np.float64)
    A_global[1:, 0] = -0.15
    b_global = np.arange(1, n_global + 1, dtype=np.float64)
    x_ref = np.linalg.solve(A_global, b_global)
    adjoint_ref = np.linalg.solve(A_global.T, 2.0 * x_ref)

    np.testing.assert_allclose(x_global, x_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_b_global, adjoint_ref, rtol=1e-5, atol=1e-6)

    grad_A_local = solver.local_matrix_gradient(grad_A_data)
    local_rows = np.repeat(
        np.arange(row_start, row_start + n_local), np.diff(np.asarray(indptr))
    )
    grad_A_ref = -adjoint_ref[local_rows] * x_ref[np.asarray(indices, dtype=np.int64)]
    np.testing.assert_allclose(
        np.asarray(grad_A_local.data), grad_A_ref, rtol=1e-5, atol=1e-6
    )

    packed_gradient = np.asarray(grad_A_data.addressable_shards[0].data)
    np.testing.assert_array_equal(packed_gradient[len(data) :], 0)
    assert _local_status(info) == 0


def test_sharded_symmetric_warm_start_gradients(sharding_context):
    """Cover the symmetric optimization and zero x0 cotangent."""
    comm, rank, nranks, mesh = sharding_context
    n_local = 4
    n_global = n_local * nranks
    A_local, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0, dtype=jnp.float32
    )
    b_local = np.linspace(row_start + 1.0, row_end, n_local, dtype=np.float32)
    x0_local = np.full(n_local, 0.25, dtype=np.float32)
    b = _global_vector(b_local, n_global, mesh)
    x0 = _global_vector(x0_local, n_global, mesh)
    solver = jaxamg.make_sharded_solver(
        A_local,
        b,
        comm=comm,
        mesh=mesh,
        is_symmetric=True,
        config={
            "solver": "CG",
            "preconditioner": {"solver": "JACOBI_L1"},
            "communicator": "MPI_DIRECT",
            "max_iters": 100,
            "tolerance": 1e-8,
        },
    )

    def loss(matrix_data, rhs, guess):
        x, _ = solver(rhs, guess, A_data=matrix_data)
        return jnp.sum(x**2)

    x, info = solver(b, x0, A_data=solver.A_data)
    with jax.set_mesh(mesh):
        grad_A_data, grad_b, grad_x0 = jax.grad(loss, argnums=(0, 1, 2))(
            solver.A_data, b, x0
        )
    x.block_until_ready()
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()
    grad_x0.block_until_ready()

    x_global = _gather_global(x, comm)
    grad_b_global = _gather_global(grad_b, comm)
    grad_x0_global = _gather_global(grad_x0, comm)
    A_global = 4.0 * np.eye(n_global, dtype=np.float64)
    A_global += np.diag(-np.ones(n_global - 1), 1)
    A_global += np.diag(-np.ones(n_global - 1), -1)
    b_global = np.arange(1, n_global + 1, dtype=np.float64)
    x_ref = np.linalg.solve(A_global, b_global)
    adjoint_ref = np.linalg.solve(A_global.T, 2.0 * x_ref)

    np.testing.assert_allclose(x_global, x_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(grad_b_global, adjoint_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(grad_x0_global, 0)

    grad_A_local = solver.local_matrix_gradient(grad_A_data)
    row_indices = np.repeat(
        np.arange(row_start, row_end), np.diff(np.asarray(A_local.indptr))
    )
    grad_A_ref = (
        -adjoint_ref[row_indices] * x_ref[np.asarray(A_local.indices, dtype=np.int64)]
    )
    np.testing.assert_allclose(
        np.asarray(grad_A_local.data), grad_A_ref, rtol=1e-5, atol=1e-6
    )
    assert _local_status(info) == 0
