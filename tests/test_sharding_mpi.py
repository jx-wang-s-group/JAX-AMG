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
from jaxamg.mpi_utils import get_partition_info  # noqa: E402
from jaxamg.sharding import ShardedSolve  # noqa: E402

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


def _gather_unpadded(
    array: jax.Array, solver: ShardedSolve, comm: MPI.Comm
) -> np.ndarray:
    local = np.asarray(solver.local_vector(array))
    return np.concatenate(comm.allgather(local))


def _local_status(info: dict[str, jax.Array]) -> int:
    status = np.asarray(info["status"].addressable_shards[0].data)
    return int(status.item())


def _distributed_block_system(
    n_blocks: int,
    rank: int,
    nranks: int,
    *,
    symmetric: bool,
) -> tuple[jsp.BCSR, np.ndarray, int, int]:
    """Build a block-aligned local partition and its small dense reference."""
    import scipy.sparse

    block_dim = 2
    lower = -np.ones(n_blocks - 1, dtype=np.float32)
    upper = lower if symmetric else -1.25 * np.ones_like(lower)
    node_matrix = scipy.sparse.diags(
        (lower, 4.0 * np.ones(n_blocks, dtype=np.float32), upper),
        offsets=(-1, 0, 1),
        format="csr",
    )
    coupling = np.array(
        [[2.0, 0.25], [0.25 if symmetric else 0.5, 1.5]], dtype=np.float32
    )
    A_global = scipy.sparse.kron(node_matrix, coupling, format="csr")

    block_start, block_end, _ = get_partition_info(n_blocks, rank, nranks)
    row_start = block_start * block_dim
    row_end = block_end * block_dim
    A_partition = A_global[row_start:row_end]
    A_local = jsp.BCSR(
        (
            jnp.asarray(A_partition.data, dtype=jnp.float32),
            jnp.asarray(A_partition.indices, dtype=jnp.int32),
            jnp.asarray(A_partition.indptr, dtype=jnp.int32),
        ),
        shape=A_partition.shape,
    )
    return A_local, A_global.toarray(), row_start, row_end


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

    def cached_loss(rhs):
        x, _ = solver(rhs)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(
            lambda matrix_data, rhs: solver(rhs, A_data=matrix_data)
        )
        compiled_cached_solve = jax.jit(solver)
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
        compiled_cached_grad = jax.jit(jax.grad(cached_loss))
        x, info = compiled_solve(A_data, b)
        x_cached, _ = compiled_cached_solve(b)
        grad_A_data, grad_b = compiled_grad(A_data, b)
        grad_b_cached = compiled_cached_grad(b)
    x.block_until_ready()
    x_cached.block_until_ready()
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()
    grad_b_cached.block_until_ready()

    x_global = _gather_global(x, comm)
    grad_b_global = _gather_global(grad_b, comm)
    A_global = 4.1 * np.eye(n_global, dtype=np.float64)
    A_global[1:, 0] = -0.15
    b_global = np.arange(1, n_global + 1, dtype=np.float64)
    x_ref = np.linalg.solve(A_global, b_global)
    adjoint_ref = np.linalg.solve(A_global.T, 2.0 * x_ref)

    np.testing.assert_allclose(x_global, x_ref, rtol=1e-5, atol=1e-6)
    A_cached_global = 4.0 * np.eye(n_global, dtype=np.float64)
    A_cached_global[1:, 0] = -0.25
    x_cached_ref = np.linalg.solve(A_cached_global, b_global)
    adjoint_cached_ref = np.linalg.solve(A_cached_global.T, 2.0 * x_cached_ref)
    np.testing.assert_allclose(
        _gather_global(x_cached, comm), x_cached_ref, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        _gather_global(grad_b_cached, comm),
        adjoint_cached_ref,
        rtol=1e-5,
        atol=1e-6,
    )
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

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(
            lambda matrix_data, rhs, guess: solver(rhs, guess, A_data=matrix_data)
        )
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        x, info = compiled_solve(solver.A_data, b, x0)
        grad_A_data, grad_b, grad_x0 = compiled_grad(solver.A_data, b, x0)
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


@pytest.mark.mpi(min_size=3)
def test_sharded_uneven_row_partitions(sharding_context):
    """Exercise padded vectors for a global size not divisible by rank count."""
    comm, rank, nranks, mesh = sharding_context
    n_global = 4 * nranks + 1
    A_local, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0, dtype=jnp.float32
    )
    n_local = row_end - row_start
    b_local = np.arange(row_start + 1, row_end + 1, dtype=np.float32)
    x0_local = np.full(n_local, 0.25, dtype=np.float32)
    b = jaxamg.make_sharded_vector(b_local, mesh=mesh, global_size=n_global)
    x0 = jaxamg.make_sharded_vector(
        x0_local, comm=comm, mesh=mesh, global_size=n_global
    )
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
    A_data = solver.A_data + jnp.asarray(0.05, solver.A_data.dtype)

    def loss(matrix_data, rhs, guess):
        x, _ = solver(rhs, guess, A_data=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(
            lambda matrix_data, rhs, guess: solver(rhs, guess, A_data=matrix_data)
        )
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        x, info = compiled_solve(A_data, b, x0)
        grad_A_data, grad_b, grad_x0 = compiled_grad(A_data, b, x0)
    x.block_until_ready()
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()
    grad_x0.block_until_ready()

    x_global = _gather_unpadded(x, solver, comm)
    grad_b_global = _gather_unpadded(grad_b, solver, comm)
    grad_x0_global = _gather_unpadded(grad_x0, solver, comm)
    A_global = 4.05 * np.eye(n_global, dtype=np.float64)
    A_global += np.diag(-0.95 * np.ones(n_global - 1), 1)
    A_global += np.diag(-0.95 * np.ones(n_global - 1), -1)
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

    x_physical = np.asarray(x.addressable_shards[0].data)
    grad_b_physical = np.asarray(grad_b.addressable_shards[0].data)
    np.testing.assert_array_equal(x_physical[n_local:], 0)
    np.testing.assert_array_equal(grad_b_physical[n_local:], 0)
    assert solver.global_size == n_global
    assert solver.local_size == n_local
    assert _local_status(info) == 0


@pytest.mark.parametrize("is_symmetric", [True, False])
def test_sharded_block_matrix_gradients(sharding_context, is_symmetric):
    """Cover block solves and VJPs with uneven block-aligned partitions."""
    comm, rank, nranks, mesh = sharding_context
    block_dim = 2
    n_blocks = 2 * nranks + 1
    n_global = block_dim * n_blocks
    A_local, A_global, row_start, row_end = _distributed_block_system(
        n_blocks, rank, nranks, symmetric=is_symmetric
    )
    n_local = row_end - row_start
    b_local = np.linspace(row_start + 1.0, row_end, n_local, dtype=np.float32)
    b = jaxamg.make_sharded_vector(b_local, mesh=mesh, global_size=n_global)
    solver = jaxamg.make_sharded_solver(
        A_local,
        b,
        comm=comm,
        mesh=mesh,
        is_symmetric=is_symmetric,
        block_dim=block_dim,
        config={
            "solver": "FGMRES",
            "preconditioner": {"solver": "BLOCK_JACOBI"},
            "communicator": "MPI_DIRECT",
            "max_iters": 200,
            "tolerance": 1e-8,
        },
    )

    def loss(matrix_data, rhs):
        x, _ = solver(rhs, A_data=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(
            lambda matrix_data, rhs: solver(rhs, A_data=matrix_data)
        )
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
        x, info = compiled_solve(solver.A_data, b)
        grad_A_data, grad_b = compiled_grad(solver.A_data, b)
    x.block_until_ready()
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()

    b_global = np.arange(1, n_global + 1, dtype=np.float64)
    A_reference = np.asarray(A_global, dtype=np.float64)
    x_reference = np.linalg.solve(A_reference, b_global)
    adjoint_reference = np.linalg.solve(A_reference.T, 2.0 * x_reference)
    np.testing.assert_allclose(
        _gather_unpadded(x, solver, comm), x_reference, rtol=1e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        _gather_unpadded(grad_b, solver, comm),
        adjoint_reference,
        rtol=1e-5,
        atol=1e-6,
    )

    grad_A_local = solver.local_matrix_gradient(grad_A_data)
    local_rows = np.repeat(
        np.arange(row_start, row_end), np.diff(np.asarray(A_local.indptr))
    )
    grad_A_reference = (
        -adjoint_reference[local_rows]
        * x_reference[np.asarray(A_local.indices, dtype=np.int64)]
    )
    np.testing.assert_allclose(
        np.asarray(grad_A_local.data),
        grad_A_reference,
        rtol=1e-5,
        atol=1e-6,
    )

    x_physical = np.asarray(x.addressable_shards[0].data)
    grad_b_physical = np.asarray(grad_b.addressable_shards[0].data)
    grad_A_physical = np.asarray(grad_A_data.addressable_shards[0].data)
    np.testing.assert_array_equal(x_physical[n_local:], 0)
    np.testing.assert_array_equal(grad_b_physical[n_local:], 0)
    np.testing.assert_array_equal(grad_A_physical[len(A_local.data) :], 0)
    assert n_local % block_dim == 0
    assert solver.global_size == n_global
    assert solver.local_size == n_local
    assert _local_status(info) == 0


def test_sharded_batched_rhs_gradients(sharding_context):
    """Solve multiple RHS columns and accumulate their shared matrix VJP."""
    comm, rank, nranks, mesh = sharding_context
    nrhs = 3
    n_global = 4 * nranks + 1
    template, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0, dtype=jnp.float32
    )
    n_local = row_end - row_start
    local_rows = np.repeat(
        np.arange(row_start, row_end), np.diff(np.asarray(template.indptr))
    )
    local_columns = np.asarray(template.indices)
    local_values = np.where(
        local_columns < local_rows,
        -0.75,
        np.where(local_columns > local_rows, -1.25, 4.0),
    ).astype(np.float32)
    A_local = jsp.BCSR(
        (jnp.asarray(local_values), template.indices, template.indptr),
        shape=template.shape,
    )

    local_indices = np.arange(row_start, row_end, dtype=np.float32)
    b_local = np.stack(
        (
            local_indices + 1.0,
            1.0 + 0.1 * local_indices,
            2.0 - 0.05 * local_indices,
        ),
        axis=1,
    )
    x0_local = np.full((n_local, nrhs), 0.1, dtype=np.float32)
    b = jaxamg.make_sharded_vector(b_local, comm=comm, mesh=mesh, global_size=n_global)
    x0 = jaxamg.make_sharded_vector(
        x0_local, comm=comm, mesh=mesh, global_size=n_global
    )
    solver = jaxamg.make_sharded_solver(
        A_local,
        b,
        comm=comm,
        mesh=mesh,
        config={
            "solver": "FGMRES",
            "preconditioner": {"solver": "JACOBI_L1"},
            "communicator": "MPI_DIRECT",
            "max_iters": 100,
            "tolerance": 1e-8,
        },
    )
    A_data = solver.A_data + jnp.asarray(0.05, solver.A_data.dtype)

    def loss(matrix_data, rhs, guess):
        x, _ = solver(rhs, guess, A_data=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(
            lambda matrix_data, rhs, guess: solver(rhs, guess, A_data=matrix_data)
        )
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        x, info = compiled_solve(A_data, b, x0)
        grad_A_data, grad_b, grad_x0 = compiled_grad(A_data, b, x0)
    x.block_until_ready()
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()
    grad_x0.block_until_ready()

    A_reference = 4.05 * np.eye(n_global, dtype=np.float64)
    A_reference += np.diag(-1.20 * np.ones(n_global - 1), 1)
    A_reference += np.diag(-0.70 * np.ones(n_global - 1), -1)
    global_indices = np.arange(n_global, dtype=np.float64)
    b_reference = np.stack(
        (
            global_indices + 1.0,
            1.0 + 0.1 * global_indices,
            2.0 - 0.05 * global_indices,
        ),
        axis=1,
    )
    x_reference = np.linalg.solve(A_reference, b_reference)
    adjoint_reference = np.linalg.solve(A_reference.T, 2.0 * x_reference)
    x_global = _gather_unpadded(x, solver, comm)
    np.testing.assert_allclose(x_global, x_reference, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        _gather_unpadded(grad_b, solver, comm),
        adjoint_reference,
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        _gather_unpadded(grad_x0, solver, comm), np.zeros_like(b_reference)
    )

    grad_A_local = solver.local_matrix_gradient(grad_A_data)
    grad_A_reference = -np.sum(
        adjoint_reference[local_rows]
        * x_reference[np.asarray(A_local.indices, dtype=np.int64)],
        axis=1,
    )
    np.testing.assert_allclose(
        np.asarray(grad_A_local.data),
        grad_A_reference,
        rtol=1e-5,
        atol=1e-6,
    )

    x_physical = np.asarray(x.addressable_shards[0].data)
    grad_b_physical = np.asarray(grad_b.addressable_shards[0].data)
    grad_A_physical = np.asarray(grad_A_data.addressable_shards[0].data)
    np.testing.assert_array_equal(x_physical[n_local:], 0)
    np.testing.assert_array_equal(grad_b_physical[n_local:], 0)
    np.testing.assert_array_equal(grad_A_physical[len(A_local.data) :], 0)
    assert info["iterations"].shape == (nranks, nrhs)
    assert info["residual_history"].shape[:2] == (nranks, nrhs)
    local_status = np.asarray(info["status"].addressable_shards[0].data)
    np.testing.assert_array_equal(local_status, 0)
