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
from jaxamg.matrices import (  # noqa: E402
    poisson_operator,
    rhs_ones,
    tridiagonal_matrix_distributed,
)
from jaxamg.mpi_utils import get_partition_info, partition_operator  # noqa: E402
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
    b_local_device = jnp.asarray(b_local)
    with jax.transfer_guard_device_to_host("disallow"):
        b = jaxamg.make_sharded_vector(b_local_device, mesh=mesh, global_size=n_global)
        matrix = jaxamg.make_sharded_matrix(A_local, b)
    # Rank-local transpose and halo constants must remain local even when the
    # surrounding application keeps an explicit global mesh active.
    with jax.set_mesh(mesh):
        solver = jaxamg.make_sharded_solver(
            matrix,
            b,
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
    A_data = matrix.data + jnp.asarray(0.1, dtype=matrix.data.dtype)

    def loss(matrix_data, rhs):
        x, _ = solver(rhs, A=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(lambda matrix_data, rhs: solver(rhs, A=matrix_data))
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
        compiled_cached_grad = jax.jit(jax.grad(loss, argnums=1))
        x, info = compiled_solve(A_data, b)
        grad_A_data, grad_b = compiled_grad(A_data, b)
        grad_b_cached = compiled_cached_grad(matrix.data, b)
    # The cached-value convenience remains efficient for a direct call; only
    # enclosing JAX transforms require an explicit matrix-data operand.
    x_cached, _ = solver(b)
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

    grad_A_local = matrix.local_matrix(grad_A_data)
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
    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)
    solver = jaxamg.make_sharded_solver(
        matrix,
        b,
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
        x, _ = solver(rhs, guess, A=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(
            lambda matrix_data, rhs, guess: solver(rhs, guess, A=matrix_data)
        )
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1, 2)))
        compiled_vmap = jax.jit(
            lambda matrix_data, rhs_batch: jax.vmap(
                lambda rhs: solver(rhs, A=matrix_data)[0]
            )(rhs_batch)
        )
        x, info = compiled_solve(matrix.data, b, x0)
        grad_A_data, grad_b, grad_x0 = compiled_grad(matrix.data, b, x0)
        x_batched = compiled_vmap(matrix.data, jnp.stack((b, 2 * b)))
    x.block_until_ready()
    grad_A_data.block_until_ready()
    grad_b.block_until_ready()
    grad_x0.block_until_ready()
    x_batched.block_until_ready()

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
    x_batched_global = np.concatenate(
        comm.allgather(np.asarray(x_batched.addressable_shards[0].data)), axis=1
    )
    np.testing.assert_allclose(
        x_batched_global,
        np.stack((x_ref, 2 * x_ref)),
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(grad_b_global, adjoint_ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(grad_x0_global, 0)

    grad_A_local = matrix.local_matrix(grad_A_data)
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


@pytest.mark.parametrize("layout", ["axis", "tuple", "grid"])
def test_sharded_multi_axis_program_mesh(sharding_context, layout):
    """Solve and differentiate on an application's multi-axis mesh, in one
    program with that mesh's own sharding constraints (no second mesh)."""
    comm, rank, nranks, _ = sharding_context
    if layout == "grid":
        if nranks % 2 or nranks < 4:
            pytest.skip(
                "a two-dimensional rank grid needs an even count of at least four"
            )
        mesh = jax.make_mesh((2, nranks // 2, 1), ("x", "y", "z"))
        axes = ("x", "y")
    else:
        mesh = jax.make_mesh((nranks, 1), ("x", "y"))
        axes = "x" if layout == "axis" else ("x", "y")
    n_local = 4
    n_global = n_local * nranks
    A_local, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0, dtype=jnp.float32
    )
    b_local = np.linspace(row_start + 1.0, row_end, n_local, dtype=np.float32)
    b = jaxamg.make_sharded_vector(b_local, comm=comm, mesh=mesh, axis_name=axes)
    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, axis_name=axes)
    solver = jaxamg.make_sharded_solver(
        matrix,
        b,
        is_symmetric=True,
        config={
            "solver": "CG",
            "preconditioner": {"solver": "JACOBI_L1"},
            "communicator": "MPI_DIRECT",
            "max_iters": 100,
            "tolerance": 1e-8,
        },
    )
    rows = jax.NamedSharding(mesh, jax.P(axes))

    def loss(matrix_data, rhs):
        # The application's own constraint on the same mesh.
        rhs = jax.lax.with_sharding_constraint(2.0 * rhs, rows)
        x, _ = solver(rhs, A=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        value, (grad_A_data, grad_b) = jax.jit(
            jax.value_and_grad(loss, argnums=(0, 1))
        )(matrix.data, b)
    A_global = (
        4.0 * np.eye(n_global)
        + np.diag(-np.ones(n_global - 1), 1)
        + np.diag(-np.ones(n_global - 1), -1)
    )
    x_ref = np.linalg.solve(A_global, 2.0 * np.arange(1, n_global + 1))
    adjoint_ref = np.linalg.solve(A_global.T, 2.0 * x_ref)
    np.testing.assert_allclose(float(value), np.sum(x_ref**2), rtol=1e-5)
    np.testing.assert_allclose(
        _gather_global(grad_b, comm), 2.0 * adjoint_ref, rtol=1e-5, atol=1e-5
    )
    grad_A_local = matrix.local_matrix(grad_A_data)
    row_indices = np.repeat(
        np.arange(row_start, row_end), np.diff(np.asarray(A_local.indptr))
    )
    grad_A_ref = (
        -adjoint_ref[row_indices] * x_ref[np.asarray(A_local.indices, dtype=np.int64)]
    )
    np.testing.assert_allclose(
        np.asarray(grad_A_local.data), grad_A_ref, rtol=1e-5, atol=1e-5
    )


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
    b_local_device = jnp.asarray(b_local)
    x0_local = np.full(n_local, 0.25, dtype=np.float32)
    with jax.transfer_guard_device_to_host("disallow"):
        b = jaxamg.make_sharded_vector(b_local_device, global_size=n_global)
    x0 = jaxamg.make_sharded_vector(
        x0_local, comm=comm, mesh=mesh, global_size=n_global
    )
    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)
    solver = jaxamg.make_sharded_solver(
        matrix,
        b,
        config={
            "solver": "GMRES",
            "preconditioner": {"solver": "JACOBI_L1"},
            "communicator": "MPI_DIRECT",
            "max_iters": 100,
            "tolerance": 1e-8,
        },
    )
    A_data = matrix.data + jnp.asarray(0.05, matrix.data.dtype)

    def loss(matrix_data, rhs, guess):
        x, _ = solver(rhs, guess, A=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(
            lambda matrix_data, rhs, guess: solver(rhs, guess, A=matrix_data)
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

    grad_A_local = matrix.local_matrix(grad_A_data)
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


def test_sharded_save_stats_file(sharding_context, tmp_path):
    """A direct call with save_stats_file writes formatted stats on rank zero."""
    comm, rank, nranks, mesh = sharding_context
    n_local = 4
    n_global = n_local * nranks
    A_local, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, diagonal_value=4.0, dtype=jnp.float32
    )
    b_local = np.ones(n_local, dtype=np.float32)
    b = _global_vector(b_local, n_global, mesh)
    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)
    solver = jaxamg.make_sharded_solver(
        matrix,
        b,
        is_symmetric=True,
        save_stats=True,
        config={
            "solver": "CG",
            "preconditioner": {"solver": "AMG"},
            "communicator": "MPI_DIRECT",
            "max_iters": 100,
            "tolerance": 1e-8,
        },
    )

    stats_file = tmp_path / "sharded_stats.txt"
    x, info = solver(b, save_stats_file=stats_file)
    assert _local_status(info) == 0
    if rank == 0:
        content = stats_file.read_text()
        assert "SOLVER ITERATIONS" in content
    comm.Barrier()


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
    b = jaxamg.make_sharded_vector(b_local, global_size=n_global)
    matrix = jaxamg.make_sharded_matrix(A_local, b)
    solver = jaxamg.make_sharded_solver(
        matrix,
        b,
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
        x, _ = solver(rhs, A=matrix_data)
        return jnp.sum(x**2)

    with jax.set_mesh(mesh):
        compiled_solve = jax.jit(lambda matrix_data, rhs: solver(rhs, A=matrix_data))
        compiled_grad = jax.jit(jax.grad(loss, argnums=(0, 1)))
        x, info = compiled_solve(matrix.data, b)
        grad_A_data, grad_b = compiled_grad(matrix.data, b)
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

    grad_A_local = matrix.local_matrix(grad_A_data)
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


def test_sharded_operator_parameter_gradients(sharding_context):
    """``solver(rhs, A=operator)``: eager and compiled gradients of a parameter
    the operator closes over match the MPI interface's allreduced gradient."""
    comm, rank, nranks, mesh = sharding_context
    grid = 8
    n_global = grid * grid
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "max_iters": 200,
        "tolerance": 1e-12,
    }

    def local_operator(skew):
        operator, _, _ = partition_operator(
            poisson_operator(skew), n_global, rank, nranks
        )
        return operator

    b_local = rhs_ones(n_local)
    b = jaxamg.make_sharded_vector(b_local, comm=comm, mesh=mesh, global_size=n_global)
    coloring = jaxamg.cache_coloring(local_operator(0.0), shape=(n_local, n_global))
    matrix = jaxamg.make_sharded_matrix(
        jaxamg.with_cache(local_operator(0.0), coloring=coloring),
        b,
        comm=comm,
        mesh=mesh,
    )
    solver = jaxamg.make_sharded_solver(matrix, b, config=config)

    true_skew = 3.0
    with jax.set_mesh(mesh):
        x_target, info = solver(b, A=local_operator(true_skew))
    assert _local_status(info) == 0

    def loss(skew, rhs, target):
        x, _ = solver(rhs, A=local_operator(skew))
        return jnp.sum((x - target) ** 2) / n_global

    # Reference: the MPI interface's rank-local loss, reduced by hand as in
    # the MPI demo.
    mpi_cache = jaxamg.cache_mpi_metadata(
        config,
        comm,
        n_global,
        (row_start, row_end),
        jaxamg.with_cache(local_operator(0.0), coloring=coloring),
    )
    target_local = np.asarray(solver.local_vector(x_target))

    def loss_local(skew):
        operator = jaxamg.with_cache(
            local_operator(skew), coloring=coloring, mpi=mpi_cache
        )
        x, _ = jaxamg.solve(operator, b_local)
        return jnp.sum((x - target_local) ** 2) / n_global

    skew = 1.0
    reference_loss = comm.allreduce(float(loss_local(skew)), op=MPI.SUM)
    reference_grad = comm.allreduce(float(jax.grad(loss_local)(skew)), op=MPI.SUM)
    assert reference_grad != 0.0

    with jax.set_mesh(mesh):
        eager_loss, eager_grad = jax.value_and_grad(loss)(skew, b, x_target)
        compiled_loss, compiled_grad = jax.jit(jax.value_and_grad(loss))(
            skew, b, x_target
        )
    for value, grad in ((eager_loss, eager_grad), (compiled_loss, compiled_grad)):
        assert float(value) == pytest.approx(reference_loss, rel=1e-4)
        assert float(grad) == pytest.approx(reference_grad, rel=1e-4)

    # A value that differs across ranks cannot be closed over (its cotangent
    # would be summed). Detected under jit; a use that leaves the operator's
    # output sharded fails earlier in JAX's sharding checks.
    def sharded_closure(coefficients, rhs):
        base = local_operator(1.0)
        return solver(rhs, A=lambda v: base(v) * jnp.mean(coefficients))[0]

    with jax.set_mesh(mesh):
        with pytest.raises(ValueError, match="identical on every rank"):
            jax.jit(sharded_closure).lower(b, b)


@pytest.mark.parametrize("symmetric", [False, True])
def test_sharded_nullspace(sharding_context, symmetric, monkeypatch):
    """Singular finite-volume Poisson on a stretched grid (nonsymmetric
    volume-normalized form, or the symmetric flux form): eager and compiled
    solves and RHS gradients against the pseudo-inverse."""
    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        from jaxamg.matrices import poisson_matrix_stretched
        from jaxamg.mpi_utils import partition_csr_matrix
        from jaxamg.utils import to_scipy

        A_global, V_global = poisson_matrix_stretched(
            24, 16, 1.08, normalize=not symmetric, dtype=jnp.float64
        )
        n = A_global.shape[0]
        V_np = np.ones(n) if symmetric else np.asarray(V_global)
        rng = np.random.default_rng(0)
        b_global = rng.standard_normal(n)
        b_global -= (V_np @ b_global) / V_np.sum()  # consistent RHS
        w_global = rng.standard_normal(n) + 0.5

        A_local, row_start, row_end = partition_csr_matrix(A_global, rank, nranks)
        rows = slice(row_start, row_end)
        bases = {"nullspace": "constant"}
        if not symmetric:
            bases["transpose_nullspace"] = jnp.asarray(V_np[rows])
        b = jaxamg.make_sharded_vector(b_global[rows], comm=comm, mesh=mesh)
        w = jaxamg.make_sharded_vector(w_global[rows], comm=comm, mesh=mesh)
        A = jaxamg.make_sharded_matrix(
            jaxamg.with_cache(A_local, **bases), b, comm=comm, mesh=mesh
        )
        solver = jaxamg.make_sharded_solver(
            A, b, config={"tolerance": 1e-10, "max_iters": 300}, is_symmetric=symmetric
        )

        def loss(rhs, A_data, weights):
            return jnp.sum(weights * solver(rhs, A=A_data)[0])

        # Bases were validated at creation; tracing must not validate again
        # (that would be a host collective inside the trace).
        import jaxamg.jaxamg as core

        validations = []
        monkeypatch.setattr(
            core, "validate_basis", lambda *args: validations.append(args)
        )
        with jax.set_mesh(mesh):
            jax.jit(loss).lower(b, A.data, w)
            jax.jit(lambda rhs, x0, A_data: solver(rhs, x0, A=A_data)[0]).lower(
                b, b, A.data
            )
            jax.jit(jax.grad(loss)).lower(b, A.data, w)
        monkeypatch.undo()
        assert validations == []

        x, info = solver(b)
        assert _local_status(info) == 0
        inconsistency = np.asarray(info["rhs_inconsistency"].addressable_shards[0].data)
        assert inconsistency.item() < 1e-12

        # Scaling A preserves both null spaces: d(w·x)/ds = -w·x at s = 1.
        def scaled_loss(scale, rhs, A_data, weights):
            return loss(rhs, scale * A_data, weights)

        with jax.set_mesh(mesh):
            g_eager = jax.grad(loss)(b, A.data, w)
            g_jit = jax.jit(jax.grad(loss))(b, A.data, w)
            wx, ds_eager = jax.value_and_grad(scaled_loss)(1.0, b, A.data, w)
            ds_jit = jax.jit(jax.grad(scaled_loss))(1.0, b, A.data, w)
        np.testing.assert_allclose(float(ds_eager), -float(wx), rtol=1e-6)
        np.testing.assert_allclose(float(ds_jit), -float(wx), rtol=1e-6)

        x_np = _gather_unpadded(x, solver, comm)
        g1 = _gather_unpadded(g_eager, solver, comm)
        g2 = _gather_unpadded(g_jit, solver, comm)
        if rank == 0:
            A_dense = to_scipy(A_global).toarray().astype(np.float64)
            assert abs(x_np.mean()) < 1e-10 * np.linalg.norm(x_np)
            assert np.linalg.norm(A_dense @ x_np - b_global) < 1e-7 * np.linalg.norm(
                b_global
            )
            g_ref = np.linalg.pinv(A_dense).T @ w_global
            atol = 1e-8 * np.linalg.norm(g_ref)
            np.testing.assert_allclose(g1, g_ref, rtol=1e-6, atol=atol)
            np.testing.assert_allclose(g2, g_ref, rtol=1e-6, atol=atol)
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("reduction", ["all-reduce", "caller"])
@pytest.mark.parametrize("extreme_scale", [False, True])
def test_sharded_labels_per_solve(sharding_context, reduction, extreme_scale):
    """A solver built without a null space, given one per solve: constants
    per component on three stretched grids (the first spanning both ranks),
    one component anchored by a diagonal and labelled -1 in that solve.
    Global labels with the all-reduce, or each rank's own numbering with the
    caller's global-level reduction; the forward and the gradient, eager and
    jitted, against the pseudoinverse."""
    import scipy.sparse as sp

    from jaxamg.matrices import poisson_matrix_stretched
    from jaxamg.mpi_utils import partition_csr_matrix
    from jaxamg.utils import to_scipy

    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        parts = [
            poisson_matrix_stretched(nx, ny, r, normalize=False, dtype=jnp.float64)
            for nx, ny, r in ((14, 10, 1.06), (10, 8, 1.10), (6, 5, 1.0))
        ]
        labels_global = np.concatenate(
            [np.full(v.shape[0], i) for i, (_, v) in enumerate(parts)]
        )
        anchored = labels_global == 2
        A_global = (
            sp.block_diag([to_scipy(a) for a, _ in parts])
            + sp.diags(np.where(anchored, 0.5, 0.0))
        ).tocsr()
        n, count = A_global.shape[0], 3
        rng = np.random.default_rng(1)
        b_global = rng.standard_normal(n)
        w_global = rng.standard_normal(n) + 0.5
        A_local, row_start, row_end = partition_csr_matrix(A_global, rank, nranks)
        rows = slice(row_start, row_end)
        b = jaxamg.make_sharded_vector(b_global[rows], comm=comm, mesh=mesh)
        w = jaxamg.make_sharded_vector(w_global[rows], comm=comm, mesh=mesh)
        A = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)
        solver = jaxamg.make_sharded_solver(
            A, b, config={"tolerance": 1e-10, "max_iters": 300}, is_symmetric=True
        )
        n_max = A.max_local_size
        local = np.where(anchored, -1, labels_global)[rows]
        declared = {}
        if reduction == "caller":
            # Rank r numbers global label g as slot (g + r) mod count; the
            # reduction maps every rank's slots to the global labels, sums
            # them and maps the totals back (global arrays, sharded like b).
            local = np.where(local >= 0, (local + rank) % count, -1)
            slots = np.arange(nranks)[:, None] * count + np.arange(count)[None, :]
            global_of = (np.arange(count)[None, :] - np.arange(nranks)[:, None]) % count

            def label_sum(sums):
                whole = jax.sharding.reshard(sums, jax.P())
                totals = jnp.zeros((count, sums.shape[1]), sums.dtype)
                totals = totals.at[global_of.ravel()].add(whole[slots.ravel()])
                return jax.sharding.reshard(
                    totals[global_of.ravel()], jax.P("rank", None)
                )

            declared["label_sum"] = label_sum
        labels = _global_vector(
            np.pad(local.astype(np.int32), (0, n_max - len(local)), constant_values=-1),
            nranks * n_max,
            mesh,
        )
        basis = "constant"
        if extreme_scale:
            values = np.array([1e250, 1e-250, 1.0])[labels_global[rows]]
            basis = _global_vector(
                np.pad(values, (0, n_max - len(values))), nranks * n_max, mesh
            )

        # A bad value only on rank 1 must raise everywhere before solving.
        invalid = local.astype(np.int64).copy()
        if rank == 1:
            invalid[0] = 2**32  # validate before int32 narrowing
        invalid = _global_vector(
            np.pad(invalid, (0, n_max - len(invalid)), constant_values=-1),
            nranks * n_max,
            mesh,
        )
        with pytest.raises(ValueError, match=r"\[-1, 3\)"):
            solver(b, A=A.data, nullspace=basis, labels=(count, invalid), **declared)

        def solve(rhs, A_data, rows_, basis_):
            return solver(
                rhs, A=A_data, nullspace=basis_, labels=(count, rows_), **declared
            )[0]

        def loss(rhs, A_data, rows_, basis_, weights):
            return jnp.sum(weights * solve(rhs, A_data, rows_, basis_))

        with jax.set_mesh(mesh):
            static = () if extreme_scale else (3,)
            x_eager = solve(b, A.data, labels, basis)
            x_jit = jax.jit(solve, static_argnums=static)(b, A.data, labels, basis)
            g_jit = jax.jit(jax.grad(loss), static_argnums=static)(
                b, A.data, labels, basis, w
            )
        x1, x2, g = (_gather_unpadded(v, solver, comm) for v in (x_eager, x_jit, g_jit))
        if rank == 0:
            pinv = np.linalg.pinv(A_global.toarray())
            consistent = b_global.copy()
            for label in (0, 1):
                on = labels_global == label
                consistent[on] -= b_global[on].mean()
            x_ref = pinv @ consistent
            for x in (x1, x2):
                np.testing.assert_allclose(
                    x, x_ref, rtol=1e-6, atol=1e-8 * np.linalg.norm(x_ref)
                )
            g_ref = pinv.T @ w_global
            np.testing.assert_allclose(
                g, g_ref, rtol=1e-6, atol=1e-8 * np.linalg.norm(g_ref)
            )
    finally:
        jax.config.update("jax_enable_x64", False)


def test_sharded_operator_float64_precision(sharding_context):
    """A float64 sharded solve through a rank-local operator is as accurate as
    the float64 matrix."""
    from test_mpi import _variable_coefficient_operator

    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        grid = 16
        n = grid * grid
        op, dense = _variable_coefficient_operator(grid)
        local_op, start, end = partition_operator(op, n, rank, nranks)
        shape = (end - start, n)
        local_op = jaxamg.with_cache(
            local_op, coloring=jaxamg.cache_coloring(local_op, shape)
        )
        b_global = np.random.default_rng(1).standard_normal(n)
        reference = np.linalg.solve(dense, b_global)
        b = jaxamg.make_sharded_vector(
            b_global[start:end], comm=comm, mesh=mesh, global_size=n
        )
        matrix = jaxamg.make_sharded_matrix(local_op, b, comm=comm, mesh=mesh)
        config = {
            "solver": "PCG",
            "preconditioner": {"solver": "AMG", "max_iters": 1},
            "tolerance": 1e-13,
            "max_iters": 400,
            "communicator": "MPI_DIRECT",
        }
        solver = jaxamg.make_sharded_solver(matrix, b, config=config)
        for current_operator in (None, local_op):
            with jax.set_mesh(mesh):
                x, _ = solver(b, A=current_operator)
            x = _gather_unpadded(x, solver, comm)
            assert np.linalg.norm(x - reference) / np.linalg.norm(reference) < 1e-10
    finally:
        jax.config.update("jax_enable_x64", False)


def _periodic_skew_operator(grid, skew):
    """A nonsymmetric, diagonally dominant 5-point operator on a torus: every
    row (and column) has five entries, so any equal row partition gives every
    rank the same local sizes for A and Aᵀ."""
    import scipy.sparse

    rows, cols, vals = [], [], []
    for i in range(grid):
        for j in range(grid):
            r = i * grid + j
            rows += [r] * 5
            cols += [
                r,
                ((i + 1) % grid) * grid + j,
                ((i - 1) % grid) * grid + j,
                i * grid + (j + 1) % grid,
                i * grid + (j - 1) % grid,
            ]
            vals += [4.5, -1 - skew, -1 + skew, -1 - skew / 2, -1 + skew / 2]
    A = scipy.sparse.csr_matrix(
        (vals, (rows, cols)), shape=(grid * grid,) * 2, dtype=np.float64
    )
    A.sort_indices()
    return A


def test_sharded_implicit_derivatives_all_orders(sharding_context):
    """The sharded implicit core: forward, reverse and second-order derivatives
    of a global objective match a dense float64 reference (global arrays are
    passed as arguments: a jit cannot close over multi-process arrays)."""
    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        grid = 10
        n = grid * grid
        A_global = _periodic_skew_operator(grid, 0.3)
        start, end, _ = get_partition_info(n, rank, nranks)
        rows = A_global[start:end]
        rows.sort_indices()
        A_local = jsp.BCSR(
            (
                jnp.asarray(rows.data),
                jnp.asarray(rows.indices, jnp.int64),
                jnp.asarray(rows.indptr, jnp.int32),
            ),
            shape=rows.shape,
        )
        b_global = np.random.default_rng(1).standard_normal(n)
        w_global = np.random.default_rng(2).standard_normal(n)
        b = jaxamg.make_sharded_vector(
            b_global[start:end], comm=comm, mesh=mesh, global_size=n
        )
        w = jaxamg.make_sharded_vector(
            w_global[start:end], comm=comm, mesh=mesh, global_size=n
        )
        matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)
        config = {
            # A tiny distributed system: an AMG hierarchy here fails inside
            # AmgX ("CUDA kernel launch error"), so a Jacobi-preconditioned
            # BiCGSTAB, as in the other sharded tests.
            "solver": "PBICGSTAB",
            "preconditioner": {"solver": "JACOBI_L1"},
            "tolerance": 1e-14,
            "max_iters": 1000,
            "communicator": "MPI_DIRECT",
        }
        solver = jaxamg.make_sharded_solver(matrix, b, config=config)

        def loss(t, data, rhs, weights):
            x, _ = solver(rhs * (1 + t), A=data * (1 + 0.1 * t))
            return jnp.sum(weights * x)

        def dense(t):
            return w_global @ np.linalg.solve(
                A_global.toarray() * (1 + 0.1 * t), b_global * (1 + t)
            )

        t0, h = 0.3, 1e-4
        first = (dense(t0 + h) - dense(t0 - h)) / (2 * h)
        second = (dense(t0 + h) - 2 * dense(t0) + dense(t0 - h)) / h**2
        args = (matrix.data, b, w)
        with jax.set_mesh(mesh):
            results = {
                "reverse": (jax.jit(jax.grad(loss))(t0, *args), first),
                "forward": (jax.jit(jax.jacfwd(loss))(t0, *args), first),
                "eager reverse": (jax.grad(loss)(t0, *args), first),
                "RR": (jax.jit(jax.grad(jax.grad(loss)))(t0, *args), second),
                "FR": (jax.jit(jax.jacfwd(jax.grad(loss)))(t0, *args), second),
            }
        for name, (value, expected) in results.items():
            np.testing.assert_allclose(float(value), expected, rtol=1e-5, err_msg=name)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_sharded_implicit_with_unequal_local_sizes(sharding_context):
    """Ranks with different local nonzero counts run one program (rank-local
    structure enters at run time): reverse derivatives match a dense adjoint
    reference and forward mode gives the analytic tangent."""
    import scipy.sparse

    from jaxamg.matrices import poisson_matrix

    comm, rank, nranks, mesh = sharding_context
    grid = 10
    n = grid * grid
    # skew=2 cancels some couplings exactly: the ranks' nonzero counts differ.
    A_global = scipy.sparse.csr_matrix(
        np.asarray(poisson_matrix(grid, skew=2.0).todense(), dtype=np.float32)
    )
    start, end, _ = get_partition_info(n, rank, nranks)
    rows = A_global[start:end]
    rows.sort_indices()
    assert len(set(comm.allgather(rows.nnz))) > 1
    A_local = jsp.BCSR(
        (
            jnp.asarray(rows.data),
            jnp.asarray(rows.indices, jnp.int32),
            jnp.asarray(rows.indptr, jnp.int32),
        ),
        shape=rows.shape,
    )
    b = jaxamg.make_sharded_vector(
        np.ones(end - start, np.float32), comm=comm, mesh=mesh, global_size=n
    )
    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)
    config = {
        "solver": "PBICGSTAB",
        "preconditioner": {"solver": "JACOBI_L1"},
        "communicator": "MPI_DIRECT",
        "tolerance": 1e-7,
        "max_iters": 1000,
    }
    solver = jaxamg.make_sharded_solver(matrix, b, config=config)

    def local(value):
        return np.asarray(value.addressable_shards[0].data)

    with jax.set_mesh(mesh):
        grad_data, grad_b = jax.jit(
            jax.grad(lambda d, r: jnp.sum(solver(r, A=d)[0]), argnums=(0, 1))
        )(matrix.data, b)
        x = jax.jit(lambda d, r: solver(r, A=d)[0])(matrix.data, b)
        # x(A (1 + t)) = x / (1 + t): the tangent along A itself is -x.
        tangent = jax.jit(
            lambda d, r: jax.jvp(lambda dd: solver(r, A=dd)[0], (d,), (d,))[1]
        )(matrix.data, b)
    # Dense adjoint reference: λ = A⁻ᵀ 1, b̄ = λ, Ā_ij = -λ_i x_j on the pattern.
    dense = A_global.toarray().astype(np.float64)
    x_ref = np.linalg.solve(dense, np.ones(n))
    lam = np.linalg.solve(dense.T, np.ones(n))
    local_rows = np.repeat(np.arange(start, end), np.diff(rows.indptr))
    n_local = end - start
    np.testing.assert_allclose(local(x)[:n_local], x_ref[start:end], rtol=1e-4)
    np.testing.assert_allclose(local(grad_b)[:n_local], lam[start:end], rtol=1e-4)
    np.testing.assert_allclose(
        local(grad_data)[: rows.nnz],
        -lam[local_rows] * x_ref[rows.indices],
        rtol=1e-3,
        atol=1e-6,
    )
    np.testing.assert_allclose(local(tangent), -local(x), rtol=1e-4, atol=1e-6)


def test_sharded_halo_operator(sharding_context):
    """A halo-form operator in the sharding interface: the same solution and
    matrix-value gradient as the directly assembled local rows, including as a
    traced ``A=`` override derived with ``with_fn``."""
    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        n = 8 * nranks
        r0, r1, n_local = get_partition_info(n, rank, nranks)
        left, right = (r0 - 1) % n, r1 % n
        ghost_ids = np.array(sorted({left, right}), dtype=np.int64)
        slot_left = int(np.searchsorted(ghost_ids, left))
        slot_right = int(np.searchsorted(ghost_ids, right))
        skew = 0.3

        def fn(scale):
            def apply(x_local, x_ghost):
                prev = jnp.concatenate(
                    [x_ghost[slot_left : slot_left + 1], x_local[:-1]]
                )
                nxt = jnp.concatenate(
                    [x_local[1:], x_ghost[slot_right : slot_right + 1]]
                )
                return scale * (4.5 * x_local - (1 + skew) * prev - (1 - skew) * nxt)

            return apply

        rows, cols, vals = [], [], []
        for k, c in enumerate(range(r0, r1)):
            entries = {c: 4.5, (c - 1) % n: -(1 + skew), (c + 1) % n: -(1 - skew)}
            for col in sorted(entries):
                rows.append(k)
                cols.append(col)
                vals.append(entries[col])
        indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n_local))))
        direct = jsp.BCSR(
            (
                jnp.asarray(vals),
                jnp.asarray(cols, jnp.int64),
                jnp.asarray(indptr, jnp.int32),
            ),
            shape=(n_local, n),
        )
        b = jaxamg.make_sharded_vector(
            np.random.default_rng(rank).standard_normal(n_local),
            comm=comm,
            mesh=mesh,
            global_size=n,
        )
        config = {
            "solver": "PBICGSTAB",
            "preconditioner": {"solver": "JACOBI_L1"},
            "tolerance": 1e-13,
            "max_iters": 500,
            "communicator": "MPI_DIRECT",
        }
        base = jaxamg.halo_operator(fn(1.0), n_local=n_local, ghost_ids=ghost_ids)
        halo_matrix = jaxamg.make_sharded_matrix(base, b, comm=comm, mesh=mesh)
        direct_matrix = jaxamg.make_sharded_matrix(direct, b, comm=comm, mesh=mesh)
        np.testing.assert_array_equal(
            np.asarray(halo_matrix.data.addressable_shards[0].data),
            np.asarray(direct_matrix.data.addressable_shards[0].data),
        )
        solver = jaxamg.make_sharded_solver(halo_matrix, b, config=config)
        x_halo = solver(b)[0]
        x_direct = jaxamg.make_sharded_solver(direct_matrix, b, config=config)(b)[0]
        np.testing.assert_allclose(
            np.asarray(x_halo.addressable_shards[0].data),
            np.asarray(x_direct.addressable_shards[0].data),
            rtol=1e-12,
        )

        # d/ds of sum(x(s)) through a traced halo override, against the analytic
        # value -sum(x)/s at s = 1 (x scales as 1/s).
        def loss(s, rhs):
            return jnp.sum(solver(rhs, A=base.with_fn(fn(s)))[0])

        with jax.set_mesh(mesh):
            grad = jax.jit(jax.grad(loss))(1.0, b)
        total = comm.allreduce(
            float(np.sum(np.asarray(x_halo.addressable_shards[0].data)[:n_local]))
        )
        np.testing.assert_allclose(float(grad), -total, rtol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("communicator", ["MPI", "MPI_DIRECT"])
def test_sharded_global_operator(sharding_context, communicator):
    """A global-view operator (its own communication: neighbour ppermutes on
    the global vector), probed with the distributed colouring: the same packed
    values and solution as the directly assembled rows, eagerly and jitted,
    and parameter gradients through a traced ``with_fn`` operator, jitted and
    eager."""
    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        n = 8 * nranks
        r0, r1, n_local = get_partition_info(n, rank, nranks)
        skew = 0.3

        axis = "rank"
        forward = [(i, (i + 1) % nranks) for i in range(nranks)]
        backward = [(i, (i - 1) % nranks) for i in range(nranks)]

        def shifts(x):  # this rank's block of v -> blocks of v[i-1] and v[i+1]
            before = jax.lax.ppermute(x[-1:], axis, perm=forward)
            after = jax.lax.ppermute(x[:1], axis, perm=backward)
            return jnp.concatenate([before, x[:-1]]), jnp.concatenate([x[1:], after])

        def fn(scale):
            def apply(v):
                def body(x):
                    prev, nxt = shifts(x)
                    return scale * (4.5 * x - (1 + skew) * prev - (1 - skew) * nxt)

                return jax.shard_map(
                    body, mesh=mesh, in_specs=jax.P(axis), out_specs=jax.P(axis)
                )(v)

            return apply

        rows, cols, vals = [], [], []
        for k, c in enumerate(range(r0, r1)):
            entries = {c: 4.5, (c - 1) % n: -(1 + skew), (c + 1) % n: -(1 - skew)}
            for col in sorted(entries):
                rows.append(k)
                cols.append(col)
                vals.append(entries[col])
        indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n_local))))
        direct = jsp.BCSR(
            (
                jnp.asarray(vals),
                jnp.asarray(cols, jnp.int64),
                jnp.asarray(indptr, jnp.int32),
            ),
            shape=(n_local, n),
        )
        b = jaxamg.make_sharded_vector(
            np.random.default_rng(rank).standard_normal(n_local),
            comm=comm,
            mesh=mesh,
            global_size=n,
        )
        config = {
            "solver": "PBICGSTAB",
            "preconditioner": {"solver": "JACOBI_L1"},
            "tolerance": 1e-13,
            "max_iters": 500,
            "communicator": communicator,
        }
        op = jaxamg.global_operator(fn(1.0), cols, indptr, comm=comm, mesh=mesh)
        # Each column conflicts with its four distance-one/two neighbours.
        # Priority-greedy colouring need not attain the chromatic minimum.
        assert op.n_colors <= 5
        matrix = jaxamg.make_sharded_matrix(op, b, comm=comm, mesh=mesh)
        direct_matrix = jaxamg.make_sharded_matrix(direct, b, comm=comm, mesh=mesh)
        np.testing.assert_allclose(
            np.asarray(matrix.data.addressable_shards[0].data),
            np.asarray(direct_matrix.data.addressable_shards[0].data),
            rtol=0,
            atol=1e-15,
        )
        solver = jaxamg.make_sharded_solver(matrix, b, config=config)
        x = solver(b)[0]
        x_direct = jaxamg.make_sharded_solver(direct_matrix, b, config=config)(b)[0]
        np.testing.assert_allclose(
            np.asarray(x.addressable_shards[0].data),
            np.asarray(x_direct.addressable_shards[0].data),
            rtol=1e-12,
        )
        # Eager materialization of new parameters, direct call.
        x2 = solver(b, A=op.with_fn(fn(2.0)))[0]
        np.testing.assert_allclose(
            np.asarray(x2.addressable_shards[0].data),
            np.asarray(x.addressable_shards[0].data) / 2,
            rtol=1e-12,
        )
        total = comm.allreduce(
            float(np.sum(np.asarray(x.addressable_shards[0].data)[:n_local]))
        )
        with jax.set_mesh(mesh):
            # Jitted materialization: exact packed values (sum of squares).
            packed = jax.jit(lambda rhs: op._packed_values(rhs))(b)
            grad_data = jax.jit(
                jax.grad(lambda s, data, rhs: jnp.sum(solver(rhs, A=data * s)[0]))
            )(1.0, matrix.data, b)
            grad_op = jax.jit(
                jax.grad(lambda s, rhs: jnp.sum(solver(rhs, A=op.with_fn(fn(s)))[0]))
            )(1.0, b)
            # Eagerly, with the right-hand side closed over (not traced).
            grad_eager = jax.grad(lambda s: jnp.sum(solver(b, A=op.with_fn(fn(s)))[0]))(
                1.0
            )
        np.testing.assert_array_equal(
            np.asarray(packed.addressable_shards[0].data),
            np.asarray(matrix.data.addressable_shards[0].data),
        )
        np.testing.assert_allclose(float(grad_data), -total, rtol=1e-9)
        np.testing.assert_allclose(float(grad_op), -total, rtol=1e-9)
        np.testing.assert_allclose(float(grad_eager), -total, rtol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_sharded_overrides_must_keep_the_structure(sharding_context):
    """An operator override with the solver's nonzero counts but other
    columns is refused on every rank (its values would otherwise be read in
    the solver's structure): global-view operators and colour-recovered
    callables alike."""
    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        n_local = 8
        n = n_local * nranks
        lo = rank * n_local
        rows = np.arange(lo, lo + n_local)
        indptr = np.arange(n_local + 1, dtype=np.int32)
        b = jaxamg.make_sharded_vector(
            np.arange(lo + 1, lo + n_local + 1, dtype=np.float64),
            comm=comm,
            mesh=mesh,
            global_size=n,
        )
        diagonal = jsp.BCSR(
            (
                jnp.full(n_local, 2.0),
                jnp.asarray(rows, jnp.int64),
                jnp.asarray(indptr),
            ),
            shape=(n_local, n),
        )
        solver = jaxamg.make_sharded_solver(
            jaxamg.make_sharded_matrix(diagonal, b, comm=comm, mesh=mesh),
            b,
            config={
                "solver": "PBICGSTAB",
                "preconditioner": {"solver": "JACOBI_L1"},
                "tolerance": 1e-12,
                "max_iters": 100,
                "communicator": "MPI_DIRECT",
            },
        )
        swap = jax.shard_map(
            lambda x: 2 * x.reshape(-1, 2)[:, ::-1].reshape(-1),
            mesh=mesh,
            in_specs=jax.P("rank"),
            out_specs=jax.P("rank"),
        )
        swapped = jaxamg.global_operator(swap, rows ^ 1, indptr, comm=comm, mesh=mesh)
        with pytest.raises(ValueError, match="sparsity structure"):
            solver(b, A=swapped)
        # A colouring recovering the swapped columns (one entry per row).
        coloring = (np.arange(n_local), rows ^ 1, np.arange(n) % 2, 2, (n_local, n))
        local = jaxamg.with_cache(
            lambda x: 2 * x[lo : lo + n_local].reshape(-1, 2)[:, ::-1].reshape(-1),
            coloring=coloring,
        )
        with pytest.raises(ValueError, match="sparsity structure"):
            solver(b, A=local)
        # The solver's own structure is still accepted.
        x = solver(b, A=diagonal)[0]
        np.testing.assert_allclose(
            np.asarray(solver.local_vector(x)), np.arange(lo + 1, lo + n_local + 1) / 2
        )
    finally:
        jax.config.update("jax_enable_x64", False)


def test_sharded_rank_local_operator_all_derivative_modes(sharding_context):
    """A rank-local operator ``θ·I`` passed as ``A=``: forward, reverse and
    second derivatives through the ownership maps (x = b/θ)."""
    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        n_local = 8
        n = n_local * nranks
        lo = rank * n_local
        values = np.arange(lo + 1, lo + n_local + 1, dtype=np.float64)
        b = jaxamg.make_sharded_vector(values, comm=comm, mesh=mesh, global_size=n)
        coloring = (
            np.arange(n_local),
            np.arange(lo, lo + n_local),
            np.zeros(n, np.int32),
            1,
            (n_local, n),
        )
        base = jaxamg.with_cache(
            lambda x: 2.0 * x[lo : lo + n_local], coloring=coloring
        )
        solver = jaxamg.make_sharded_solver(
            jaxamg.make_sharded_matrix(base, b, comm=comm, mesh=mesh),
            b,
            config={
                "solver": "PBICGSTAB",
                "preconditioner": {"solver": "JACOBI_L1"},
                "tolerance": 1e-13,
                "max_iters": 100,
                "communicator": "MPI_DIRECT",
            },
        )

        def total(theta, rhs):
            op = jaxamg.with_cache(
                lambda x: theta * x[lo : lo + n_local], coloring=coloring
            )
            return jnp.sum(solver(rhs, A=op)[0])

        s = np.sum(np.arange(1, n + 1))  # Σ b; total(θ) = s/θ
        theta = jnp.array(2.0)
        with jax.set_mesh(mesh):
            _, forward = jax.jvp(lambda t: total(t, b), (theta,), (jnp.array(1.0),))
            reverse = jax.jit(jax.grad(total))(theta, b)
            second = jax.jit(jax.jacfwd(jax.grad(total)))(theta, b)
        theta = float(theta)
        np.testing.assert_allclose(float(forward), -s / theta**2, rtol=1e-10)
        np.testing.assert_allclose(float(reverse), -s / theta**2, rtol=1e-10)
        np.testing.assert_allclose(float(second), 2 * s / theta**3, rtol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_sharded_two_column_null_space(sharding_context):
    """Two interleaved periodic Laplacians (even and odd rows): a two-column
    null space, set up across ranks (elementwise column maxima) and solved."""
    import scipy.sparse as sp

    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        n_local = 8
        n = n_local * nranks
        lo, hi = rank * n_local, (rank + 1) * n_local
        A = sp.lil_matrix((n, n))
        for i in range(n):
            A[i, i] = 2.0
            A[i, (i + 2) % n] -= 1.0
            A[i, (i - 2) % n] -= 1.0
        A = A.tocsr()
        basis = np.stack([np.arange(n) % 2 == 0, np.arange(n) % 2 == 1], axis=1)
        basis = basis.astype(np.float64)
        b_global = np.sin(np.arange(n)) + 0.2
        b_global -= basis @ np.linalg.lstsq(basis, b_global, rcond=None)[0]
        rows = A[lo:hi]
        local = jsp.BCSR(
            (
                jnp.asarray(rows.data),
                jnp.asarray(rows.indices, jnp.int64),
                jnp.asarray(rows.indptr, jnp.int32),
            ),
            shape=(n_local, n),
        )
        local_basis = jnp.asarray(basis[lo:hi])
        b = jaxamg.make_sharded_vector(
            b_global[lo:hi], comm=comm, mesh=mesh, global_size=n
        )
        matrix = jaxamg.make_sharded_matrix(
            jaxamg.with_cache(
                local, nullspace=local_basis, transpose_nullspace=local_basis
            ),
            b,
            comm=comm,
            mesh=mesh,
        )
        solver = jaxamg.make_sharded_solver(
            matrix,
            b,
            config={"solver": "PCG", "tolerance": 1e-12, "max_iters": 500},
            is_symmetric=True,
        )
        x = _gather_unpadded(solver(b)[0], solver, comm)
        np.testing.assert_allclose(A @ x, b_global, atol=1e-9)
        np.testing.assert_allclose(basis.T @ x, 0, atol=1e-9)
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.parametrize("family", ["dense", "uneven", "empty_rank", "empty"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_global_operator_tiled_values_and_derivatives(
    sharding_context, monkeypatch, family, dtype
):
    """Small tiles, missing local colours, output promotion, and all AD modes."""
    import importlib

    module = importlib.import_module("jaxamg.global_operator")
    monkeypatch.setattr(module, "_PROBE_TILE_ENTRIES", 3)
    comm, rank, nranks, mesh = sharding_context
    jax.config.update("jax_enable_x64", True)
    try:
        width = 8
        # Blocks differ by rank, so some colours have no entries on a rank.
        dense = np.arange(1, width**2 + 1, dtype=np.float64).reshape(width, width)
        if family == "uneven" and rank == 0:
            dense = np.diag(np.diag(dense))
        elif family == "empty" or (family == "empty_rank" and rank == 0):
            dense[:] = 0
        rows, cols = np.nonzero(dense)
        ptr = np.r_[0, np.cumsum(np.bincount(rows, minlength=width))]
        coefficients = _global_vector(dense.ravel(), width**2 * nranks, mesh)
        template = _global_vector(np.ones(width, dtype=dtype), width * nranks, mesh)
        traces = []

        def fn(scale, coefficients):
            def apply(x):
                traces.append(None)
                return jax.shard_map(
                    lambda a, v, s: s**2 * (a.reshape(width, width) @ v),
                    mesh=mesh,
                    in_specs=(jax.P("rank"), jax.P("rank"), jax.P()),
                    out_specs=jax.P("rank"),
                )(coefficients, x, scale)

            return apply

        op = module.global_operator(
            fn(1.0, coefficients), cols + rank * width, ptr, comm=comm, mesh=mesh
        )
        with jax.set_mesh(mesh):
            actual = jax.jit(lambda s, x, a: op.with_fn(fn(s, a))._packed_values(x))(
                2.0, template, coefficients
            )
            # Tracing is independent of the number of colours/tiles.
            assert len(traces) <= 3
            expected = np.pad(4 * dense[rows, cols], (0, op.max_nnz - rows.size))
            np.testing.assert_array_equal(actual.addressable_shards[0].data, expected)
            assert actual.dtype == jnp.float64
            total = comm.allreduce(float(dense.sum()))

            def loss(s, x, a):
                return jnp.sum(op.with_fn(fn(s, a))._packed_values(x))

            np.testing.assert_allclose(
                jax.jit(jax.grad(loss))(2.0, template, coefficients), 4 * total
            )
            np.testing.assert_allclose(
                jax.jit(jax.jacfwd(loss))(2.0, template, coefficients), 4 * total
            )
            np.testing.assert_allclose(
                jax.jit(jax.grad(jax.grad(loss)))(2.0, template, coefficients),
                2 * total,
            )
            if family == "dense":
                grad = jax.jit(jax.grad(loss, argnums=2))(2.0, template, coefficients)
                np.testing.assert_array_equal(grad.addressable_shards[0].data, 4.0)
    finally:
        jax.config.update("jax_enable_x64", False)


def test_global_operator_gradient_memory_does_not_stack_probes(sharding_context):
    """A high-degree row adds colours without making the whole matrix dense."""
    comm, rank, nranks, mesh = sharding_context
    n = 65536
    template = _global_vector(np.ones(n, np.float32), n * nranks, mesh)
    memory = []
    for width in (8, 128):

        def fn(scale):
            def apply(x):
                return jax.shard_map(
                    lambda v, s: (s * v).at[0].set(s * jnp.sum(v[:width])),
                    mesh=mesh,
                    in_specs=(jax.P("rank"), jax.P()),
                    out_specs=jax.P("rank"),
                )(x, scale)

            return apply

        indices = np.r_[np.arange(width), np.arange(1, n)] + rank * n
        indptr = np.r_[0, width + np.arange(n)]
        op = jaxamg.global_operator(fn(1.0), indices, indptr, comm=comm, mesh=mesh)
        with jax.set_mesh(mesh):
            loss = lambda s, x: jnp.sum(op.with_fn(fn(s))._packed_values(x))
            executable = jax.jit(jax.grad(loss)).lower(1.0, template).compile()
            memory.append(executable.memory_analysis().temp_size_in_bytes)
            np.testing.assert_allclose(
                executable(1.0, template), (n + width - 1) * nranks
            )
    # A stack of 128 probe vectors alone would cost 32 MiB per rank. Allow
    # compiler variation while rejecting storage proportional to colours*rows.
    assert memory[1] < 2 * memory[0] + 1024**2


def test_global_operator_canonicalizes_and_owns_structure(sharding_context):
    comm, rank, nranks, mesh = sharding_context
    width = 8
    indices = np.repeat(np.arange(width) + rank * width, 2)
    indptr = np.arange(width + 1) * 2
    op = jaxamg.global_operator(lambda x: 2 * x, indices, indptr, comm=comm, mesh=mesh)
    indices[:] = 0
    indptr[:] = 0
    assert op.nnz == width
    np.testing.assert_array_equal(op.indices, np.arange(width) + rank * width)
    for array in (op.indices, op.indptr):
        with pytest.raises(ValueError):
            array.flags.writeable = True
    b = _global_vector(np.ones(width, np.float32), width * nranks, mesh)
    np.testing.assert_array_equal(op._packed_values(b).addressable_shards[0].data, 2.0)


@pytest.mark.parametrize("bad", ["columns", "pointers", "dtype"])
def test_global_operator_refuses_invalid_structure_collectively(sharding_context, bad):
    comm, rank, nranks, mesh = sharding_context
    cols = np.array([rank], np.int64)
    ptr = np.array([0, 1], np.int64)
    if rank == 0:
        if bad == "columns":
            cols[0] = -1
        elif bad == "pointers":
            ptr[1] = 2
        else:
            cols = cols.astype(float)
    with pytest.raises(ValueError, match="invalid CSR"):
        jaxamg.global_operator(lambda x: x, cols, ptr, comm=comm, mesh=mesh)
