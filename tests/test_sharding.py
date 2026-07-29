import sys
from types import ModuleType, SimpleNamespace

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

import jaxamg
import jaxamg.sharding as sharding_module
from jaxamg.mpi_utils import TransposePlan

pytestmark = pytest.mark.skipif(
    not hasattr(jax, "shard_map"), reason="jax.shard_map is unavailable"
)


def test_sharding_comm_defaults_to_world(monkeypatch):
    default_comm = object()
    mpi4py = ModuleType("mpi4py")
    mpi4py.MPI = SimpleNamespace(COMM_WORLD=default_comm)
    monkeypatch.setitem(sys.modules, "mpi4py", mpi4py)

    explicit_comm = object()
    assert sharding_module._resolve_comm(None) is default_comm
    assert sharding_module._resolve_comm(explicit_comm) is explicit_comm


def _single_device_array(values):
    mesh = jax.make_mesh((1,), ("rank",), devices=[jax.devices()[0]])
    sharding = jax.NamedSharding(mesh, jax.P("rank"))
    return mesh, jax.device_put(jnp.asarray(values), sharding)


def _single_rank_transpose_plan(A):
    nnz = len(A.data)
    return TransposePlan(
        np.asarray(A.indices),
        np.asarray(A.indptr),
        np.arange(nnz, dtype=np.int32),
        np.arange(nnz, dtype=np.int32),
        np.full((1, 1), nnz, dtype=np.int32),
        np.full((1, 1), nnz, dtype=np.int32),
        nnz,
    )


def test_make_sharded_vector_constructs_default_mesh():
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        allgather=lambda value: [value],
    )
    values = np.arange(4, dtype=np.float32)

    b = jaxamg.make_sharded_vector(values, comm=comm, global_size=4)

    assert isinstance(b.sharding, jax.NamedSharding)
    # The default mesh holds one device per MPI rank, so extra local devices
    # (e.g. on a multi-GPU workstation) do not invalidate a small communicator.
    assert b.sharding.mesh.size == 1
    assert b.sharding.mesh.devices.flat[0] == jax.devices()[0]
    assert b.sharding.spec == jax.P("rank")
    np.testing.assert_array_equal(np.asarray(b), values)


def test_make_sharded_vector_rejects_batched_rhs():
    mesh = jax.make_mesh((1,), ("rank",), devices=[jax.devices()[0]])
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        allgather=lambda value: [value],
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        jaxamg.make_sharded_vector(
            np.ones((4, 2), dtype=np.float32), comm=comm, mesh=mesh
        )


def test_sharded_inputs_preserve_device_arrays(monkeypatch):
    mesh, local_values = _single_device_array(np.arange(4, dtype=np.float32))
    A_local = jsp.BCSR.fromdense(jnp.eye(4, dtype=jnp.float32))
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        Get_rank=lambda: 0,
        allgather=lambda value: [value],
    )
    monkeypatch.setattr(sharding_module, "_validate_runtime", lambda *args: None)

    with jax.transfer_guard_device_to_host("disallow"):
        b = jaxamg.make_sharded_vector(
            local_values, comm=comm, mesh=mesh, global_size=4
        )
        matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)

    assert b.addressable_shards[0].device == next(iter(local_values.devices()))
    assert matrix.data.addressable_shards[0].device == next(
        iter(A_local.data.devices())
    )


def test_make_sharded_solver_preserves_global_array_contract(monkeypatch):
    mesh, b = _single_device_array(np.arange(4, dtype=np.float32))
    A_local = jsp.BCSR.fromdense(jnp.eye(4, dtype=jnp.float32))
    allgather_calls = []

    def allgather(value):
        allgather_calls.append(value)
        return [value]

    comm = SimpleNamespace(
        Get_size=lambda: 1,
        Get_rank=lambda: 0,
        allgather=allgather,
    )
    halo_plan = SimpleNamespace(
        n_ghost=0,
        max_n_ghost=0,
        col_to_combined=np.arange(4, dtype=np.int32),
        send_ids_2d=np.zeros((1, 1), dtype=np.int32),
        recv_ghost_slot_2d=np.zeros((1, 1), dtype=np.int32),
    )
    halo_plan_calls = []

    def fake_build_halo_plan(*args, **kwargs):
        halo_plan_calls.append(args)
        return halo_plan

    monkeypatch.setattr(sharding_module, "_validate_runtime", lambda *args: None)
    monkeypatch.setattr(
        sharding_module,
        "build_halo_plan",
        fake_build_halo_plan,
    )
    monkeypatch.setattr(
        sharding_module,
        "_build_mpi_cache",
        lambda config, comm, nglobal, row_counts, max_nnz, nnz_out, plan, **kwargs: {
            "lrank": 99,
            "recvcounts_tuple": row_counts,
            "max_nnz": max_nnz,
            "nnz_out": nnz_out,
            "halo_plan": plan,
        },
    )
    monkeypatch.setattr(
        sharding_module,
        "with_cache",
        lambda A, **kwargs: A,
    )
    normalization_calls = []

    def fake_to_bcsr(A, **kwargs):
        normalization_calls.append(A)
        return A

    monkeypatch.setattr(sharding_module, "to_bcsr_matrix", fake_to_bcsr)
    transpose_calls = []

    def fake_transpose(*args):
        transpose_calls.append(A_local)
        return _single_rank_transpose_plan(A_local)

    monkeypatch.setattr(sharding_module, "build_transpose_plan", fake_transpose)

    def fake_solve(A, rhs, x0=None, **kwargs):
        x = A.data * rhs
        if x0 is not None:
            x = x + x0
        info = {
            "iterations": jnp.asarray(2.0),
            "residual": jnp.asarray(1e-6, dtype=rhs.dtype),
            "status": jnp.asarray(0.0),
            "residual_history": jnp.asarray([1.0, 0.1, 1e-6], dtype=rhs.dtype),
        }
        return x, info

    monkeypatch.setattr(sharding_module, "solve", fake_solve)

    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm)
    # Omit mesh to exercise inference from b.sharding.
    solver = jaxamg.make_sharded_solver(matrix, b)
    assert len(allgather_calls) == 2
    assert len(normalization_calls) == 1
    assert len(halo_plan_calls) == 1
    x, info = solver(b)
    np.testing.assert_array_equal(np.asarray(x), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(solver.local_vector(x)), np.asarray(b))
    assert solver.global_size == 4
    assert solver.local_size == 4
    np.testing.assert_array_equal(np.asarray(info["iterations"]), [2])
    assert info["residual_history"].shape == (1, 3)

    x_jit, _ = jax.jit(lambda matrix_data, rhs: solver(rhs, A_data=matrix_data))(
        matrix.data, b
    )
    np.testing.assert_array_equal(np.asarray(x_jit), np.asarray(b))

    # Multiple RHS use the same public batching path as the ordinary solver:
    # vmap a solver that accepts one vector at a time.
    batched_b = jnp.stack((b, 2 * b))
    batched_x = jax.vmap(lambda rhs: solver(rhs, A_data=matrix.data)[0])(batched_b)
    np.testing.assert_array_equal(np.asarray(batched_x), np.asarray(batched_b))

    with pytest.raises(ValueError, match="A_data must be passed explicitly"):
        jax.jit(solver).lower(b)

    # A traced x0 with a concrete RHS must not embed the cached matrix values.
    with pytest.raises(ValueError, match="A_data must be passed explicitly"):
        jax.jit(lambda guess: solver(b, guess)).lower(b)

    x_updated, _ = solver(b, A_data=2 * matrix.data)
    np.testing.assert_array_equal(np.asarray(x_updated), 2 * np.asarray(b))

    x_warm, _ = solver(b, b)
    np.testing.assert_array_equal(np.asarray(x_warm), 2 * np.asarray(b))

    with jax.set_mesh(mesh):
        grad_b = jax.grad(
            lambda data, rhs: jnp.sum(solver(rhs, A_data=data)[0] ** 2),
            argnums=1,
        )(matrix.data, b)
        grad_A_data = jax.grad(lambda data: jnp.sum(solver(b, A_data=data)[0] ** 2))(
            matrix.data
        )
        grad_b_warm, grad_x0 = jax.grad(
            lambda data, rhs, x0: jnp.sum(solver(rhs, x0, A_data=data)[0] ** 2),
            argnums=(1, 2),
        )(matrix.data, b, b)
    np.testing.assert_array_equal(np.asarray(grad_b), 2 * np.asarray(b))
    grad_A_local = matrix.local_matrix(grad_A_data)
    np.testing.assert_array_equal(
        np.asarray(grad_A_local.data), -2 * np.asarray(b) ** 2
    )
    np.testing.assert_array_equal(np.asarray(grad_b_warm), 4 * np.asarray(b))
    np.testing.assert_array_equal(np.asarray(grad_x0), np.zeros_like(np.asarray(b)))

    assert len(transpose_calls) == 1


def test_sharded_matrix_validates_partition(monkeypatch):
    mesh, b = _single_device_array(np.ones(4, dtype=np.float32))
    monkeypatch.setattr(sharding_module, "_validate_runtime", lambda *args: None)
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        Get_rank=lambda: 0,
        allgather=lambda value: [value],
    )

    with pytest.raises(ValueError, match="row counts"):
        jaxamg.make_sharded_matrix(
            SimpleNamespace(shape=(3, 4)),
            b,
            comm=comm,
            mesh=mesh,
        )


def test_sharded_solver_requires_sharded_matrix():
    _, b = _single_device_array(np.ones(4, dtype=np.float32))

    with pytest.raises(TypeError, match="make_sharded_matrix"):
        jaxamg.make_sharded_solver(jnp.eye(4), b)  # type: ignore[arg-type]
