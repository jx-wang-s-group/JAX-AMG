from types import SimpleNamespace

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

import jaxamg
import jaxamg.sharding as sharding_module

pytestmark = pytest.mark.skipif(
    not hasattr(jax, "shard_map"), reason="jax.shard_map is unavailable"
)


def _single_device_array(values):
    mesh = jax.make_mesh((1,), ("rank",), devices=[jax.devices()[0]])
    sharding = jax.NamedSharding(mesh, jax.P("rank"))
    return mesh, jax.device_put(jnp.asarray(values), sharding)


def test_make_sharded_solver_preserves_global_array_contract(monkeypatch):
    mesh, b = _single_device_array(np.arange(4, dtype=np.float32))
    A_local = jsp.BCSR.fromdense(jnp.eye(4, dtype=jnp.float32))
    comm = SimpleNamespace(Get_size=lambda: 1, allgather=lambda value: [value])
    halo_plan = SimpleNamespace(
        n_ghost=0,
        col_to_combined=np.arange(4, dtype=np.int32),
        send_ids_2d=np.zeros((1, 1), dtype=np.int32),
        recv_ghost_slot_2d=np.zeros((1, 1), dtype=np.int32),
    )

    monkeypatch.setattr(sharding_module, "_validate_runtime", lambda *args: None)
    monkeypatch.setattr(
        sharding_module,
        "cache_mpi_metadata",
        lambda *args, **kwargs: {
            "lrank": 99,
            "recvcounts_tuple": (4,),
            "max_nnz": 4,
            "halo_plan": halo_plan,
        },
    )
    monkeypatch.setattr(
        sharding_module,
        "with_cache",
        lambda A, **kwargs: A,
    )
    monkeypatch.setattr(
        sharding_module,
        "to_bcsr_matrix",
        lambda A, **kwargs: A,
    )
    transpose_calls = []

    def fake_transpose(A, *args):
        transpose_calls.append(A)
        return A, np.arange(len(A.data), dtype=np.int32)

    monkeypatch.setattr(
        sharding_module, "_transpose_distributed_matrix", fake_transpose
    )

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

    # Omit mesh to exercise inference from b.sharding.
    solver = jaxamg.make_sharded_solver(A_local, b, comm=comm)
    x, info = solver(b)
    np.testing.assert_array_equal(np.asarray(x), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(info["iterations"]), [2])
    assert info["residual_history"].shape == (1, 3)

    x_jit, _ = jax.jit(solver)(b)
    np.testing.assert_array_equal(np.asarray(x_jit), np.asarray(b))

    x_updated, _ = solver(b, A_data=2 * solver.A_data)
    np.testing.assert_array_equal(np.asarray(x_updated), 2 * np.asarray(b))

    x_warm, _ = solver(b, b)
    np.testing.assert_array_equal(np.asarray(x_warm), 2 * np.asarray(b))

    with jax.set_mesh(mesh):
        grad_b = jax.grad(lambda rhs: jnp.sum(solver(rhs)[0] ** 2))(b)
        grad_A_data = jax.grad(lambda data: jnp.sum(solver(b, A_data=data)[0] ** 2))(
            solver.A_data
        )
        grad_b_warm, grad_x0 = jax.grad(
            lambda rhs, x0: jnp.sum(solver(rhs, x0)[0] ** 2),
            argnums=(0, 1),
        )(b, b)
    np.testing.assert_array_equal(np.asarray(grad_b), 2 * np.asarray(b))
    grad_A_local = solver.local_matrix_gradient(grad_A_data)
    np.testing.assert_array_equal(
        np.asarray(grad_A_local.data), -2 * np.asarray(b) ** 2
    )
    np.testing.assert_array_equal(np.asarray(grad_b_warm), 4 * np.asarray(b))
    np.testing.assert_array_equal(np.asarray(grad_x0), np.zeros_like(np.asarray(b)))

    x_once, _ = jaxamg.solve_sharded(A_local, b, comm=comm, mesh=mesh)
    np.testing.assert_array_equal(np.asarray(x_once), np.asarray(b))
    assert len(transpose_calls) == 2


def test_sharded_solver_validates_matrix_partition(monkeypatch):
    mesh, b = _single_device_array(np.ones(4, dtype=np.float32))
    monkeypatch.setattr(sharding_module, "_validate_runtime", lambda *args: None)

    with pytest.raises(ValueError, match="expected shape"):
        jaxamg.make_sharded_solver(
            SimpleNamespace(shape=(3, 4)),
            b,
            comm=object(),
            mesh=mesh,
        )
