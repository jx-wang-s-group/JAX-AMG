import os
import sys
import warnings
from types import ModuleType, SimpleNamespace

import jax
import jax.experimental.sparse as jsp
import jax.numpy as jnp
import numpy as np
import pytest

import jaxamg
import jaxamg.sharding as sharding_module
from jaxamg.mpi_utils import TransposePlan
from jaxamg.transport import NeighbourPlan


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


# One rank: nothing travels, so the exchange is never called.
_EMPTY_EXCHANGE = NeighbourPlan(0, (), (), (), ())


def _single_rank_transpose_plan(A):
    nnz = len(A.data)
    return TransposePlan(
        np.asarray(A.indices),
        np.asarray(A.indptr),
        np.arange(nnz, dtype=np.int32),
        np.arange(nnz, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        np.zeros(0, dtype=np.int32),
        _EMPTY_EXCHANGE,
        nnz,
    )


def test_make_sharded_vector_constructs_default_mesh():
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        allgather=lambda value: [value],
        allreduce=lambda value, op=None: value,
        Allreduce=lambda *args, **kwargs: None,
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
        allreduce=lambda value, op=None: value,
        Allreduce=lambda *args, **kwargs: None,
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
        allreduce=lambda value, op=None: value,
        Allreduce=lambda *args, **kwargs: None,
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


@pytest.fixture
def mock_sharded_solver(monkeypatch):
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
        allreduce=lambda value, op=None: value,
        Allreduce=lambda *args, **kwargs: None,
    )
    halo_plan = SimpleNamespace(
        n_ghost=0,
        max_n_ghost=0,
        col_to_combined=np.arange(4, dtype=np.int32),
        send_ids=np.zeros(1, dtype=np.int32),  # plans keep one entry
        exchange=_EMPTY_EXCHANGE,
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
    monkeypatch.setattr(sharding_module, "register_comm", lambda comm: 0)
    # max_iters matching fake_native's three residual-history entries.
    monkeypatch.setattr(
        sharding_module.amgx_config,
        "prepare_config",
        lambda *args, **kwargs: '{"solver": {"max_iters": 2}}',
    )
    normalization_calls = []

    def fake_to_bcsr(A, **kwargs):
        normalization_calls.append(A)
        return A

    monkeypatch.setattr(sharding_module, "to_bcsr_matrix", fake_to_bcsr)
    transpose_calls = []

    def fake_transpose(*args, **kwargs):
        transpose_calls.append(A_local)
        return _single_rank_transpose_plan(A_local)

    monkeypatch.setattr(sharding_module, "build_transpose_plan", fake_transpose)

    # The services' native layer: rank-local arrays held in memory, and a
    # native solve returning x = values * rhs (+ x0) for this diagonal matrix
    # with a three-entry residual history.
    import jaxamg.jaxamg as jaxamg_module
    import jaxamg.local_arrays as local_arrays_module

    def fake_register(comm, array, device=None):
        return SimpleNamespace(value=np.asarray(array))

    def fake_load(handle, varying_axis=None):
        value = jnp.asarray(handle.value)
        if varying_axis is not None:
            value = jax.lax.pcast(value, varying_axis, to="varying")
        return value

    def fake_native(indptr, indices, values, rhs, x0, *args, use_x0=False, **kwargs):
        x = values * rhs
        if use_x0:
            x = x + x0
        stats = jnp.asarray([2.0, 1e-6, 0.0, 1.0, 0.1, 1e-6], dtype=rhs.dtype)
        return x, stats

    monkeypatch.setattr(local_arrays_module, "register_local", fake_register)
    monkeypatch.setattr(local_arrays_module, "load_local", fake_load)
    monkeypatch.setattr(jaxamg_module, "_amgx_solve_mpi_impl", fake_native)

    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm)
    # Omit mesh to exercise inference from b.sharding.
    solver = jaxamg.make_sharded_solver(matrix, b)
    return SimpleNamespace(
        mesh=mesh,
        b=b,
        A_local=A_local,
        comm=comm,
        matrix=matrix,
        solver=solver,
        allgather_calls=allgather_calls,
        normalization_calls=normalization_calls,
        halo_plan_calls=halo_plan_calls,
        transpose_calls=transpose_calls,
    )


def test_make_sharded_solver_preserves_global_array_contract(
    monkeypatch, mock_sharded_solver
):
    ctx = mock_sharded_solver
    mesh, b, A_local, comm = ctx.mesh, ctx.b, ctx.A_local, ctx.comm
    matrix, solver = ctx.matrix, ctx.solver
    allgather_calls, normalization_calls = ctx.allgather_calls, ctx.normalization_calls
    halo_plan_calls, transpose_calls = ctx.halo_plan_calls, ctx.transpose_calls
    # Only the row counts (the partition offsets) are gathered; nonzero counts
    # and the null-space schema are agreed by O(1) reductions.
    assert len(allgather_calls) == 1
    assert len(normalization_calls) == 1
    assert len(halo_plan_calls) == 1
    x, info = solver(b)
    np.testing.assert_array_equal(np.asarray(x), np.asarray(b))
    np.testing.assert_array_equal(np.asarray(solver.local_vector(x)), np.asarray(b))
    assert solver.global_size == 4
    assert solver.local_size == 4
    np.testing.assert_array_equal(np.asarray(info["iterations"]), [2])
    assert info["residual_history"].shape == (1, 3)

    x_jit, _ = jax.jit(lambda matrix_data, rhs: solver(rhs, A=matrix_data))(
        matrix.data, b
    )
    np.testing.assert_array_equal(np.asarray(x_jit), np.asarray(b))

    # Multiple RHS use the same public batching path as the ordinary solver:
    # vmap a solver that accepts one vector at a time.
    batched_b = jnp.stack((b, 2 * b))
    batched_x = jax.vmap(lambda rhs: solver(rhs, A=matrix.data)[0])(batched_b)
    np.testing.assert_array_equal(np.asarray(batched_x), np.asarray(batched_b))

    with pytest.raises(ValueError, match="A must be passed explicitly"):
        jax.jit(solver).lower(b)

    # A traced x0 with a concrete RHS must not embed the cached matrix values.
    with pytest.raises(ValueError, match="A must be passed explicitly"):
        jax.jit(lambda guess: solver(b, guess)).lower(b)

    x_updated, _ = solver(b, A=2 * matrix.data)
    np.testing.assert_array_equal(np.asarray(x_updated), 2 * np.asarray(b))

    x_warm, _ = solver(b, b)
    np.testing.assert_array_equal(np.asarray(x_warm), 2 * np.asarray(b))

    with jax.set_mesh(mesh):
        grad_b = jax.grad(
            lambda data, rhs: jnp.sum(solver(rhs, A=data)[0] ** 2),
            argnums=1,
        )(matrix.data, b)
        grad_A_data = jax.grad(lambda data: jnp.sum(solver(b, A=data)[0] ** 2))(
            matrix.data
        )
        grad_b_warm, grad_x0 = jax.grad(
            lambda data, rhs, x0: jnp.sum(solver(rhs, x0, A=data)[0] ** 2),
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

    # A matrix with traced values through ``A=``: d/dscale = sum(dL/dA.data
    # * data), with dL/dA.data = -2 b**2 as asserted above.
    def scaled_matrix(scale):
        return jsp.BCSR(
            (scale * A_local.data, A_local.indices, A_local.indptr),
            shape=A_local.shape,
        )

    def scaled_loss(scale):
        return jnp.sum(solver(b, A=scaled_matrix(scale))[0] ** 2)

    with jax.set_mesh(mesh):
        grad_scale = jax.grad(scaled_loss)(1.0)
        grad_scale_jit = jax.jit(jax.grad(scaled_loss))(1.0)
        x_scaled, _ = solver(b, A=scaled_matrix(3.0))
    expected = -2 * float(jnp.sum(b**2))
    assert float(grad_scale) == pytest.approx(expected)
    assert float(grad_scale_jit) == pytest.approx(expected)
    np.testing.assert_array_equal(np.asarray(x_scaled), 3 * np.asarray(b))

    with pytest.raises(ValueError, match="sparsity structure"):
        solver(b, A=jsp.BCSR.fromdense(jnp.ones((4, 4), dtype=jnp.float32)))
    # Same nnz, different column indices.
    with pytest.raises(ValueError, match="sparsity structure"):
        solver(b, A=jsp.BCSR.fromdense(jnp.fliplr(jnp.eye(4, dtype=jnp.float32))))
    # Same CSR arrays, different declared shape.
    with pytest.raises(ValueError, match="local shape"):
        solver(
            b, A=jsp.BCSR((A_local.data, A_local.indices, A_local.indptr), shape=(4, 8))
        )
    with pytest.raises(ValueError, match="concrete"):
        jax.jit(
            lambda idx, rhs: solver(
                rhs, A=jsp.BCSR((A_local.data, idx, A_local.indptr), shape=(4, 4))
            )[0]
        )(A_local.indices, b)

    # Stats: written after a direct call, rejected under a transform, and a
    # solver created without save_stats warns about incomplete output.
    stats_calls = []
    monkeypatch.setattr(
        sharding_module,
        "_capture_and_save_stats",
        lambda path, comm=None: stats_calls.append(path),
    )
    stats_solver = jaxamg.make_sharded_solver(matrix, b, save_stats=True)
    stats_solver(b, save_stats_file="stats.txt")
    assert stats_calls == ["stats.txt"]

    with pytest.raises(ValueError, match="direct solver call"):
        jax.jit(
            lambda rhs: stats_solver(rhs, A=matrix.data, save_stats_file="s.txt")
        ).lower(b)

    with pytest.warns(UserWarning, match="save_stats=True"):
        solver(b, save_stats_file="warned.txt")
    assert stats_calls == ["stats.txt", "warned.txt"]


@pytest.mark.parametrize("bad", [-2, 2, 99, 2**32])
def test_sharded_labels_reject_invalid_values(mock_sharded_solver, bad):
    ctx = mock_sharded_solver
    # Host inputs must be checked before JAX can narrow int64 to int32.
    with pytest.raises(ValueError, match=r"\[-1, 2\)"):
        ctx.solver(
            ctx.b,
            nullspace="constant",
            labels=(2, np.array([0, 0, bad, bad], np.int64)),
        )
    with jax.enable_x64(True):
        rows = jax.device_put(jnp.array([0, 0, bad, bad], jnp.int64), ctx.b.sharding)
        with pytest.raises(ValueError, match=r"\[-1, 2\)"):
            ctx.solver(
                ctx.b,
                nullspace="constant",
                transpose_nullspace="constant",
                labels=(2, rows),
            )


@pytest.mark.parametrize("custom_sum", [False, True])
def test_sharded_label_scaling_eager_jit_and_grad(mock_sharded_solver, custom_sum):
    ctx = mock_sharded_solver
    rows = jax.device_put(jnp.array([0, 0, 1, 1]), ctx.b.sharding)
    basis = jax.device_put(jnp.array([1e30, 1e30, 1e-30, 1e-30]), ctx.b.sharding)
    declared = {"label_sum": lambda sums: sums} if custom_sum else {}

    def solve(rhs, data, B, labels):
        return ctx.solver(
            rhs,
            A=data,
            nullspace=B,
            transpose_nullspace=B,
            labels=(3, labels),
            **declared,
        )[0]

    expected = np.array([-0.5, 0.5, -0.5, 0.5])
    with jax.set_mesh(ctx.mesh):
        args = (ctx.b, ctx.matrix.data, basis, rows)
        np.testing.assert_allclose(solve(*args), expected, atol=1e-6)
        np.testing.assert_allclose(jax.jit(solve)(*args), expected, atol=1e-6)
        gradient = jax.jit(
            jax.grad(lambda rhs, data, B, r, w: jnp.sum(w * solve(rhs, data, B, r)))
        )(*args, ctx.b)
        np.testing.assert_allclose(gradient, expected, atol=1e-6)
        # Excluded rows and an empty label still retain their exact semantics.
        excluded = jax.device_put(jnp.array([0, 0, -1, -1]), ctx.b.sharding)
        np.testing.assert_allclose(
            solve(ctx.b, ctx.matrix.data, basis, excluded), [-0.5, 0.5, 2, 3], atol=1e-6
        )


def test_sharded_matrix_validates_partition(monkeypatch):
    mesh, b = _single_device_array(np.ones(4, dtype=np.float32))
    monkeypatch.setattr(sharding_module, "_validate_runtime", lambda *args: None)
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        Get_rank=lambda: 0,
        allgather=lambda value: [value],
        allreduce=lambda value, op=None: value,
        Allreduce=lambda *args, **kwargs: None,
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


def test_shard_autotuning_flag(monkeypatch):
    monkeypatch.setattr(sharding_module, "_backends_are_initialized", lambda: False)

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_autotune_level=2")
    assert sharding_module._disable_shard_autotuning()
    assert os.environ["XLA_FLAGS"] == (
        "--xla_gpu_autotune_level=2 --xla_gpu_shard_autotuning=false"
    )

    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_shard_autotuning=true")
    assert not sharding_module._disable_shard_autotuning()
    assert os.environ["XLA_FLAGS"] == "--xla_gpu_shard_autotuning=true"

    monkeypatch.setattr(sharding_module, "_backends_are_initialized", lambda: True)
    monkeypatch.delenv("XLA_FLAGS")
    assert not sharding_module._disable_shard_autotuning()
    assert "XLA_FLAGS" not in os.environ


def test_shard_autotuning_warning(monkeypatch):
    monkeypatch.setattr(jax, "process_count", lambda: 2)

    monkeypatch.setattr(sharding_module, "_SHARD_AUTOTUNING_DISABLED", False)
    with pytest.warns(UserWarning, match="sharded autotuning"):
        sharding_module._check_shard_autotuning()

    monkeypatch.setattr(sharding_module, "_SHARD_AUTOTUNING_DISABLED", True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sharding_module._check_shard_autotuning()


def test_make_sharded_vector_normalizes_dtype_and_rejects_empty_ranks():
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        allgather=lambda value: [value],
        allreduce=lambda value, op=None: value,
        Allreduce=lambda *args, **kwargs: None,
    )

    b = jaxamg.make_sharded_vector(np.arange(4), comm=comm)
    assert b.dtype == jnp.float32

    with pytest.raises(ValueError, match="at least one row"):
        jaxamg.make_sharded_vector(np.zeros(0, dtype=np.float32), comm=comm)


def test_sharded_matrix_keeps_nullspace_bases(monkeypatch):
    mesh, b = _single_device_array(np.ones(4, dtype=np.float32))
    monkeypatch.setattr(sharding_module, "_validate_runtime", lambda *args: None)
    comm = SimpleNamespace(
        Get_size=lambda: 1,
        Get_rank=lambda: 0,
        allgather=lambda value: [value],
        allreduce=lambda value, op=None: value,
    )
    A_local = jaxamg.with_cache(
        jsp.BCSR.fromdense(jnp.eye(4, dtype=jnp.float32)), nullspace="constant"
    )

    matrix = jaxamg.make_sharded_matrix(A_local, b, comm=comm, mesh=mesh)

    np.testing.assert_array_equal(np.asarray(matrix._nullspace), np.ones((4, 1)))
    assert matrix._nullspace.dtype == jnp.float32
    assert matrix._transpose_nullspace is None


def test_distributed_basis_allows_more_columns_than_local_rows():
    from jaxamg.nullspace import as_nullspace_basis

    basis = np.ones((1, 2), dtype=np.float32)
    assert as_nullspace_basis(basis, 1, jnp.float32, "nullspace", 4).shape == (1, 2)
    with pytest.raises(ValueError, match="k ≤ 1"):
        as_nullspace_basis(basis, 1, jnp.float32, "nullspace")
    with pytest.raises(ValueError, match="k ≤ 1"):
        as_nullspace_basis(basis, 1, jnp.float32, "nullspace", 1)


def test_row_axes_of_a_multi_axis_mesh():
    """Rows may be partitioned over axes of an application's mesh, in mesh
    order; every other axis has size one; one-name tuples normalize."""
    from jaxamg.sharding import _axis_spec, _row_axes, _row_count

    mesh = jax.make_mesh((1, 1, 1), ("x", "y", "z"))
    assert _row_axes(mesh, ("x", "y")) == ("x", "y") and _row_count(mesh, "y") == 1
    assert _axis_spec(("x",)) == "x" and _axis_spec(("x", "y")) == ("x", "y")
    with pytest.raises(ValueError, match="mesh axis order"):
        _row_axes(mesh, ("y", "x"))
    with pytest.raises(ValueError, match="distinct axes"):
        _row_axes(mesh, ("x", "w"))
    with pytest.raises(ValueError, match="axis name"):
        _axis_spec(())
