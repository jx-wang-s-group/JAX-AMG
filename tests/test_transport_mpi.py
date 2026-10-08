"""Neighbour-only transport (``jaxamg.transport``) under MPI: peer discovery,
the exchange and its declared transpose, halo gathers, padding, batching and
ordering against mpi4jax collectives. Run with ``mpirun -np N pytest --only-mpi``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.mpi(min_size=2)]


@pytest.fixture
def comm():
    from mpi4py import MPI

    return MPI.COMM_WORLD


def _peers(rank, size):
    """An irregular, nonsymmetric communication graph: rank r sends to r+1 and
    r+3 (mod size) with rank-dependent counts, never to itself."""
    out = {}
    for step, base in ((1, 2), (3, 1)):
        peer = (rank + step) % size
        if peer != rank:
            out[peer] = out.get(peer, 0) + base + rank % 3
    return out


def _message(source, dest, count):
    return 1000.0 * source + 10.0 * dest + np.arange(count)


def _plan_and_reference(comm):
    from jaxamg.transport import make_plan, sparse_exchange

    rank, size = comm.Get_rank(), comm.Get_size()
    sends = _peers(rank, size)
    counts = sparse_exchange(
        comm, {p: np.array([c]) for p, c in sends.items()}, dtype=np.int64
    )
    recvs = {s: int(v[0]) for s, v in counts.items()}
    # The discovered sources are exactly the ranks whose graph names this one.
    expected = {
        s: _peers(s, size)[rank] for s in range(size) if rank in _peers(s, size)
    }
    assert recvs == expected
    plan = make_plan(comm, sends, recvs)
    send_buf = np.concatenate(
        [_message(rank, p, sends[p]) for p in sorted(sends)]
    ).astype(np.float32)
    recv_ref = np.concatenate(
        [_message(s, rank, recvs[s]) for s in sorted(recvs)]
    ).astype(np.float32)
    return plan, send_buf, recv_ref


def test_exchange_forward_reverse_and_transpose(comm):
    from mpi4py import MPI

    from jaxamg.transport import exchange

    plan, send_buf, recv_ref = _plan_and_reference(comm)
    got = exchange(jnp.asarray(send_buf), plan)
    np.testing.assert_array_equal(got, recv_ref)
    # The reverse exchange returns every message to its sender.
    back = exchange(got, plan, reverse=True)
    np.testing.assert_array_equal(back, send_buf)
    # Jitted, with a padded output (zeros past the received values).
    padded = jax.jit(lambda v: exchange(v, plan, out_size=len(recv_ref) + 5))(
        jnp.asarray(send_buf)
    )
    np.testing.assert_array_equal(padded[: len(recv_ref)], recv_ref)
    np.testing.assert_array_equal(padded[len(recv_ref) :], 0)

    # Declared transpose, by the global dot-product test <E u, w> = <u, Eᵀ w>.
    rng = np.random.default_rng(comm.Get_rank())
    u = jnp.asarray(rng.standard_normal(plan.send_total), jnp.float32)
    w = jnp.asarray(rng.standard_normal(plan.recv_total), jnp.float32)
    _, vjp = jax.vjp(lambda v: exchange(v, plan), u)
    lhs = comm.allreduce(float(jnp.dot(exchange(u, plan), w)), op=MPI.SUM)
    rhs = comm.allreduce(float(jnp.dot(u, vjp(w)[0])), op=MPI.SUM)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5)
    # Forward mode applies the exchange to the tangent.
    _, tangent = jax.jvp(lambda v: exchange(v, plan), (u,), (2.0 * u,))
    np.testing.assert_allclose(tangent, 2.0 * exchange(u, plan), rtol=1e-6)


def test_exchange_batching_and_ordering(comm):
    import mpi4jax
    from mpi4py import MPI

    from jaxamg.transport import exchange

    plan, send_buf, recv_ref = _plan_and_reference(comm)
    batch = jnp.stack([jnp.asarray(send_buf), 2.0 * jnp.asarray(send_buf)])
    got = jax.vmap(lambda v: exchange(v, plan))(batch)
    np.testing.assert_array_equal(got, np.stack([recv_ref, 2.0 * recv_ref]))

    # Interleaved with mpi4jax collectives in one jitted program on per-rank
    # shapes: every rank issues them in program order.
    @jax.jit
    def program(v):
        total = mpi4jax.allreduce(jnp.sum(v), op=MPI.SUM, comm=comm)
        received = exchange(v * total, plan)
        again = mpi4jax.allreduce(jnp.sum(received), op=MPI.SUM, comm=comm)
        return exchange(received, plan, reverse=True) / total, again

    returned, again = program(jnp.asarray(send_buf))
    np.testing.assert_allclose(returned, send_buf, rtol=1e-6)
    total = comm.allreduce(float(send_buf.sum()), op=MPI.SUM)
    np.testing.assert_allclose(float(again), total * total, rtol=1e-5)


def test_halo_plan_gather_and_gradient(comm):
    from mpi4py import MPI

    from jaxamg.matrices import poisson_matrix_distributed
    from jaxamg.mpi_utils import build_halo_plan
    from jaxamg.transport import halo_gather

    rank, size = comm.Get_rank(), comm.Get_size()
    grid = 8
    A_local, row_start, row_end = poisson_matrix_distributed(grid, grid, rank, size)
    counts = tuple(comm.allgather(row_end - row_start))
    plan = build_halo_plan(A_local.indices, counts, (row_start, row_end), comm)
    assert plan.exchange.max_degree <= 2  # a 1D slab partition: at most two peers
    x_global = np.arange(grid * grid, dtype=np.float32) + 1.0
    x_local = jnp.asarray(x_global[row_start:row_end])
    combined = halo_gather(x_local, jnp.asarray(plan.send_ids), plan.exchange)
    cols = np.asarray(A_local.indices)
    np.testing.assert_array_equal(
        np.asarray(combined)[np.asarray(plan.col_to_combined)], x_global[cols]
    )

    # The gather's transpose returns ghost cotangents to their owners.
    def objective(v):
        c = halo_gather(v, jnp.asarray(plan.send_ids), plan.exchange)
        return jnp.sum(c[jnp.asarray(plan.col_to_combined)])

    grad = jax.grad(objective)(x_local)
    # d/dx_j of sum over every rank's entries of x[col] = number of entries of
    # column j globally.
    local_hits = np.bincount(cols, minlength=grid * grid).astype(np.float64)
    hits = np.empty_like(local_hits)
    comm.Allreduce(local_hits, hits, op=MPI.SUM)
    np.testing.assert_array_equal(grad, hits[row_start:row_end])

    # The same structure reuses its registered plan (no new registration).
    again = build_halo_plan(A_local.indices, counts, (row_start, row_end), comm)
    assert again.exchange.plan_id == plan.exchange.plan_id

    # Padded form (sharded layout): shapes at the global maxima.
    padded = build_halo_plan(
        A_local.indices, counts, (row_start, row_end), comm, pad=True
    )
    maxima = comm.allreduce(plan.n_ghost, op=MPI.MAX)
    assert padded.max_n_ghost == maxima
    combined_padded = halo_gather(
        x_local, jnp.asarray(padded.send_ids), padded.exchange, n_ghost=maxima
    )
    np.testing.assert_array_equal(combined_padded[: len(combined)], combined)
    np.testing.assert_array_equal(combined_padded[len(combined) :], 0)


def test_oversized_messages_are_refused_on_every_rank(comm):
    """MPI counts are C ints; a plan whose largest message (on any rank)
    exceeds 2 GiB for the buffer's dtype is refused before anything is sent,
    by every rank alike."""
    from jaxamg.transport import exchange, make_plan

    rank, size = comm.Get_rank(), comm.Get_size()
    big = 2**30 if rank == 0 else 1  # float32: 4 GiB, from rank 0 only
    plan = make_plan(comm, {(rank + 1) % size: big}, {(rank - 1) % size: 1})
    assert plan.max_count == 2**30
    with pytest.raises(ValueError, match="message limit"):
        exchange(jnp.zeros(1, jnp.float32), plan)
    comm.Barrier()


def test_transport_communicator_follows_its_communicator(comm):
    """The transport duplicate belongs to its communicator, not to a handle
    value: after the caller frees a communicator and MPI reuses its handle
    differently on each rank, every rank still makes the same decision."""
    from mpi4py import MPI

    from jaxamg.transport import transport_comm

    first = comm.Dup()
    old = transport_comm(first)
    assert transport_comm(first) is old
    first.Free()
    # Occupy the freed handle on one rank only.
    hold = MPI.COMM_SELF.Dup() if comm.Get_rank() == 1 else None
    second = comm.Dup()
    new = transport_comm(second)  # collective on every rank
    assert new is not old
    assert MPI.Comm.Compare(new, second) == MPI.CONGRUENT
    comm.Barrier()
    second.Free()
    if hold is not None:
        hold.Free()


def test_programs_own_registered_entries(comm):
    """A compiled program keeps the rank-local arrays and exchange plans it
    uses: it runs after every other reference and JAX's caches are dropped,
    including for data registered while tracing, and releases them with it."""
    import gc

    from jaxamg._ext import _amgx
    from jaxamg.deferred_release import wait_for_releases
    from jaxamg.local_arrays import load_local, register_local
    from jaxamg.transport import exchange

    rank = comm.Get_rank()

    def counts():
        return _amgx.local_array_count(), _amgx.exchange_plan_count()

    jax.clear_caches()  # earlier programs' entries
    gc.collect()
    wait_for_releases(30)
    before = counts()

    def compile_program():
        # Its plan and array are referenced only by the program returned.
        plan, send, expected = _plan_and_reference(comm)
        handle = register_local(comm, np.arange(4, dtype=np.float32) + 10 * rank)

        def program(x):
            return exchange(x, plan), load_local(handle)

        return jax.jit(program).lower(jnp.asarray(send)).compile(), send, expected

    compiled, send, expected = compile_program()
    jax.clear_caches()
    gc.collect()
    received, loaded = compiled(jnp.asarray(send))
    np.testing.assert_array_equal(np.asarray(received)[: len(expected)], expected)
    np.testing.assert_array_equal(np.asarray(loaded), np.arange(4) + 10 * rank)
    del compiled, received, loaded
    gc.collect()
    assert wait_for_releases(30)
    assert counts() == before

    @jax.jit
    def registered_while_tracing(x):
        return x + load_local(register_local(comm, np.full(4, rank, np.float32)))

    for _ in range(2):
        gc.collect()
        np.testing.assert_array_equal(
            np.asarray(registered_while_tracing(jnp.ones(4))), 1 + rank
        )
    del registered_while_tracing
    jax.clear_caches()
    gc.collect()
    assert wait_for_releases(30)
    assert counts() == before


def test_release_waits_for_queued_work(comm):
    """Dropping the last owner of a registered array while an execution that
    loads it is still queued defers the release until the device work has
    finished; the execution reads the intact data."""
    import gc

    from jaxamg._ext import _amgx
    from jaxamg.deferred_release import wait_for_releases
    from jaxamg.local_arrays import load_local, register_local

    rank = comm.Get_rank()
    jax.clear_caches()
    gc.collect()
    wait_for_releases(30)
    before = _amgx.local_array_count()

    def dispatch():
        handle = register_local(comm, np.arange(4, dtype=np.float32) + 10 * rank)

        def program(m):
            # About a second of device work before the load.
            m = jax.lax.fori_loop(0, 300, lambda _, a: (a @ a) / 4096.0, m)
            return load_local(handle) + 0 * m[0, :4]

        compiled = jax.jit(program).lower(jnp.ones((4096, 4096))).compile()
        return compiled(jnp.ones((4096, 4096)))  # asynchronous

    result = dispatch()  # the handle and the executable are dropped here
    jax.clear_caches()
    gc.collect()
    if _amgx.local_array_count() == before:
        # Released already: only allowed once the execution has finished (it
        # can, if collecting took longer than the device work).
        assert result.is_ready()
    else:
        assert _amgx.local_array_count() == before + 1
    np.testing.assert_array_equal(np.asarray(result), np.arange(4) + 10 * rank)
    assert wait_for_releases(30)
    assert _amgx.local_array_count() == before


def test_registries_release_unreferenced_entries(comm):
    """Registered rank-local arrays and exchange plans are released natively
    once nothing references their handles."""
    import gc

    from jaxamg._ext import _amgx
    from jaxamg.deferred_release import wait_for_releases
    from jaxamg.local_arrays import load_local, register_local
    from jaxamg.transport import make_plan

    def counts():
        return _amgx.local_array_count(), _amgx.exchange_plan_count()

    jax.clear_caches()
    gc.collect()
    wait_for_releases(30)
    before = counts()
    values = np.arange(8, dtype=np.int32) + comm.Get_rank()
    handle = register_local(comm, values)
    plan = make_plan(comm, {}, {})
    assert counts() == (before[0] + 1, before[1] + 1)
    np.testing.assert_array_equal(load_local(handle), values)
    del handle, plan
    gc.collect()
    assert wait_for_releases(30)
    assert counts() == before


@pytest.mark.parametrize("invalid", [{-1: 1}, {0: -1}, {0: 1.5}, {0: 2**64}])
def test_invalid_exchange_plan_is_rejected_collectively(comm, invalid):
    from jaxamg.transport import make_plan

    with pytest.raises(ValueError, match="invalid on at least one rank"):
        make_plan(comm, invalid if comm.rank == 0 else {}, {})


@pytest.mark.parametrize("kind", ["halo", "transpose"])
@pytest.mark.parametrize("bad", ["negative", "out_of_range", "fractional"])
def test_bad_plan_columns_are_rejected_collectively(comm, kind, bad):
    from jaxamg.mpi_utils import build_halo_plan, build_transpose_plan

    columns = np.array([comm.rank])
    if comm.rank == 0:
        columns = np.array(
            [{"negative": -1, "out_of_range": comm.size, "fractional": 0.5}[bad]]
        )
    counts = (1,) * comm.size
    part = (comm.rank, comm.rank + 1)
    with pytest.raises(ValueError, match="invalid CSR structure"):
        if kind == "halo":
            build_halo_plan(columns, counts, part, comm)
        else:
            build_transpose_plan(columns, np.array([0, 1]), counts, part, comm)


def test_bad_transpose_row_pointers_are_rejected_collectively(comm):
    from jaxamg.mpi_utils import build_transpose_plan

    pointers = np.array([0, 2] if comm.rank == 0 else [0, 1])
    with pytest.raises(ValueError, match="invalid CSR structure"):
        build_transpose_plan(
            np.array([comm.rank]),
            pointers,
            (1,) * comm.size,
            (comm.rank, comm.rank + 1),
            comm,
        )


def test_local_label_basis_failure_is_rejected_collectively(comm):
    from jaxamg.nullspace import validate_basis

    basis = jnp.asarray(np.random.default_rng(6).normal(size=(6, 2)), jnp.float32)
    if comm.rank == 0:
        basis = basis.at[:, 1].set(basis[:, 0])
    with pytest.raises(ValueError, match="dependent columns"):
        validate_basis(
            basis,
            "nullspace",
            comm=comm,
            labels=(1, np.zeros(6, np.int32)),
            label_sum=lambda x: x,
        )
