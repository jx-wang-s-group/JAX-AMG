"""Distributed colouring (``jaxamg.distributed_coloring``), host-only under MPI:
``mpirun -np N pytest --only-mpi tests/test_distributed_coloring_mpi.py``.

Every result is checked independently on rank 0 from the gathered (small) test
patterns: owner-authoritative colours agree with every rank's entry colours,
the distinct columns of every row have distinct colours, and the count stays
within the greedy bound (largest conflict degree + 1)."""

import numpy as np
import pytest

pytestmark = pytest.mark.mpi(min_size=2)


def _brick(rank, size, m=6):
    """A 7-point 3D stencil on a brick decomposition (brick-major rows)."""
    from mpi4py import MPI

    px, py, pz = MPI.Compute_dims(size, 3)  # balanced: several partitioned axes
    bx, by, bz = rank // (py * pz), (rank // pz) % py, rank % pz
    ii, jj, kk = np.meshgrid(*(np.arange(m),) * 3, indexing="ij")
    X, Y, Z = (bx * m + ii).ravel(), (by * m + jj).ravel(), (bz * m + kk).ravel()
    rows, cols = [], []
    for d in (
        (0, 0, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ):
        x, y, z = X + d[0], Y + d[1], Z + d[2]
        ok = (x >= 0) & (x < px * m) & (y >= 0) & (y < py * m) & (z >= 0) & (z < pz * m)
        owner = ((x // m) * py + y // m) * pz + z // m
        rows.append(np.nonzero(ok)[0])
        cols.append((owner * m**3 + ((x % m) * m + y % m) * m + z % m)[ok])
    return np.concatenate(rows), np.concatenate(cols), tuple([m**3] * size)


def _irregular(rank, size):
    """Unequal row counts; each row: its diagonal, two near and two far random
    columns, some owned by other ranks (ghost-to-ghost conflicts)."""
    counts = tuple(20 + 7 * r for r in range(size))
    n = sum(counts)
    first = sum(counts[:rank])
    rng = np.random.default_rng(100 + rank)
    rows, cols = [], []
    for k in range(counts[rank]):
        g = first + k
        picks = {g, (g + 1) % n, (g + 5) % n, *rng.integers(0, n, 2).tolist()}
        rows += [k] * len(picks)
        cols += sorted(picks)
    return np.asarray(rows), np.asarray(cols), counts


def _csr(rows, cols, n_local):
    order = np.lexsort((cols, rows))
    rows, cols = rows[order], cols[order]
    indptr = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=n_local))))
    return cols.astype(np.int64), indptr.astype(np.int64)


@pytest.mark.parametrize("family", ["brick", "irregular"])
def test_distributed_coloring_is_valid_and_owner_consistent(family):
    from mpi4py import MPI

    from jaxamg.distributed_coloring import distributed_coloring

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rows, cols, counts = (_brick if family == "brick" else _irregular)(rank, size)
    indices, indptr = _csr(rows, cols, counts[rank])
    owned, entries, n_colors = distributed_coloring(
        indices, indptr, counts, comm, seed=3
    )
    assert owned.shape == (counts[rank],) and entries.shape == indices.shape
    gathered = comm.gather((indices, indptr, owned, entries), root=0)
    if rank == 0:
        truth = np.concatenate([g[2] for g in gathered])  # owner colours
        adjacency: dict[int, set] = {}
        for ind, ptr, _, ent in gathered:
            np.testing.assert_array_equal(ent, truth[ind])  # owner-authoritative
            for r in range(len(ptr) - 1):
                row = np.unique(ind[ptr[r] : ptr[r + 1]])
                assert len(np.unique(truth[row])) == len(row)
                for c in row:
                    adjacency.setdefault(int(c), set()).update(int(x) for x in row)
        max_degree = max(len(v) - 1 for v in adjacency.values())
        assert n_colors == truth.max() + 1 <= max_degree + 1
    comm.Barrier()


def _serial_greedy(parts, n, seed, priority):
    """Independent small-graph oracle; deliberately expands conflicts in tests."""
    neighbors = [set() for _ in range(n)]
    for indices, indptr in parts:
        for a, b in zip(indptr[:-1], indptr[1:]):
            columns = set(indices[a:b].tolist())
            for c in columns:
                neighbors[c].update(columns - {c})
    ids = np.arange(n)
    colors = np.full(n, -1, np.int32)
    for c in np.lexsort((ids, priority(ids, seed)))[::-1]:
        used = {colors[j] for j in neighbors[c]}
        value = 0
        while value in used:
            value += 1
        colors[c] = value
    return colors


@pytest.mark.parametrize(
    "family", ["irregular", "duplicates", "dense", "empty", "zero_rank"]
)
@pytest.mark.parametrize("ties", [False, True])
def test_coloring_matches_serial_priority_greedy(family, ties, monkeypatch):
    from mpi4py import MPI

    import jaxamg.distributed_coloring as module

    comm = MPI.COMM_WORLD
    if ties:
        monkeypatch.setattr(
            module, "_priority", lambda ids, seed: np.zeros(ids.size, np.uint64)
        )
    if family in ("irregular", "duplicates"):
        rows, cols, counts = _irregular(comm.rank, comm.size)
        indices, indptr = _csr(rows, cols, counts[comm.rank])
        if family == "duplicates":
            # Reverse each row and repeat entries; output must retain input order.
            pieces = [
                np.repeat(indices[a:b][::-1], 2)
                for a, b in zip(indptr[:-1], indptr[1:])
            ]
            indices = np.concatenate(pieces)
            indptr = indptr * 2
    else:
        counts = tuple(
            0 if family == "zero_rank" and r == 0 else 40 for r in range(comm.size)
        )
        if family in ("dense", "zero_rank") and counts[comm.rank]:
            indices = np.arange(sum(counts), dtype=np.int64)
            indptr = np.r_[0, np.full(counts[comm.rank], indices.size)]
        else:
            indices = np.empty(0, np.int64)
            indptr = np.zeros(counts[comm.rank] + 1, np.int64)
    owned, entries, n_colors = module.distributed_coloring(
        indices, indptr, counts, comm, seed=9
    )
    parts = comm.allgather((indices, indptr))
    expected = _serial_greedy(parts, sum(counts), 9, module._priority)
    first = sum(counts[: comm.rank])
    np.testing.assert_array_equal(owned, expected[first : first + counts[comm.rank]])
    np.testing.assert_array_equal(entries, expected[indices])
    assert n_colors == int(expected.max(initial=-1)) + 1
    if family == "dense":
        assert n_colors > 64


def test_small_workspace_does_not_change_coloring(monkeypatch):
    from mpi4py import MPI

    import jaxamg.distributed_coloring as module

    comm = MPI.COMM_WORLD
    rows, cols, counts = _irregular(comm.rank, comm.size)
    indices, indptr = _csr(rows, cols, counts[comm.rank])
    expected = module.distributed_coloring(indices, indptr, counts, comm)
    monkeypatch.setattr(module, "_BATCH_ENTRIES", 7)
    actual = module.distributed_coloring(indices, indptr, counts, comm)
    for result, truth in zip(actual, expected):
        np.testing.assert_array_equal(result, truth)


@pytest.mark.parametrize("bad", ["columns", "pointers", "counts"])
def test_invalid_local_input_is_refused_collectively(bad):
    from mpi4py import MPI

    from jaxamg.distributed_coloring import distributed_coloring

    comm = MPI.COMM_WORLD
    indices = np.array([comm.rank], np.int64)
    indptr = np.array([0, 1], np.int64)
    counts = (1,) * comm.size
    if comm.rank == 0:
        if bad == "columns":
            indices[0] = -1
        elif bad == "pointers":
            indptr[1] = 2
        else:
            counts = ()
    with pytest.raises(ValueError, match="invalid CSR"):
        distributed_coloring(indices, indptr, counts, comm)


def test_dependency_storage_tracks_incidences_not_column_pairs():
    from jaxamg.distributed_coloring import _canonical_rows, _Coloring

    width = 4096
    ids = np.arange(2**33, 2**33 + width, dtype=np.int64)
    indices, indptr = _canonical_rows(ids[::-1], np.array([0, width]))
    np.testing.assert_array_equal(indices, ids)
    state = _Coloring(indices, indptr, width, np.empty(0, np.int64), 2**33, 0)
    stored = sum(a.nbytes for a in vars(state).values() if isinstance(a, np.ndarray))
    assert stored < 100 * width
