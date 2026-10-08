"""Distributed column colouring for global-view operators.

Two columns conflict when some row holds both. Each rank sends its rows'
column sets to the columns' owners; the owners colour their columns by
Jones–Plassmann (hashed priorities), exchanging boundary colours with fixed
peers each round. Ghost columns receive their owners' colours and the result
is validated collectively. No global graph is replicated.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from mpi4py.MPI import Comm


def _priority(ids: np.ndarray, seed: int) -> np.ndarray:
    """splitmix64 of the global id: a rank-independent pseudo-random priority."""
    with np.errstate(over="ignore"):
        z = ids.astype(np.uint64) + np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


# Bounds gather/sort workspace, not the size of the caller's CSR structure.
_BATCH_ENTRIES = 262144


def _ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Flatten ragged integer ranges without a Python loop per row."""
    ends = np.cumsum(lengths)
    return np.repeat(starts - (ends - lengths), lengths) + np.arange(
        int(ends[-1]) if ends.size else 0, dtype=np.int64
    )


def _take_rows(indices, indptr, rows):
    lengths = indptr[rows + 1] - indptr[rows]
    return indices[_ranges(indptr[rows], lengths)], np.r_[0, np.cumsum(lengths)]


def _canonical_rows(indices, indptr):
    """Sort/deduplicate within rows, preserving empty rows and 64-bit IDs."""
    if not indices.size:
        return indices, indptr
    boundary = np.zeros(indices.size, dtype=bool)
    boundary[indptr[:-1][np.diff(indptr) > 0]] = True
    if np.any((indices[1:] < indices[:-1]) & ~boundary[1:]):
        rows = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
        indices = indices[np.lexsort((indices, rows))]
    keep = boundary
    keep[1:] |= indices[1:] != indices[:-1]
    if keep.all():
        return indices, indptr
    counts = np.r_[0, np.cumsum(keep)]
    return indices[keep], counts[indptr]


def _row_batches(indptr):
    """Contiguous whole-row batches; a single long row remains indivisible."""
    start = 0
    while start < indptr.size - 1:
        end = max(
            start + 1,
            int(np.searchsorted(indptr, indptr[start] + _BATCH_ENTRIES, side="right"))
            - 1,
        )
        end = min(end, indptr.size - 1)
        yield start, end
        start = end


def _exchange_rows(indices, indptr, offsets, comm):
    """Send each row once to each owner it touches, as packed CSR arrays."""
    from .transport import owners_of, sparse_exchange

    rank = comm.Get_rank()
    owner = owners_of(indices, offsets)
    # Sorted rows imply sorted owners within each row.
    starts = np.zeros(indices.size, dtype=bool)
    if indices.size:
        starts[0] = True
        starts[1:] = owner[1:] != owner[:-1]
        starts[indptr[:-1][np.diff(indptr) > 0]] = True
    at = np.flatnonzero(starts)
    rows = np.searchsorted(indptr, at, side="right") - 1
    destinations = owner[at]
    order = np.argsort(destinations, kind="stable")
    destinations, rows = destinations[order], rows[order]
    peers, firsts, counts = np.unique(
        destinations, return_index=True, return_counts=True
    )
    outgoing = {}
    local = (np.empty(0, np.int64), np.zeros(1, np.int64))
    for q, first, count in zip(peers, firsts, counts):
        ids, ptr = _take_rows(indices, indptr, rows[first : first + count])
        if q == rank:
            local = ids, ptr
        else:
            outgoing[int(q)] = np.concatenate(([ptr.size - 1], np.diff(ptr), ids))
    received = sparse_exchange(comm, outgoing)
    parts, sizes = [local[0]], [np.diff(local[1])]
    for buf in received.values():
        n = int(buf[0])
        sizes.append(buf[1 : n + 1])
        parts.append(buf[n + 1 :])
    return np.concatenate(parts), np.r_[0, np.cumsum(np.concatenate(sizes))]


def _slots(ids, first, last, ghosts):
    own = (ids >= first) & (ids < last)
    slots = ids - first
    slots[~own] = last - first + np.searchsorted(ghosts, ids[~own])
    return slots


class _Coloring:
    """Row priority chains and column incidences, without a conflict graph."""

    def __init__(self, indices, indptr, n_owned, ghosts, first, seed):
        self.n_owned = n_owned
        self.indptr = indptr
        # Only the relative priorities within each row matter. Sort in bounded
        # batches rather than form a global permutation over all incidences.
        self.columns = np.empty(indices.size, dtype=np.int64)
        for a, b in _row_batches(indptr):
            lo, hi = indptr[a], indptr[b]
            ids = indices[lo:hi]
            rows = np.repeat(np.arange(b - a), np.diff(indptr[a : b + 1]))
            order = np.lexsort((~ids, ~_priority(ids, seed), rows))
            self.columns[lo:hi] = _slots(ids[order], first, first + n_owned, ghosts)
        self.rows = np.repeat(np.arange(indptr.size - 1), np.diff(indptr))
        n_slots = n_owned + ghosts.size
        self.color = np.full(n_slots, -1, dtype=np.int64)
        self.by_column = np.argsort(self.columns, kind="stable")
        self.column_ptr = np.r_[
            0, np.cumsum(np.bincount(self.columns, minlength=n_slots))
        ]
        # Each incidence waits only for its predecessor in the row's priority
        # chain. Once that predecessor is coloured, all earlier ones are too.
        waiting = np.ones(self.columns.size, dtype=bool)
        waiting[indptr[:-1][np.diff(indptr) > 0]] = False
        self.pending = np.bincount(self.columns[waiting], minlength=n_slots)
        self.row_mask = np.zeros(indptr.size - 1, dtype=np.uint64)

    def incidences(self, columns):
        ptr = self.column_ptr
        return self.by_column[_ranges(ptr[columns], ptr[columns + 1] - ptr[columns])]

    def assign(self, ready):
        """Lowest available colour; one-word fast path, sparse overflow path."""
        ptr = self.column_ptr
        lengths = ptr[ready + 1] - ptr[ready]
        incidence = self.incidences(ready)
        mask = np.zeros(ready.size, dtype=np.uint64)
        nonempty = lengths > 0
        starts = np.r_[0, np.cumsum(lengths)]
        mask[nonempty] = np.bitwise_or.reduceat(
            self.row_mask[self.rows[incidence]], starts[:-1][nonempty]
        )
        # A power of two is exactly representable even after conversion to
        # float64. frexp finds the first zero bit without 64 separate tests.
        with np.errstate(over="ignore"):
            bit = ~mask & (mask + np.uint64(1))
        chosen = (np.frexp(bit.astype(np.float64))[1] - 1).astype(np.int64)
        overflow = np.flatnonzero(bit == 0)
        if overflow.size:
            # Ready vertices never share a row. This gather is at most the
            # incident row storage, not the row-degree-squared conflict graph.
            selected = ready[overflow]
            incident = self.incidences(selected)
            edge = self.rows[incident]
            counts = self.indptr[edge + 1] - self.indptr[edge]
            group = np.repeat(
                np.arange(selected.size), ptr[selected + 1] - ptr[selected]
            )
            group = np.repeat(group, counts)
            used = self.color[self.columns[_ranges(self.indptr[edge], counts)]]
            keep = used >= 64
            group, used = group[keep], used[keep]
            order = np.lexsort((used, group))
            group, used = group[order], used[order]
            unique = np.ones(used.size, dtype=bool)
            unique[1:] = (group[1:] != group[:-1]) | (used[1:] != used[:-1])
            group, used = group[unique], used[unique]
            totals = np.bincount(group, minlength=selected.size)
            begin = np.r_[0, np.cumsum(totals[:-1])]
            position = np.arange(used.size) - begin[group]
            missing = used != position + 64
            mex = totals + 64
            np.minimum.at(mex, group[missing], position[missing] + 64)
            chosen[overflow] = mex
        return chosen

    def publish(self, columns, colors):
        """Apply new authoritative colours and return newly ready owned columns."""
        self.color[columns] = colors
        incidence = self.incidences(columns)
        rows = self.rows[incidence]
        values = self.color[self.columns[incidence]]
        low = values < 64
        np.bitwise_or.at(
            self.row_mask,
            rows[low],
            np.left_shift(np.uint64(1), values[low].astype(np.uint64)),
        )
        has_next = incidence + 1 < self.indptr[rows + 1]
        successors = self.columns[incidence[has_next] + 1]
        np.subtract.at(self.pending, successors, 1)
        candidates = np.unique(successors[successors < self.n_owned])
        return candidates[
            (self.pending[candidates] == 0) & (self.color[candidates] < 0)
        ]

    def complete_local(self, ready):
        while ready.size:
            # Bound the fast-path gather by the number of incidences.
            lengths = self.column_ptr[ready + 1] - self.column_ptr[ready]
            ptr = np.r_[0, np.cumsum(lengths)]
            next_ready = []
            for a, b in _row_batches(ptr):
                current = ready[a:b]
                next_ready.append(self.publish(current, self.assign(current)))
            ready = np.unique(np.concatenate(next_ready))


def _validate_coloring(indptr, colors):
    """No duplicate colour among a row's distinct columns; bounded workspace."""
    for a, b in _row_batches(indptr):
        lo, hi = indptr[a], indptr[b]
        # Caller supplies colours corresponding to canonical indices.
        rows = np.repeat(np.arange(b - a), np.diff(indptr[a : b + 1]))
        cs = colors[lo:hi]
        order = np.lexsort((cs, rows))
        r, c = rows[order], cs[order]
        if np.any(c < 0) or np.any((r[1:] == r[:-1]) & (c[1:] == c[:-1])):
            return False
    return True


def _validated_csr(indices, indptr, row_counts, comm):
    """Refuse malformed local input together, before any peer exchange."""
    from mpi4py import MPI

    try:
        indices, indptr = np.asarray(indices), np.asarray(indptr)
        counts = np.asarray(row_counts)
        if any(
            a.ndim != 1 or (a.size and a.dtype.kind not in "iu")
            for a in (indices, indptr, counts)
        ):
            raise ValueError
        if counts.size != comm.Get_size() or np.any(counts < 0):
            raise ValueError
        total = sum(int(n) for n in counts)
        if total > np.iinfo(np.int64).max:
            raise ValueError
        indices = indices.astype(np.int64, copy=False)
        indptr = indptr.astype(np.int64, copy=False)
        offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        valid = (
            indptr.size == counts[comm.Get_rank()] + 1
            and indptr.size > 0
            and indptr[0] == 0
            and indptr[-1] == indices.size
            and np.all(indptr[1:] >= indptr[:-1])
            and np.all((indices >= 0) & (indices < total))
        )
    except (TypeError, ValueError, OverflowError):
        valid = False
    if comm.allreduce(not valid, op=MPI.LOR):
        raise ValueError("invalid CSR pattern or row partition in distributed_coloring")
    return indices, indptr, offsets


def distributed_coloring(
    indices: np.ndarray,
    indptr: np.ndarray,
    row_counts: tuple[int, ...],
    comm: Comm,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Collectively colour CSR columns, returning owned and per-entry colours.

    Priorities depend only on global column IDs and seed. Only row incidences
    and ghost columns are stored; no global or expanded column-conflict graph
    is built. New boundary colours travel only to ranks that requested them.
    """
    from mpi4py import MPI

    from .transport import owners_of, sparse_exchange, transport_comm

    tcomm = transport_comm(comm)
    rank = comm.Get_rank()
    original, ptr, offsets = _validated_csr(indices, indptr, row_counts, tcomm)
    first, last = int(offsets[rank]), int(offsets[rank + 1])
    indices, indptr = _canonical_rows(original, ptr)
    edges, edge_ptr = _exchange_rows(indices, indptr, offsets, comm)
    remote = np.concatenate(
        (
            edges[(edges < first) | (edges >= last)],
            indices[(indices < first) | (indices >= last)],
        )
    )
    ghosts = np.unique(remote)
    del remote
    owners = owners_of(ghosts, offsets)
    requests = {int(q): ghosts[owners == q] for q in np.unique(owners)}
    asked = sparse_exchange(comm, requests)
    send_to = {q: ids - first for q, ids in asked.items()}
    recv_at = {q: last - first + np.flatnonzero(owners == q) for q in requests}
    state = _Coloring(edges, edge_ptr, last - first, ghosts, first, seed)
    del edges
    announced = {q: np.zeros(ids.size, dtype=bool) for q, ids in send_to.items()}
    ready = np.flatnonzero(state.pending[: last - first] == 0)
    while True:
        state.complete_local(ready)
        sends = []
        for q, slots in send_to.items():
            fresh = (state.color[slots] >= 0) & ~announced[q]
            announced[q] |= fresh
            # Positions are in the receiver's fixed request list, not global IDs.
            payload = np.stack((np.flatnonzero(fresh), state.color[slots[fresh]]), 1)
            sends.append(tcomm.isend(payload, dest=q, tag=7300))
        released = []
        for q, slots in recv_at.items():
            payload = tcomm.recv(source=q, tag=7300)
            if payload.size:
                released.append(state.publish(slots[payload[:, 0]], payload[:, 1]))
        MPI.Request.Waitall(sends)
        if (
            tcomm.allreduce(bool(np.any(state.color[: last - first] < 0)), op=MPI.LOR)
            == 0
        ):
            break
        ready = (
            np.unique(np.concatenate(released)) if released else np.empty(0, np.int64)
        )
    canonical_colors = state.color[_slots(indices, first, last, ghosts)]
    bad = not _validate_coloring(indptr, canonical_colors)
    if tcomm.allreduce(bad, op=MPI.LOR):
        raise RuntimeError("distributed colouring failed its post-validation")
    owned = state.color[: last - first]
    n_colors = int(tcomm.allreduce(int(owned.max(initial=-1)) + 1, op=MPI.MAX))
    entries = state.color[_slots(original, first, last, ghosts)]
    return owned.astype(np.int32), entries.astype(np.int32), n_colors
