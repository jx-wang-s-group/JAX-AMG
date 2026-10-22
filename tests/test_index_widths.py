"""Large global indices with tiny local storage; no large allocations or GPU."""

import importlib
import importlib.machinery
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp


@pytest.mark.parametrize("kind", ["tridiagonal", "poisson", "random"])
def test_distributed_matrix_columns_past_int32(kind):
    from jaxamg import matrices

    # Many hypothetical ranks, one local row: only a few entries are stored.
    total = 2**31 + 8
    nranks = 2**31 - 1
    if kind == "tridiagonal":
        matrix, start, end = matrices.tridiagonal_matrix_distributed(
            total, nranks - 1, nranks
        )
        expected = [total - 2, total - 1]
    elif kind == "poisson":
        matrix, start, end = matrices.poisson_matrix_distributed(
            2, total // 2, nranks - 1, nranks
        )
        expected = [total - 3, total - 2, total - 1]
    else:
        matrix, start, end = matrices.random_matrix_distributed(
            total, nranks - 1, nranks, density=0
        )
        expected = [total - 1]
    assert (start, end) == (total - 1, total)
    assert matrix.indices.dtype == np.int64
    assert matrix.indptr.dtype == np.int32
    assert matrix.data.dtype == np.float32
    np.testing.assert_array_equal(matrix.indices, expected)


@pytest.mark.parametrize("wrong_size", [None, "nglobal", "columns"])
def test_direct_mpi_partition_past_int32(monkeypatch, mock_mpi, wrong_size):
    solver = importlib.import_module("jaxamg.jaxamg")
    monkeypatch.setattr(
        sys.modules["mpi4py"],
        "__spec__",
        importlib.machinery.ModuleSpec("mpi4py", None),
    )
    monkeypatch.setattr(mock_mpi, "LOR", object(), raising=False)
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(solver, "_ensure_backend", lambda: None)
    monkeypatch.setattr(solver, "HAS_MPI", True)
    monkeypatch.setattr(solver, "register_comm", lambda comm: 1)
    counts = (2**30, 2**30, 1)
    start, total = 2**31, 2**31 + 1
    comm = SimpleNamespace(
        Get_rank=lambda: 2,
        allgather=lambda n: [*counts[:-1], n],
        allreduce=lambda value, op: value,
    )
    local = sp.csr_matrix(
        (np.ones(1, np.float32), [start], [0, 1]),
        shape=(1, total + (wrong_size == "columns")),
    )

    class ReachedHaloPlan(Exception):
        pass

    def check_plan(indices, recvcounts, partition, comm):
        assert recvcounts == counts
        assert partition == (start, total)
        np.testing.assert_array_equal(indices, [start])
        raise ReachedHaloPlan

    monkeypatch.setattr(solver, "build_halo_plan", check_plan)
    error = ReachedHaloPlan if wrong_size is None else ValueError
    match = None if wrong_size is None else "must match the sum"
    with pytest.raises(error, match=match):
        solver.solve(
            local,
            jnp.ones(1),
            comm=comm,
            nglobal=total + (wrong_size == "nglobal"),
            partition_info=(start, total),
        )


@pytest.mark.parametrize("mismatch", [None, "local", "remote"])
def test_cached_mpi_global_size_validation(monkeypatch, mock_mpi, mismatch):
    import jaxamg.mpi_utils as mpi_utils
    from jaxamg.cache import cache_mpi_metadata

    monkeypatch.setattr(mock_mpi, "LOR", object(), raising=False)
    counts = (2**30, 2**30, 1)
    start, total = 2**31, 2**31 + 1
    reductions = []

    def allreduce(value, op):
        assert op is mock_mpi.LOR
        reductions.append(value)
        return value or mismatch == "remote"

    comm = SimpleNamespace(allgather=lambda n: [*counts[:-1], n], allreduce=allreduce)
    local = sp.csr_matrix((np.ones(1, np.float32), [start], [0, 1]), shape=(1, total))

    class ReachedHaloPlan(Exception):
        pass

    def check_plan(indices, recvcounts, partition, comm):
        assert recvcounts == counts
        assert partition == (start, total)
        raise ReachedHaloPlan

    monkeypatch.setattr(mpi_utils, "build_halo_plan", check_plan)
    error = ReachedHaloPlan if mismatch is None else ValueError
    match = None if mismatch is None else "nglobal must match the sum"
    with pytest.raises(error, match=match):
        cache_mpi_metadata(
            {},
            comm,
            total + (mismatch == "local"),
            (start, total),
            local,
            is_symmetric=True,
        )
    assert reductions == [mismatch == "local"]


def test_host_allgather_displacements_past_int32(monkeypatch):
    from jaxamg.mpi_utils import make_allgather_vector

    counts = (2**30, 2**30, 1)
    total = sum(counts)

    def check_gather(values, receive):
        buffer, sizes, displacements, dtype = receive
        np.testing.assert_array_equal(sizes, counts)
        np.testing.assert_array_equal(displacements, [0, 2**30, 2**31])
        buffer[:] = values

    comm = SimpleNamespace(allgather=lambda n: counts, Allgatherv=check_gather)
    gather = make_allgather_vector(comm, (total - 1, total), total, backend="host")
    empty = np.empty
    monkeypatch.setattr(
        np, "empty", lambda shape, **kw: empty(1 if shape == total else shape, **kw)
    )

    # _forward_host imports MPI only to choose the floating point datatype.
    monkeypatch.setitem(
        sys.modules,
        "mpi4py",
        SimpleNamespace(MPI=SimpleNamespace(FLOAT=object(), DOUBLE=object())),
    )
    monkeypatch.setattr(jax, "pure_callback", lambda fn, shape, values: fn(values))
    np.testing.assert_array_equal(gather(jnp.ones(1)), [1])


def test_native_index_boundaries(tmp_path):
    """Compile/run with AMGX_ROOT and CUDA_HOME set, for either header ABI."""
    amgx_root, cuda_home = os.getenv("AMGX_ROOT"), os.getenv("CUDA_HOME")
    compiler = shutil.which(os.getenv("MPICXX", "mpicxx"))
    if not amgx_root or not cuda_home or not compiler:
        pytest.skip("requires AMGX_ROOT, CUDA_HOME and an MPI C++ compiler")
    from jax import ffi

    source = tmp_path / "indices.cc"
    source.write_text(r"""
#include "_amgx_solvers.h"
#include <cassert>
int main() {
  std::vector<int> offsets32;
  std::vector<int64_t> offsets64;
  const int limit = std::numeric_limits<int>::max();
  assert(!BuildPartitionOffsets({limit, 0}, 1, offsets32));
  assert(offsets32.back() == limit);
  assert(BuildPartitionOffsets({limit, 1}, 1, offsets32));
  assert(!BuildPartitionOffsets({limit, 1}, 1, offsets64));
  assert(offsets64.back() == int64_t{limit} + 1);
  assert(!BuildPartitionOffsets({limit - 1, 2}, 2, offsets32));
  assert(offsets32.back() == (int64_t{limit} + 1) / 2);
  assert(BuildPartitionOffsets({2, 3}, 2, offsets64));

  int rows[] = {0, 2, 3};
  int64_t cols[] = {2, (int64_t{1} << 34) + 1, int64_t{1} << 34};
  std::vector<int> rp, map, narrow;
  std::vector<int64_t> ci;
  assert(!BuildBsrScatterMap<int64_t>(2, 3, 2, false, rows, cols, rp, ci, map));
  assert((rp == std::vector<int>{0, 2}));
  assert((ci == std::vector<int64_t>{1, int64_t{1} << 33}));
  assert((map == std::vector<int>{0, 5, 6}));
  assert((BuildBsrScatterMap<int64_t, int>(2, 3, 2, false, rows, cols, rp, narrow, map)));

  // Scalar columns can exceed int32 even when stock block columns fit.
  cols[1] = (int64_t{1} << 31) + 1;
  cols[2] = int64_t{1} << 31;
  assert((!BuildBsrScatterMap<int64_t, int>(2, 3, 2, false, rows, cols, rp, narrow, map)));
  assert((narrow == std::vector<int>{1, 1 << 30}));
}
""")
    executable = tmp_path / "indices"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-DJAXAMG_WITH_MPI",
            "-DOMPI_SKIP_MPICXX",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wl,--gc-sections",
            "-I" + str(Path(__file__).resolve().parents[1] / "jaxamg"),
            "-I" + str(Path(amgx_root) / "include"),
            "-I" + str(Path(cuda_home) / "include"),
            "-I" + ffi.include_dir(),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(executable)], check=True)
