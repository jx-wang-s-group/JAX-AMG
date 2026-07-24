import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import jaxamg
from jaxamg.matrices import (
    poisson_matrix,
    poisson_matrix_distributed,
    rhs_linear,
    tridiagonal_matrix_distributed,
)
from jaxamg.mpi_utils import (
    gather_vector,
    get_partition_info,
    make_allgather_vector,
    partition_csr_matrix,
    partition_operator,
    partition_vector,
    validate_partition,
)


@pytest.fixture
def mpi_context():
    """Fixture providing MPI context (comm, rank, nranks)."""
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()

    yield comm, rank, nranks

    comm.Barrier()
    jaxamg.finalize()


@pytest.mark.mpi(min_size=2)
def test_mpi_poisson(mpi_context):
    comm, rank, nranks = mpi_context

    grid_size = 16
    n = grid_size**2

    # Create local matrix for each process
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )

    b_global = rhs_linear(n)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    # Solve the system on each process
    x_local, info = jaxamg.solve(
        A_local,
        b_local,
        comm=comm,
        nglobal=n,
        partition_info=(row_start, row_end),
        solver="PCG",
        preconditioner={"solver": "MULTICOLOR_DILU", "max_iters": 100},
    )

    # Gather the solution to the root process
    x = gather_vector(x_local, comm, root=0)

    # Check if the solve was successful
    assert info["status"] == jaxamg.AMGXStatus.SUCCESS

    # Check if the solution is correct
    if rank == 0:
        A_global = poisson_matrix(grid_size)
        b_global = rhs_linear(n)
        np.testing.assert_allclose(A_global @ x, b_global, atol=1e-5)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("enable_x64", [False, True])
def test_mpi_autodiff_jit(mpi_context, enable_x64):
    comm, rank, nranks = mpi_context

    # Test with both 32-bit and 64-bit precision
    jax.config.update("jax_enable_x64", enable_x64)

    n_global = 16

    b_global = jnp.ones(n_global)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    # Pre-cache MPI metadata
    config = {"solver": "CG"}
    dummy_A, row_start, row_end = tridiagonal_matrix_distributed(
        n_global, rank, nranks, 4.0
    )
    mpi_cache = jaxamg.cache_mpi_metadata(
        config, comm, n_global, (row_start, row_end), dummy_A, is_symmetric=True
    )

    def loss_fn(diag_val):
        # Create matrix
        A, _, _ = tridiagonal_matrix_distributed(
            n_global, rank, nranks, diagonal_value=diag_val
        )

        # Attach MPI cache
        A = jaxamg.with_cache(A, mpi=mpi_cache, is_symmetric=True)

        # Solve
        x_local, _ = jaxamg.solve(A, b_local)

        return jnp.sum(x_local**2)

    diag_val = 5.0

    # Compute gradient with JIT and make sure no warnings
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        grad = jax.jit(jax.grad(loss_fn))(diag_val)

        # Filter out mpi4jax warnings
        non_mpi4jax_warnings = [
            warning for warning in w if "mpi4jax" not in str(warning.filename)
        ]

        # Fail if there are any warnings
        if non_mpi4jax_warnings:
            raise AssertionError(
                f"Found {len(non_mpi4jax_warnings)} warning(s):\n"
                + "\n".join(
                    f"{w.filename}:{w.lineno}: {w.message}"
                    for w in non_mpi4jax_warnings
                )
            )

    # Compare with finite difference
    check_grads(loss_fn, (diag_val,), order=1, modes=["rev"])

    # Compare with non-JIT execution
    def loss_nojit(diag_val):
        A, _, _ = tridiagonal_matrix_distributed(
            n_global, rank, nranks, diagonal_value=diag_val
        )
        x_local, _ = jaxamg.solve(
            A,
            b_local,
            comm=comm,
            nglobal=n_global,
            partition_info=(row_start, row_end),
            config=config,
        )
        return jnp.sum(x_local**2)

    grad_nojit = jax.grad(loss_nojit)(diag_val)

    # Gradients should match
    np.testing.assert_allclose(grad, grad_nojit)

    # Reset to default precision
    jax.config.update("jax_enable_x64", False)


@pytest.mark.mpi(min_size=2)
def test_mpi_partition(mpi_context):
    comm, rank, nranks = mpi_context

    grid_size = 4
    n = grid_size**2

    # Create local matrix based on predefined partition function
    # for 2D Poisson matrix
    A, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )
    validate_partition(A, n, row_start, row_end)

    # Create local matrix based on auto partition from global matrix
    A_global = poisson_matrix(grid_size)
    A_local, row_start_auto, row_end_auto = partition_csr_matrix(A_global, rank, nranks)
    validate_partition(A_local, n, row_start_auto, row_end_auto)

    # Check if the two partitions are the same
    np.testing.assert_array_equal(A.todense(), A_local.todense())
    np.testing.assert_array_equal(row_start, row_start_auto)
    np.testing.assert_array_equal(row_end, row_end_auto)


@pytest.mark.mpi(min_size=2)
def test_mpi_allgatherv(mpi_context):
    """Test variable-size allgatherv."""
    comm, rank, nranks = mpi_context
    from jaxamg.mpi_utils import _mpi4jax_allgatherv

    # Create variable sized arrays per rank: Rank r sends [r] * (r+1)
    # Rank 0: [0]
    # Rank 1: [1, 1]
    # Rank 2: [2, 2, 2]
    size_local = rank + 1
    sendbuf = jnp.ones(size_local, dtype=jnp.int32) * rank

    # Expected global array
    expected_parts = []
    recvcounts = []
    for r in range(nranks):
        count = r + 1
        expected_parts.append(np.ones(count, dtype=np.int32) * r)
        recvcounts.append(count)

    expected_global = np.concatenate(expected_parts)
    recvcounts_tuple = tuple(recvcounts)

    # Run allgatherv
    gathered = _mpi4jax_allgatherv(sendbuf, recvcounts_tuple, comm)

    # Check result
    np.testing.assert_array_equal(gathered, expected_global)

    # Check JIT compatibility
    @jax.jit
    def gathered_jit_fn(sendbuf):
        return _mpi4jax_allgatherv(sendbuf, recvcounts_tuple, comm)

    gathered_jit = gathered_jit_fn(sendbuf)
    np.testing.assert_array_equal(gathered_jit, expected_global)


@pytest.mark.mpi(min_size=2)
def test_mpi_transpose(mpi_context):
    """Test distributed transpose on a non-symmetric matri."""
    comm, rank, nranks = mpi_context
    from mpi4py import MPI

    from jaxamg.mpi_utils import (
        _mpi4jax_alltoallv_transpose,
        local_transpose_nnz,
    )

    grid_size = 4
    n_global = grid_size**2

    # 1. Poisson matrix as base matrix
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )

    # 2. Make it non-symmetric by using global index to generate unique values
    nnz_local = A_local.data.shape[0]

    # Compute row indices for each element
    row_counts = A_local.indptr[1:] - A_local.indptr[:-1]
    row_indices_local = jnp.repeat(
        jnp.arange(A_local.shape[0], dtype=jnp.int32),
        row_counts,
        total_repeat_length=nnz_local,
    )
    row_indices_global = row_indices_local + row_start

    # Generate unique non-symmetric values
    # Cast to float to match matrix dtype
    new_data = (row_indices_global * n_global + A_local.indices).astype(
        A_local.data.dtype
    )

    # Gather row counts (needed by transpose)
    n_local = row_end - row_start
    row_counts_global = comm.allgather(n_local)
    recvcounts_tuple = tuple(row_counts_global)

    # Calculate max_nnz across ranks (send buffers) and this rank's local
    # nnz(A^T) (output). For this structurally symmetric graph the latter equals
    # the local nnz of A, so both transpose directions use the same value.
    max_nnz = comm.allreduce(nnz_local, op=MPI.MAX)
    nnz_out = local_transpose_nnz(A_local.indices, recvcounts_tuple, comm)

    # 3. Compute A^T
    data_T, indices_T, indptr_T = _mpi4jax_alltoallv_transpose(
        new_data,
        A_local.indices,
        A_local.indptr,
        recvcounts_tuple,
        comm,
        max_nnz,
        nnz_out,
    )

    # Verify A^T is different from A (sanity check)
    assert not np.allclose(data_T, new_data)

    # 4. Compute (A^T)^T -> should be A
    data_TT, indices_TT, indptr_TT = _mpi4jax_alltoallv_transpose(
        data_T, indices_T, indptr_T, recvcounts_tuple, comm, max_nnz, nnz_out
    )

    # 5. Verify (A^T)^T == A
    np.testing.assert_array_equal(indptr_TT, A_local.indptr)
    np.testing.assert_array_equal(indices_TT, A_local.indices)
    np.testing.assert_array_equal(data_TT, new_data)

    # 6. Verify JIT compatibility
    @jax.jit
    def transpose_jit_fn(data, indices, indptr):
        return _mpi4jax_alltoallv_transpose(
            data, indices, indptr, recvcounts_tuple, comm, max_nnz, nnz_out
        )

    # Run JIT-compiled transpose
    data_T_jit, indices_T_jit, indptr_T_jit = transpose_jit_fn(
        new_data, A_local.indices, A_local.indptr
    )

    # Verify JIT output matches non-JIT output
    np.testing.assert_array_equal(indptr_T_jit, indptr_T)
    np.testing.assert_array_equal(indices_T_jit, indices_T)
    np.testing.assert_allclose(data_T_jit, data_T)


@pytest.mark.mpi(min_size=2)
def test_mpi_transpose_nonsymmetric_nnz(mpi_context):
    """Distributed transpose of a structurally nonsymmetric matrix whose row
    ownership gives different local nnz for A and A^T.

    A has a dense first column plus a diagonal, so global column 0 is nonzero in
    every row. The rank owning global row 0 therefore holds a dense A^T row (n
    entries) while its A rows hold ~2 -- exactly the unequal-count case an
    nnz(A)-sized output would truncate (dropping the off-diagonal transpose
    entries). Sizing the output by local nnz(A^T) must recover the full A^T.
    """
    comm, rank, nranks = mpi_context
    import scipy.sparse as sp
    from mpi4py import MPI

    from jaxamg.mpi_utils import _mpi4jax_alltoallv_transpose, local_transpose_nnz

    n = 4 * nranks
    rows, cols, vals = [], [], []
    for i in range(n):
        rows.append(i)  # dense first column: A[i, 0] != 0 for every row i
        cols.append(0)
        vals.append(2.0 + i)
        if i > 0:  # diagonal (i == 0 already covered by the column-0 entry)
            rows.append(i)
            cols.append(i)
            vals.append(3.0 + i)
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    A.sort_indices()

    A_local, row_start, row_end = partition_csr_matrix(A, rank, nranks)
    n_local = row_end - row_start
    recvcounts_tuple = tuple(comm.allgather(n_local))
    max_nnz = comm.allreduce(int(A_local.data.shape[0]), op=MPI.MAX)
    nnz_out = local_transpose_nnz(A_local.indices, recvcounts_tuple, comm)

    data_T, indices_T, indptr_T = _mpi4jax_alltoallv_transpose(
        A_local.data,
        A_local.indices,
        A_local.indptr,
        recvcounts_tuple,
        comm,
        max_nnz,
        nnz_out,
    )
    data_T = np.asarray(data_T)
    indices_T = np.asarray(indices_T)
    indptr_T = np.asarray(indptr_T)

    # Ground truth: this rank's rows of the true global transpose.
    at_true = A.T.tocsr()[row_start:row_end].toarray()

    got = np.zeros((n_local, n), dtype=np.float64)
    for r in range(n_local):
        for k in range(int(indptr_T[r]), int(indptr_T[r + 1])):
            got[r, int(indices_T[k])] += data_T[k]

    np.testing.assert_allclose(got, at_true, atol=1e-6)

    # The scenario must actually exercise unequal local counts on some rank
    # (otherwise it would not distinguish the fix from the old nnz(A) sizing).
    grew = comm.allreduce(int(nnz_out != int(A_local.data.shape[0])), op=MPI.SUM)
    assert grew > 0, "test matrix should give unequal local nnz across transpose"


@pytest.mark.mpi(min_size=2)
def test_mpi_autodiff_nonsymmetric(mpi_context):
    """End-to-end distributed reverse-mode AD through a structurally nonsymmetric
    matrix (is_symmetric=False), driving the distributed-transpose backward path
    with unequal local nnz between A and A^T.

    A has a dense first column plus a diagonal (diagonally dominant), so global
    column 0 is nonzero in every row and local nnz(A^T) != nnz(A). The
    distributed VJP returns each rank's contribution to the gradient of the total
    loss L(theta) = sum_r sum(x_local_r ** 2) (the collective backward solve
    combines all ranks' cotangents), so the summed gradient is checked against a
    central finite difference of L.
    """
    import jax.experimental.sparse as jsp
    from mpi4py import MPI

    comm, rank, nranks = mpi_context
    jax.config.update("jax_enable_x64", True)
    try:
        n_global = 4 * nranks
        row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)
        b_local, _, _ = partition_vector(
            jnp.ones(n_global, dtype=jnp.float64), rank, nranks
        )

        # Static pattern for this rank's rows: dense first column + diagonal.
        loc_indices, is_diag, ptr, k = [], [], [0], 0
        for gi in range(row_start, row_end):
            if gi == 0:  # global row 0: diagonal only (it sits in column 0)
                loc_indices.append(0)
                is_diag.append(1.0)
                k += 1
            else:  # (i, 0) off-diagonal and (i, i) diagonal
                loc_indices.extend([0, gi])
                is_diag.extend([0.0, 1.0])
                k += 2
            ptr.append(k)
        indices = jnp.asarray(loc_indices, dtype=jnp.int32)
        indptr = jnp.asarray(ptr, dtype=jnp.int32)
        diag_mask = jnp.asarray(is_diag, dtype=jnp.float64)

        def loss_fn(theta):
            data = jnp.where(diag_mask > 0, 4.0 + theta, -1.0)
            A_local = jsp.BCSR((data, indices, indptr), shape=(n_local, n_global))
            x_local, _ = jaxamg.solve(
                A_local,
                b_local,
                comm=comm,
                nglobal=n_global,
                partition_info=(row_start, row_end),
                solver="GMRES",
                preconditioner="JACOBI_L1",
                max_iters=200,
                tolerance=1e-12,
            )
            return jnp.sum(x_local**2)

        theta = 5.0
        g_total = comm.allreduce(float(jax.grad(loss_fn)(theta)), op=MPI.SUM)

        def total_loss(t):
            return comm.allreduce(float(loss_fn(t)), op=MPI.SUM)

        eps = 1e-5
        g_fd = (total_loss(theta + eps) - total_loss(theta - eps)) / (2 * eps)
        np.testing.assert_allclose(g_total, g_fd, rtol=1e-4, atol=1e-8)
    finally:
        jax.config.update("jax_enable_x64", False)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("backend", ["host", "mpi4jax"])
def test_make_allgather_vector(mpi_context, backend):
    """Differentiable all-gather: forward equals a plain allgather, and the
    VJP is the adjoint slice (each rank keeps its own segment)."""
    comm, rank, nranks = mpi_context

    jax.config.update("jax_enable_x64", True)

    # n_global not divisible by nranks -> exercise uneven partitions.
    n_global = 23
    row_start, row_end, n_local = get_partition_info(n_global, rank, nranks)

    # Distinct local data on each rank.
    x_local = jnp.asarray(np.arange(row_start, row_end, dtype=np.float64) + 0.5 * rank)

    allgather = make_allgather_vector(
        comm, (row_start, row_end), n_global, backend=backend
    )

    # Forward equals a plain MPI allgather + concatenate.
    x_global = allgather(x_local)
    x_global_ref = np.concatenate(comm.allgather(np.asarray(x_local)))
    np.testing.assert_allclose(np.asarray(x_global), x_global_ref, rtol=1e-12)

    # Backward: the adjoint of an all-gather is this rank's slice of the global
    # cotangent.
    g_global = jnp.asarray(np.arange(n_global, dtype=np.float64) + 1.0)
    _, vjp_fn = jax.vjp(allgather, x_local)
    (g_local,) = vjp_fn(g_global)
    np.testing.assert_allclose(
        np.asarray(g_local), np.asarray(g_global)[row_start:row_end], rtol=1e-12
    )

    # Must work under jit(grad) of a scalar loss defined on the global vector --
    # the distributed-optimization use case. The loss is replicated on every
    # rank, so the VJP yields this rank's local contribution.
    target = jnp.asarray(np.arange(n_global, dtype=np.float64))

    def loss(x_loc):
        return jnp.sum((allgather(x_loc) - target) ** 2)

    g = jax.jit(jax.grad(loss))(x_local)
    expected = 2.0 * (np.asarray(x_global) - np.asarray(target))[row_start:row_end]
    np.testing.assert_allclose(np.asarray(g), expected, rtol=1e-10)

    jax.config.update("jax_enable_x64", False)


@pytest.mark.mpi(min_size=2)
def test_partition_operator(mpi_context):
    """partition_operator turns a global operator into this rank's row-local one,
    and materializing it reproduces exactly this rank's rows of the global matrix
    -- only the local (n_local, n_global) block is ever formed."""
    comm, rank, nranks = mpi_context

    jax.config.update("jax_enable_x64", True)

    # n_global not divisible by nranks -> exercise uneven partitions.
    n_global = 23
    rng = np.random.default_rng(0)

    # A concrete non-symmetric sparse global matrix and its matvec operator.
    A = rng.standard_normal((n_global, n_global))
    A[np.abs(A) < 1.0] = 0.0  # induce sparsity
    A[np.diag_indices(n_global)] = 5.0  # nonzero diagonal
    A_j = jnp.asarray(A)
    global_op = lambda x: A_j @ x

    local_op, row_start, row_end = partition_operator(global_op, n_global, rank, nranks)

    # Partition bounds agree with get_partition_info.
    rs_ref, re_ref, n_local = get_partition_info(n_global, rank, nranks)
    assert (row_start, row_end) == (rs_ref, re_ref)

    # The local operator is exactly the global operator sliced to this rank's rows.
    x = jnp.asarray(rng.standard_normal(n_global))
    np.testing.assert_allclose(
        np.asarray(local_op(x)),
        np.asarray(global_op(x))[row_start:row_end],
        rtol=1e-12,
    )

    # Materializing the local operator reproduces this rank's rows of A.
    from jaxamg.sparsity import (
        get_column_coloring,
        materialize_sparse_matrix,
        probe_sparsity_pattern,
    )

    shape = (n_local, n_global)
    rows, cols = probe_sparsity_pattern(local_op, shape)
    colors, n_colors = get_column_coloring(rows, cols, shape)
    A_local = materialize_sparse_matrix(local_op, shape, rows, cols, colors, n_colors)
    np.testing.assert_allclose(
        np.asarray(A_local.todense()), A[row_start:row_end, :], atol=1e-10
    )

    jax.config.update("jax_enable_x64", False)


@pytest.mark.mpi(min_size=3)
def test_mpi_subcommunicator(mpi_context):
    """The differentiable solve must run its backward collectives on the user's
    communicator, not MPI.COMM_WORLD.

    The solve runs on a subcommunicator that is a proper subset of COMM_WORLD
    (the last rank stays idle and only rejoins at the fixture's COMM_WORLD
    barrier). If the backward pass used COMM_WORLD it would deadlock waiting on
    the idle rank; with the fix it uses the subcommunicator and completes, and
    the gradient (summed over the subcommunicator) matches finite difference.
    """
    from mpi4py import MPI

    comm, rank, nranks = mpi_context
    jax.config.update("jax_enable_x64", True)

    color = 0 if rank < nranks - 1 else MPI.UNDEFINED
    sub = comm.Split(color, key=rank)
    try:
        if sub != MPI.COMM_NULL:
            sub_rank, sub_size = sub.Get_rank(), sub.Get_size()
            n_global = 4 * sub_size
            b_local, _, _ = partition_vector(
                jnp.ones(n_global, jnp.float64), sub_rank, sub_size
            )

            def loss(theta):
                A, rs, re = tridiagonal_matrix_distributed(
                    n_global, sub_rank, sub_size, diagonal_value=theta
                )
                x, _ = jaxamg.solve(
                    A,
                    b_local,
                    comm=sub,
                    nglobal=n_global,
                    partition_info=(rs, re),
                    solver="CG",
                )
                return jnp.sum(x**2)

            theta = 5.0
            g_total = sub.allreduce(float(jax.grad(loss)(theta)), op=MPI.SUM)

            def total_loss(t):
                return sub.allreduce(float(loss(t)), op=MPI.SUM)

            eps = 1e-5
            g_fd = (total_loss(theta + eps) - total_loss(theta - eps)) / (2 * eps)
            np.testing.assert_allclose(g_total, g_fd, rtol=1e-4, atol=1e-8)
    finally:
        if sub != MPI.COMM_NULL:
            sub.Free()
        jax.config.update("jax_enable_x64", False)


@pytest.mark.mpi(min_size=3)
def test_mpi_multiple_communicators(mpi_context):
    """A process may perform AmgX solves on more than one communicator in the
    same run, and each must receive its own AmgX resources (keyed per
    communicator) rather than sharing the first communicator's. A COMM_WORLD
    solve is followed by a proper-subset subcommunicator solve (the last rank
    idle), with no finalize in between; both must succeed.
    """
    from mpi4py import MPI

    comm, rank, nranks = mpi_context
    config = {
        "solver": "PCG",
        "preconditioner": {"solver": "MULTICOLOR_DILU", "max_iters": 100},
    }

    def dist_solve(c, c_rank, c_size):
        n = 4 * c_size
        A, rs, re = tridiagonal_matrix_distributed(n, c_rank, c_size, 4.0)
        b, _, _ = partition_vector(jnp.ones(n), c_rank, c_size)
        _, info = jaxamg.solve(
            A, b, comm=c, nglobal=n, partition_info=(rs, re), config=config
        )
        return info

    # 1. Solve on COMM_WORLD -- creates the world communicator's resources.
    info_world = dist_solve(comm, rank, nranks)
    assert info_world["status"] == jaxamg.AMGXStatus.SUCCESS

    # 2. Solve on a proper-subset subcommunicator (last rank idle) -- must get its
    #    own resources rather than reusing the COMM_WORLD ones.
    color = 0 if rank < nranks - 1 else MPI.UNDEFINED
    sub = comm.Split(color, key=rank)
    try:
        if sub != MPI.COMM_NULL:
            info_sub = dist_solve(sub, sub.Get_rank(), sub.Get_size())
            assert info_sub["status"] == jaxamg.AMGXStatus.SUCCESS
    finally:
        if sub != MPI.COMM_NULL:
            sub.Free()


@pytest.mark.mpi(min_size=2)
def test_mpi_amg_preconditioner(mpi_context):
    """Distributed solve with the classical-AMG preconditioner.

    Covers the distributed classical-AMG upload path (an explicit contiguous
    partition vector plus the 32-bit global-index upload,
    ``AMGX_matrix_upload_all_global_32``) that a correct multi-rank classical-AMG
    hierarchy requires. The other MPI tests use DILU/Jacobi/plain-CG, which do not
    build the AMG halo this path sets up, so this is the only coverage of it.
    A single AMG solve, so it does not depend on any AmgX-internal repeated-solve
    behavior.
    """
    comm, rank, nranks = mpi_context

    grid_size = 16
    n = grid_size**2

    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )
    b_global = rhs_linear(n)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    x_local, info = jaxamg.solve(
        A_local,
        b_local,
        comm=comm,
        nglobal=n,
        partition_info=(row_start, row_end),
        solver="PCG",
        preconditioner={"solver": "AMG"},
        tolerance=1e-8,
        max_iters=200,
    )
    x = gather_vector(x_local, comm, root=0)

    assert info["status"] == jaxamg.AMGXStatus.SUCCESS
    if rank == 0:
        A_global = poisson_matrix(grid_size)
        # float32 AMG: residual plateaus near single precision, so 1e-4.
        np.testing.assert_allclose(A_global @ x, b_global, atol=1e-4)


@pytest.mark.mpi(min_size=2)
def test_mpi_repeated_solve_warm_cache(mpi_context):
    """Repeated distributed solves of the same sparsity pattern reuse the cached
    per-communicator matrix structure and solver (the warm path:
    ``AMGX_matrix_replace_coefficients`` + ``AMGX_solver_resetup`` over cache-owned
    structure buffers). Two invariants:

    1. Solving the identical system three times returns the same correct solution
       every time (the warm path must be deterministic; no cross-solve drift).
    2. Solving a value-scaled system with the same pattern (a cache hit) must
       install the new coefficients, so 2A x = b gives x scaled by 1/2 -- not a
       silent reuse of the previous matrix.

    Uses a non-AMG preconditioner and a small matrix, so it exercises the JAX
    warm-path logic without depending on any AmgX-internal behavior.
    """
    import jax.experimental.sparse as jsp

    comm, rank, nranks = mpi_context

    grid_size = 16
    n = grid_size**2
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )
    b_global = rhs_linear(n)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    solve_kwargs = dict(
        comm=comm,
        nglobal=n,
        partition_info=(row_start, row_end),
        solver="PCG",
        preconditioner={"solver": "MULTICOLOR_DILU", "max_iters": 50},
        tolerance=1e-8,
        max_iters=300,
    )

    # (1) Identical system solved three times -> warm-path (cache-hit) reuse must
    #     be correct and deterministic across repeats.
    solutions = []
    for _ in range(3):
        x_local, info = jaxamg.solve(A_local, b_local, **solve_kwargs)
        assert info["status"] == jaxamg.AMGXStatus.SUCCESS
        solutions.append(gather_vector(x_local, comm, root=0))

    # (2) Same pattern, scaled values (still a cache hit) -> replace_coefficients
    #     must install the new values, so the solution scales by 1/2.
    A_scaled = jsp.BCSR(
        (A_local.data * 2.0, A_local.indices, A_local.indptr), shape=A_local.shape
    )
    x_local, info = jaxamg.solve(A_scaled, b_local, **solve_kwargs)
    assert info["status"] == jaxamg.AMGXStatus.SUCCESS
    x_scaled = gather_vector(x_local, comm, root=0)

    if rank == 0:
        A_global = poisson_matrix(grid_size)
        # float32 residuals plateau near single precision, so 1e-4.
        for x in solutions:
            np.testing.assert_allclose(A_global @ x, b_global, atol=1e-4)
        # No warm-path drift: repeats agree far tighter than any real drift
        # (the corruption this guards against changes the solution by O(1)).
        np.testing.assert_allclose(solutions[1], solutions[0], atol=1e-5)
        np.testing.assert_allclose(solutions[2], solutions[0], atol=1e-5)
        # New coefficients really took effect: (2A) x = b  =>  A x = b / 2, and
        # x = x0 / 2 rather than a silent reuse of the first matrix.
        np.testing.assert_allclose(A_global @ x_scaled, 0.5 * b_global, atol=1e-4)
        np.testing.assert_allclose(x_scaled, 0.5 * solutions[0], atol=1e-4)


@pytest.mark.mpi(min_size=2)
def test_mpi_residual_history(mpi_context):
    """The residual history is the global convergence curve on every rank."""
    comm, rank, nranks = mpi_context

    grid_size = 16
    n = grid_size**2
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )
    b_global = rhs_linear(n)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    _, info = jaxamg.solve(
        A_local,
        b_local,
        comm=comm,
        nglobal=n,
        partition_info=(row_start, row_end),
    )

    assert info["status"] == jaxamg.AMGXStatus.SUCCESS
    history = np.asarray(info["residual_history"])
    assert history.shape == (info["iterations"] + 1,)
    assert np.isfinite(history).all()
    np.testing.assert_allclose(history[-1], info["residual"], rtol=1e-6)

    # Entry 0 is the initial GLOBAL residual norm, and the monitored norm is a
    # global reduction, so every rank must see the identical curve.
    np.testing.assert_allclose(
        history[0], np.linalg.norm(np.asarray(b_global)), rtol=1e-5
    )
    all_hist = comm.allgather(history)
    for other in all_hist[1:]:
        np.testing.assert_array_equal(other, all_hist[0])


@pytest.mark.mpi(min_size=2)
def test_mpi_initial_guess(mpi_context):
    """Warm-starting from a previous solution cuts iterations in MPI mode."""
    comm, rank, nranks = mpi_context

    grid_size = 16
    n = grid_size**2
    A_local, row_start, row_end = poisson_matrix_distributed(
        grid_size, grid_size, rank, nranks
    )
    b_global = rhs_linear(n)
    b_local, _, _ = partition_vector(b_global, rank, nranks)

    solve_kwargs = dict(comm=comm, nglobal=n, partition_info=(row_start, row_end))
    x_star, info_ref = jaxamg.solve(A_local, b_local, **solve_kwargs)
    assert info_ref["status"] == jaxamg.AMGXStatus.SUCCESS

    # ABSOLUTE convergence so the warm start lowers the bar instead of
    # tightening it (RELATIVE_INI is relative to the initial residual).
    abs_kwargs = dict(solve_kwargs, convergence="ABSOLUTE", tolerance=1e-4)
    _, cold = jaxamg.solve(A_local, b_local, **abs_kwargs)
    x_warm, warm = jaxamg.solve(A_local, b_local, x0=x_star, **abs_kwargs)

    assert warm["status"] == jaxamg.AMGXStatus.SUCCESS
    assert warm["iterations"] < cold["iterations"]

    # The warm-started solve still lands on the reference solution.
    x_warm_g = gather_vector(x_warm, comm, root=0)
    x_star_g = gather_vector(x_star, comm, root=0)
    if rank == 0:
        np.testing.assert_allclose(
            np.asarray(x_warm_g), np.asarray(x_star_g), atol=1e-3
        )


def _partition_block_aligned(A_sp, b_global, rank, nranks, block_dim):
    """Row-partition a scipy CSR system on block boundaries.

    The naive row partition can split a block across two ranks (e.g. 512 rows
    over 3 ranks -> 171/171/170), which the block solver cannot handle;
    partition whole blocks instead.
    """
    import jax.experimental.sparse as jsp

    n_blocks = A_sp.shape[0] // block_dim
    blk_start, blk_end, _ = get_partition_info(n_blocks, rank, nranks)
    row_start, row_end = blk_start * block_dim, blk_end * block_dim
    A_loc = A_sp[row_start:row_end, :]
    A_local = jsp.BCSR(
        (
            jnp.asarray(A_loc.data.astype(np.float32)),
            jnp.asarray(A_loc.indices),
            jnp.asarray(A_loc.indptr),
        ),
        shape=A_loc.shape,
    )
    b_local = jnp.asarray(b_global[row_start:row_end])
    return A_local, b_local, row_start, row_end


@pytest.mark.mpi(min_size=2)
def test_mpi_block_solve(mpi_context):
    """Distributed block solve (block_dim=2) of a coupled kron system."""
    import scipy.sparse
    import scipy.sparse.linalg as spla

    comm, rank, nranks = mpi_context

    from jaxamg.utils import to_scipy

    M = np.array([[2.0, 1.0], [1.0, 2.0]], dtype=np.float32)
    A_sp = scipy.sparse.kron(to_scipy(poisson_matrix(16)), M).tocsr()
    n = A_sp.shape[0]
    b_global = np.linspace(0.5, 1.5, n).astype(np.float32)

    A_local, b_local, row_start, row_end = _partition_block_aligned(
        A_sp, b_global, rank, nranks, block_dim=2
    )

    # Default block config (aggregation AMG) on the distributed system.
    x_local, info = jaxamg.solve(
        A_local,
        b_local,
        comm=comm,
        nglobal=n,
        partition_info=(row_start, row_end),
        block_dim=2,
    )
    x = gather_vector(x_local, comm, root=0)

    assert info["status"] == jaxamg.AMGXStatus.SUCCESS
    if rank == 0:
        x_ref = spla.spsolve(A_sp.tocsc().astype(np.float64), b_global)
        np.testing.assert_allclose(np.asarray(x), x_ref, rtol=1e-4, atol=1e-6)


@pytest.mark.mpi(min_size=2)
def test_mpi_block_gradients(mpi_context):
    """Distributed block VJP (incl. distributed transpose) vs scipy adjoint."""
    import jax.experimental.sparse as jsp
    import scipy.sparse
    import scipy.sparse.linalg as spla

    comm, rank, nranks = mpi_context

    from jaxamg.utils import to_scipy

    M = np.array([[3.0, 1.0], [0.5, 2.0]], dtype=np.float32)  # nonsymmetric
    A_sp = scipy.sparse.kron(to_scipy(poisson_matrix(12, skew=0.7)), M).tocsr()
    n = A_sp.shape[0]
    b_global = np.linspace(0.5, 1.5, n).astype(np.float32)

    A_local, b_local, row_start, row_end = _partition_block_aligned(
        A_sp, b_global, rank, nranks, block_dim=2
    )

    def loss(vals, rhs):
        A_i = jsp.BCSR((vals, A_local.indices, A_local.indptr), shape=A_local.shape)
        x_l, _ = jaxamg.solve(
            A_i,
            rhs,
            comm=comm,
            nglobal=n,
            partition_info=(row_start, row_end),
            block_dim=2,
            solver="FGMRES",
            tolerance=1e-8,
            max_iters=300,
        )
        return jnp.sum(x_l * x_l)

    g_vals, g_b = jax.grad(loss, argnums=(0, 1))(A_local.data, b_local)

    # Reference: global loss is sum over ranks; each rank's cotangent is its
    # local 2x, so lambda solves A^T lambda = 2x (globally).
    A64 = A_sp.tocsc().astype(np.float64)
    x_ref = spla.spsolve(A64, b_global)
    lam = spla.spsolve(A64.T, 2.0 * x_ref)
    rows_l = np.repeat(
        np.arange(row_start, row_end), np.diff(np.asarray(A_local.indptr))
    )
    g_vals_ref = -lam[rows_l] * x_ref[np.asarray(A_local.indices)]
    g_b_ref = lam[row_start:row_end]

    np.testing.assert_allclose(
        np.asarray(g_vals), g_vals_ref, atol=1e-3 * np.max(np.abs(g_vals_ref))
    )
    np.testing.assert_allclose(
        np.asarray(g_b), g_b_ref, atol=1e-3 * np.max(np.abs(g_b_ref))
    )
