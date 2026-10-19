"""Declared patterns (``jaxamg.patterns``). CPU-only."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxamg
from jaxamg.sparsity import materialize_sparse_matrix
from jaxamg.utils import to_bcsr_matrix


def _laplacian(n):
    """Matrix-free 1D Dirichlet Laplacian."""

    def op(x):
        y = 2 * x
        y = y.at[1:].add(-x[:-1])
        return y.at[:-1].add(-x[1:])

    return op


def _tridiagonal(n):
    i = np.arange(n)
    rows = np.concatenate([i, i[1:], i[:-1]])
    cols = np.concatenate([i, i[:-1], i[1:]])
    return jaxamg.pattern(rows, cols, (n, n))


def test_pattern_colouring_is_valid_and_materializes_the_operator():
    n = 12
    p = _tridiagonal(n)
    for r in range(n):  # no two columns of a row share a colour
        row = p.cols[p.rows == r]
        assert len(set(p.colors[row].tolist())) == len(row)
    op = _laplacian(n)
    rows, cols, colors, n_colors, shape = p._coloring()
    M = materialize_sparse_matrix(op, shape, rows, cols, colors, n_colors)
    dense = np.asarray(jax.jacfwd(op)(jnp.ones(n)))
    np.testing.assert_array_equal(np.asarray(M.todense()), dense)
    with pytest.raises(ValueError, match="outside the shape"):
        jaxamg.pattern([0], [n], (n, n))


def test_with_cache_pattern_and_coloring_are_exclusive():
    n = 4
    op = _laplacian(n)
    p = _tridiagonal(n)
    with pytest.raises(ValueError, match="either coloring or pattern"):
        jaxamg.with_cache(op, pattern=p, coloring=p._coloring())
    attached = jaxamg.with_cache(op, pattern=p)
    assert attached._coloring_info[3] == p.n_colors


def test_attaching_a_colouring_replaces_discovered_ones():
    n = 8
    op = lambda x: 2 * x - jnp.roll(x, 1)
    jaxamg.cache_coloring(op, (n, n))
    assert op._coloring_by_dtype
    plain = (np.arange(n), np.arange(n), np.zeros(n, np.int32), 1, (n, n))
    jaxamg.with_cache(op, coloring=plain)
    assert op._coloring_by_dtype == {}
    assert op._coloring_info is plain


@pytest.mark.parametrize("direct", [False, True])
def test_pattern_owns_immutable_inputs(direct):
    rows, cols, colors = np.array([1, 0]), np.array([1, 0]), np.array([0, 0])
    shape = [2, 2]
    p = (
        jaxamg.Pattern(rows, cols, shape, colors, 1)
        if direct
        else jaxamg.pattern(rows, cols, shape)
    )
    rows[:] = cols[:] = colors[:] = 9
    shape[0] = 99
    assert p.shape == (2, 2)
    np.testing.assert_array_equal(p.rows, [0, 1])
    np.testing.assert_array_equal(p.cols, [0, 1])
    for array in (p.rows, p.cols, p.colors):
        with pytest.raises(ValueError):
            array[0] = 9
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize(
    "rows,cols,shape",
    [
        ([0.5], [0], (2, 2)),
        ([0], [0, 1], (2, 2)),
        ([[0]], [[0]], (2, 2)),
        ([0], [0], (-1, 2)),
        ([0], [0], (2**32, 2)),
        ([0], [2**32], (2, 2)),
    ],
)
def test_invalid_coordinates_are_rejected(rows, cols, shape):
    with pytest.raises(ValueError):
        jaxamg.pattern(rows, cols, shape)


def test_direct_constructor_rejects_conflicting_colors():
    with pytest.raises(ValueError, match="different colors"):
        jaxamg.Pattern([0, 0], [0, 1], (2, 2), [0, 0], 1)


def test_supplied_colours_skip_assignment_and_materialize(monkeypatch):
    import jaxamg.patterns as patterns

    def unexpected(*args):
        raise AssertionError("supplied colours must not be recomputed")

    monkeypatch.setattr(patterns, "get_column_coloring", unexpected)
    colors = np.array([0, 1, -1], np.int32)
    declaration = jaxamg.pattern(
        [1, 0, 0, 0], [1, 1, 0, 1], (2, 3), colors=colors, n_colors=2
    )
    colors[:] = 0
    np.testing.assert_array_equal(declaration.colors, [0, 1, -1])
    with pytest.raises(ValueError):
        declaration.colors.flags.writeable = True

    def values(scale):
        return materialize_sparse_matrix(
            lambda vector: jnp.array([vector[0] + scale * vector[1], 2 * vector[1]]),
            declaration.shape,
            declaration.rows,
            declaration.cols,
            declaration.colors,
            declaration.n_colors,
        ).data.sum()

    value, gradient = jax.jit(jax.value_and_grad(values))(2.0)
    assert value == 5 and gradient == 1


@pytest.mark.parametrize(
    "colors,n_colors",
    [
        (None, 2),
        ([0, 1], None),
        ([0, 1], 0),
        ([0.0, 1.0], 2),
        ([0], 2),
        ([-2, 1], 2),
        ([-1, 1], 2),
        ([0, 2], 2),
        ([0, 1], 2**31),
        ([0, 0], 2),
    ],
)
def test_supplied_colours_are_validated(colors, n_colors):
    with pytest.raises(ValueError):
        jaxamg.pattern([0, 0], [0, 1], (1, 2), colors=colors, n_colors=n_colors)


@pytest.mark.parametrize("dtype", [np.int32, np.uint64])
def test_large_row_colour_keys_remain_exact(dtype):
    limit = np.iinfo(np.int32).max
    colors = np.array([limit - 2, limit - 1], dtype=dtype)
    declaration = jaxamg.pattern(
        [limit - 1, limit - 1],
        [0, 1],
        (limit, 2),
        colors=colors,
        n_colors=limit,
    )
    np.testing.assert_array_equal(declaration.colors, colors)
    assert declaration.colors.dtype == np.int32


def test_duplicates_and_zero_values_are_retained_as_a_set():
    p = jaxamg.pattern([1, 0, 0, 0], [1, 1, 0, 1], (2, 2))

    def materialize(c):
        op = jaxamg.with_cache(lambda x: x + c * jnp.array([x[1], 0]), pattern=p)
        return to_bcsr_matrix(op, jnp.ones(2)).todense()

    np.testing.assert_array_equal(jax.jit(materialize)(0.0), np.eye(2))
    np.testing.assert_array_equal(jax.jit(materialize)(2.0), [[1, 2], [0, 1]])


def test_replacing_coloring_clears_pattern():
    op = jaxamg.with_cache(_laplacian(4), pattern=_tridiagonal(4))
    assert op._pattern is not None
    plain = (np.arange(4), np.arange(4), np.zeros(4, np.int32), 1, (4, 4))
    jaxamg.with_cache(op, coloring=plain)
    assert op._pattern is None
    np.testing.assert_array_equal(to_bcsr_matrix(op, jnp.ones(4)).indices, np.arange(4))


def test_pickle_preserves_immutable_declaration():
    import pickle

    p = pickle.loads(pickle.dumps(_tridiagonal(4)))
    with pytest.raises(ValueError):
        p.rows.flags.writeable = True
    np.testing.assert_array_equal(p.rows, _tridiagonal(4).rows)


def test_halo_operator_without_ghosts():
    """A halo operator whose rows reference only owned columns (a block-
    diagonal operator, or an isolated rank) materializes with its local
    columns."""
    pattern = jaxamg.pattern([0, 1], [0, 1], (2, 2))
    op = jaxamg.halo_operator(
        lambda x, g: 2 * x, n_local=2, ghost_ids=[], pattern=pattern
    )
    matrix = op._local_matrix(4, 8, jnp.float32, traced=False)
    np.testing.assert_array_equal(np.asarray(matrix.indices), [4, 5])
    np.testing.assert_array_equal(np.asarray(matrix.data), [2, 2])


def test_halo_operator_owns_ghost_ids_and_validates_inputs():
    ids = np.array([8, 10], np.int64)
    op = jaxamg.halo_operator(lambda own, ghost: own, n_local=3, ghost_ids=ids)
    ids[:] = 0
    np.testing.assert_array_equal(op.ghost_ids, [8, 10])
    with pytest.raises(ValueError):
        op.ghost_ids.flags.writeable = True
    for invalid in ([-1], [1.5], [[1, 2]], [2, 2]):
        with pytest.raises(ValueError):
            jaxamg.halo_operator(lambda own, ghost: own, n_local=3, ghost_ids=invalid)
    for invalid in (-1, 1.5, True):
        with pytest.raises((ValueError, TypeError)):
            jaxamg.halo_operator(lambda own, ghost: own, n_local=invalid, ghost_ids=[])
