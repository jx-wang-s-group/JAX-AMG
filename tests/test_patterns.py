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
