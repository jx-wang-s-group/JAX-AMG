"""
Helper functions.
"""
import jax.numpy as jnp
import jax.experimental.sparse as jsp
import numpy as np
import scipy.sparse as sp

def from_scipy(A: sp.csr_matrix) -> jsp.CSR:
    """Convert a scipy CSR matrix to JAX CSR format.

    Args:
        A: Scipy CSR matrix

    Returns:
        JAX CSR matrix with int32 indices and float32 values
    """
    data = jnp.array(A.data, dtype=jnp.float32)
    indices = jnp.array(A.indices, dtype=jnp.int32)
    indptr = jnp.array(A.indptr, dtype=jnp.int32)
    return jsp.CSR((data, indices, indptr), shape=A.shape)


def to_scipy(A: jsp.CSR) -> sp.csr_matrix:
    """Convert a JAX CSR matrix to scipy CSR format.

    Args:
        A: JAX CSR matrix

    Returns:
        Scipy CSR matrix
    """
    return sp.csr_matrix((np.asarray(A.data), np.asarray(A.indices), np.asarray(A.indptr)), shape=A.shape)

