"""Linear maps with declared transposes, as one JAX primitive.

`LinearMap(forward, transpose, name)`: the JVP applies the map, the transpose
applies ``transpose``, batching maps one element at a time, and the forward
function's effects (mpi4jax's ordered effect) are kept. Used where JAX's own
transpose would be wrong, e.g. a collective sum, whose transpose must sum the
cotangents again.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.extend.core import Primitive
from jax.interpreters import ad, batching, mlir
from jax.sharding import NamedSharding


class LinearMap:
    """A linear map ``forward`` with transpose ``transpose``; compared by
    identity, so build one per distinct map and reuse it."""

    def __init__(self, forward: Callable, transpose: Callable, name: str):
        self.forward, self.transpose, self.name = forward, transpose, name
        self._transposed: LinearMap | None = None

    @property
    def T(self) -> LinearMap:
        if self._transposed is None:
            self._transposed = LinearMap(self.transpose, self.forward, self.name + "ᵀ")
            self._transposed._transposed = self
        return self._transposed

    def __call__(self, x: jax.Array) -> jax.Array:
        return linear_map_p.bind(x, linear_map=self)

    def __repr__(self) -> str:
        return self.name


linear_map_p = Primitive("jaxamg_linear_map")
linear_map_p.def_impl(lambda x, *, linear_map: linear_map.forward(x))


def _abstract_eval(x, *, linear_map):
    # Shape, type and effects in a fresh context (tracing inside this rule
    # would leak under an eager linearization).
    template = jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding)
    out = jax.eval_shape(linear_map.forward, template)
    effects = jax.make_jaxpr(linear_map.forward)(template).effects
    sharding = getattr(out, "sharding", None)
    if isinstance(sharding, NamedSharding):
        sharding = NamedSharding(sharding.mesh.abstract_mesh, sharding.spec)
    else:
        sharding = x.sharding
    return x.update(shape=out.shape, dtype=out.dtype, sharding=sharding), effects


linear_map_p.def_effectful_abstract_eval(_abstract_eval)
mlir.register_lowering(
    linear_map_p,
    lambda ctx, x, *, linear_map: mlir.lower_fun(
        linear_map.forward, multiple_results=False
    )(ctx, x),
)
ad.deflinear2(
    linear_map_p,
    lambda ct, x, *, linear_map: [
        linear_map_p.bind(ad.instantiate_zeros(ct), linear_map=linear_map.T)
    ],
)


def _batch(args, dims, *, linear_map):
    (x,), (d,) = args, dims
    mapped = jax.lax.map(
        lambda v: linear_map_p.bind(v, linear_map=linear_map), jnp.moveaxis(x, d, 0)
    )
    return mapped, 0


batching.primitive_batchers[linear_map_p] = _batch
