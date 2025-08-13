from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float


# import jax


def compute_optimal_scale_and_offset(
    target_image: Float[Array, "n_pixels n_pixels"],
    ref_image: Float[Array, "n_pixels n_pixels"],
) -> Tuple[Float, Float]:
    cc = jnp.mean(target_image**2)
    co = jnp.mean(ref_image * target_image)
    c = jnp.mean(target_image)
    o = jnp.mean(ref_image)

    scale = (co - c * o) / (cc - c**2)
    offset = o - scale * c

    # def print_debug_info():
    #     jax.debug.print("cc: {cc}", cc=cc)
    #     jax.debug.print("co: {co}", co=co)
    #     jax.debug.print("c: {c}", c=c)
    #     jax.debug.print("o: {o}", o=o)
    #     jax.debug.print("Optimal scale: {s}", s=scale)

    # jax.lax.cond(jnp.abs(scale) < 1e-6, print_debug_info, lambda: None)
    # jax.debug.print("Optimal offset: {o}", o=offset)

    return scale, offset
