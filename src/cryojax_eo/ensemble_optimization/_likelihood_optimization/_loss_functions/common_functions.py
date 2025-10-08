from typing import Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float


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

    return scale, offset
