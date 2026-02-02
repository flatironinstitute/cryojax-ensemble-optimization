from typing import Optional, Tuple

import jax.numpy as jnp
from jaxtyping import Array, Float


def compute_optimal_scale_and_offset(
    target_image: Float[Array, "n_pixels n_pixels"],
    ref_image: Float[Array, "n_pixels n_pixels"],
    signal_region: Optional[Float[Array, "n_pixels n_pixels"]] = None,
) -> Tuple[Float, Float]:
    if signal_region is None:
        signal_region = jnp.ones_like(target_image, dtype=bool)

    cc = jnp.mean(target_image**2, where=signal_region)
    co = jnp.mean(ref_image * target_image, where=signal_region)
    c = jnp.mean(target_image, where=signal_region)
    o = jnp.mean(ref_image, where=signal_region)

    scale = (co - c * o) / (cc - c**2)
    offset = o - scale * c

    return scale, offset
