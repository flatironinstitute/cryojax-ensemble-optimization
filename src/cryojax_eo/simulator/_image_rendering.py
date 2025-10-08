from typing import Dict, Tuple

import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
from cryojax.ndimage.transforms import CircularCosineMask
from jaxtyping import Array, Float, Int, PRNGKeyArray


def _select_potential(volumes, idx):
    funcs = [lambda i=i: volumes[i] for i in range(len(volumes))]
    return jax.lax.switch(idx, funcs)


def render_image_with_white_gaussian_noise(
    particle_parameters: Dict,
    constant_args: Tuple[
        Tuple[cxs.AbstractVolumeRepresentation],
        CircularCosineMask,
        float,
    ],
    per_particle_args: Tuple[PRNGKeyArray, Int, Float],
) -> Float[
    Array,
    "{relion_particle_stack.config.y_dim} {relion_particle_stack.config.x_dim}",  # noqa
]:
    """
    Renders an image given the particle parameters, volume,
    and noise variance. The noise is White Gaussian noise.

    **Arguments:**
        - `particle_parameters`: The particle parameters.
        - `constant_args`: A tuple with the volumes, the mask, and the data_sign.
          A data_sign of 1.0 means dark-on-light, -1.0 means light-on-dark.

        - `per_particle_args`: A containing a random jax key,
            the potential_idx to use, and the noise variance.
    **Returns:**
        The rendered image.

    """
    key_noise, potential_idx, snr = per_particle_args
    volumes, mask, data_sign = constant_args
    volume = _select_potential(volumes, potential_idx)

    image_model = cxs.make_image_model(
        volume,
        particle_parameters["image_config"],
        particle_parameters["pose"],
        particle_parameters["transfer_theory"],
        signal_region=(mask.array == 1),
        normalizes_signal=True,
        simulates_quantity=False,
    )

    distribution = cxs.UncorrelatedGaussianNoiseModel(
        image_model,
        variance=1.0,
        signal_scale_factor=jnp.sqrt(snr),
    )
    return data_sign * distribution.sample(key_noise)
