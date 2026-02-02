from typing import Tuple

import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
from cryojax.ndimage import CircularCosineMask
from jaxtyping import Array, Float, Int, PRNGKeyArray

from cryojax_eo.typing import ParticleParameterInfo


def _select_potential(volumes, idx):
    funcs = [lambda i=i: volumes[i] for i in range(len(volumes))]
    return jax.lax.switch(idx, funcs)


def simulate_image_with_white_gaussian_noise(
    particle_parameters: ParticleParameterInfo,
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
    Simulates an image given the particle parameters, volume,
    and noise variance. The noise is White Gaussian noise.

    **Arguments:**
        - `particle_parameters`: The particle parameters.
        - `constant_args`: A tuple containing: a tuple of volume parametrizations,
           a mask defining the signal region, and the data_sign.
           A data_sign of 1.0 means dark-on-light, -1.0 means light-on-dark.

        - `per_particle_args`: A containing a random jax key,
            the potential_idx to use, and the signal-to-noise ratio
    **Returns:**
        The simulated image.

    """
    key_noise, potential_idx, snr = per_particle_args
    volumes, mask, data_sign = constant_args
    volume = _select_potential(volumes, potential_idx)

    # pose = jax.lax.cond(
    #     isinstance(volume, cxs.FourierVoxelGridVolume),
    #     lambda p: p.to_inverse_rotation(),
    #     lambda p: p,
    #     particle_parameters["pose"],
    # )
    pose = particle_parameters["pose"]
    image_model = cxs.make_image_model(
        volume,
        particle_parameters["image_config"],
        pose,
        particle_parameters["transfer_theory"],
        signal_region=(mask.get() == 1),
        normalizes_signal=True,
    )

    distribution = cxs.GaussianWhiteNoiseModel(
        image_model,
        variance=1.0,
        signal_scale_factor=jnp.sqrt(snr),
    )
    return data_sign * distribution.sample(key_noise)
