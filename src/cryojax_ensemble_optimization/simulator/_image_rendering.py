from typing import Dict, Tuple

import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
from cryojax.ndimage.transforms import AbstractMask
from jaxtyping import Array, Float, Int, PRNGKeyArray


def _select_potential(potentials, idx):
    funcs = [lambda i=i: potentials[i] for i in range(len(potentials))]
    return jax.lax.switch(idx, funcs)


def render_image_with_white_gaussian_noise(
    particle_parameters: Dict,
    constant_args: Tuple[
        Tuple[cxs.AbstractPotentialRepresentation],
        AbstractMask,
    ],
    per_particle_args: Tuple[PRNGKeyArray, Int, Float],
) -> Float[
    Array,
    "{relion_particle_stack.config.y_dim} {relion_particle_stack.config.x_dim}",  # noqa
]:
    """
    Renders an image given the particle parameters, potential,
    and noise variance. The noise is White Gaussian noise.

    **Arguments:**
        particle_parameters: The particle parameters.
        constant_args: A tuple with the potential and potential integrator.
        per_particle_args: A containing a random jax key,
            the potential_idx to use, and the noise variance.
    **Returns:**
        The rendered image.

    """
    key_noise, potential_idx, snr = per_particle_args
    potentials, mask = constant_args
    potential = _select_potential(potentials, potential_idx)

    image_model = cxs.make_image_model(
        potential,
        particle_parameters["config"],
        particle_parameters["pose"],
        particle_parameters["transfer_theory"],
        signal_region=(mask.array == 1),
        normalizes_signal=True,
    )

    distribution = cxs.IndependentGaussianPixels(
        image_model,
        variance=1.0,
        signal_scale_factor=jnp.sqrt(snr),
    )
    return distribution.sample(key_noise)
