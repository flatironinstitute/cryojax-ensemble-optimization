from typing import Tuple

import cryojax.simulator as cxs
import jax.numpy as jnp
from cryojax.data import (
    RelionParticleParameters,
)
from cryojax.image.operators import AbstractMask
from cryojax.inference.distributions import IndependentGaussianPixels
from jaxtyping import Array, Float, Int, PRNGKeyArray


def render_image_with_white_gaussian_noise(
    relion_particle_parameters: RelionParticleParameters,
    constant_args: Tuple[
        Tuple[cxs.AbstractPotentialRepresentation],
        cxs.AbstractPotentialIntegrator, AbstractMask
    ],
    per_particle_args: Tuple[PRNGKeyArray, Int, Float],
) -> Float[
    Array,
    "{relion_particle_stack.instrument_config.y_dim} {relion_particle_stack.instrument_config.x_dim}",  # noqa
]:
    """
    Renders an image given the particle parameters, potential,
    and noise variance. The noise is White Gaussian noise.

    **Arguments:**
        relion_particle_parameters: The particle parameters.
        constant_args: A tuple with the potential and potential integrator.
        per_particle_args: A containing a random jax key,
            the potential_idx to use, and the noise variance.
    **Returns:**
        The rendered image.

    """
    key_noise, potential_idx, snr = per_particle_args
    potentials, potential_integrator, mask = constant_args

    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        relion_particle_parameters.pose,
        cxs.DiscreteConformationalVariable(potential_idx),
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        relion_particle_parameters.transfer_theory,
    )

    image_model = cxs.ContrastImageModel(
        relion_particle_parameters.instrument_config, scattering_theory, mask=mask
    )

    distribution = IndependentGaussianPixels(
        image_model,
        variance=1.0,
        signal_scale_factor=jnp.sqrt(snr),
        normalizes_signal=True,
    )
    return distribution.sample(key_noise, applies_mask=False)
