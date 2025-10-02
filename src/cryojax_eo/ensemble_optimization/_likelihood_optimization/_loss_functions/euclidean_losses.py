from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from ...._custom_types import ParticleStackInfo, PerParticleT
from ....simulator._dilated_mask import DilatedMask
from .common_functions import compute_optimal_scale_and_offset
from .make_model_utils import make_image_model_from_gmm


def likelihood_isotropic_gaussian(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    dilated_mask: Optional[DilatedMask] = None,
    estimates_pose: bool = False,
    *,
    constant_args: float = 1.0,
    per_particle_args: float,
) -> Float:
    """
    Compute the likelihood of a walker given a Relion stack using an isotropic Gaussian
    likelihood function.

    **Arguments:**
    - `walker`: A `walker` that is, a point cloud representing an atomic model.
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `gaussian_amplitudes`: The amplitudes for the GMM atom potential.
    - `gaussian_variances`: The variances for the GMM atom potential.
    - `dilated_mask`: An optional dilated mask to apply to the computed image.
    - `constant_args`: For this particular function the constant argument
        is the sign of the observed image. For typical Relion stacks this is -1.0.
        For data generated with cryoJAX this is 1.0.
    - `per_particle_args`: The noise variance for the likelihood function.

    **Returns:**
    - The log likelihood of the walker given the Relion stack.

    """
    noise_variance = per_particle_args

    image_model = make_image_model_from_gmm(
        walker, relion_stack, gaussian_amplitudes, gaussian_variances, estimates_pose
    )
    computed_image = image_model.simulate()
    observed_image = relion_stack["images"]

    if dilated_mask is not None:
        mask2d = dilated_mask.project(relion_stack["parameters"]["pose"])
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = constant_args * observed_image * mask2d

    scale, offset = compute_optimal_scale_and_offset(computed_image, observed_image)

    return -jnp.sum((scale * computed_image - observed_image + offset) ** 2) / (
        2 * noise_variance
    )


def likelihood_isotropic_gaussian_marginalized(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    dilated_mask: Optional[DilatedMask] = None,
    estimates_pose: bool = False,
    *,
    constant_args: float = 1.0,
    per_particle_args: PerParticleT = (),
) -> Float:
    """
    Compute the marginalized likelihood of a walker given a Relion stack using an
    isotropic Gaussian likelihood function where the variance has been marginalized.
    This is useful when the variance is not known or is not fixed.

    **Arguments:**
    - `walker`: A `walker` that is, a point cloud representing an atomic model.
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `gaussian_amplitudes`: The amplitudes for the GMM atom potential.
    - `gaussian_variances`: The variances for the GMM atom potential.
    - `dilated_mask`: An optional dilated mask to apply to the computed image.
    - `constant_args`: For this particular function the constant argument
        is the sign of the observed image. For typical Relion stacks this is -1.0.
        For data generated with cryoJAX this is 1.0.
    - `per_particle_args`: not used in this function.
    """
    image_model = make_image_model_from_gmm(
        walker, relion_stack, gaussian_amplitudes, gaussian_variances, estimates_pose
    )
    computed_image = image_model.simulate()
    observed_image = relion_stack["images"]

    if dilated_mask is not None:
        mask2d = dilated_mask.project(relion_stack["parameters"]["pose"])
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = constant_args * observed_image * mask2d

    scale, offset = compute_optimal_scale_and_offset(computed_image, observed_image)
    n_pixels = computed_image.size

    return (2 - n_pixels) * jnp.log(
        jnp.linalg.norm(scale * computed_image - observed_image + offset)
    )
