from functools import partial

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.data import ParticleStack
from jaxtyping import Array, Float

from ..._custom_types import Image, LossFn, PerParticleArgs


def _likelihood_isotropic_gaussian(
    computed_image: Image,
    observed_image: Image,
    noise_variance: Float,
) -> Float:
    cc = jnp.mean(computed_image**2)
    co = jnp.mean(observed_image * computed_image)
    c = jnp.mean(computed_image)
    o = jnp.mean(observed_image)

    scale = (co - c * o) / (cc - c**2)
    bias = o - scale * c

    return jnp.sum((scale * computed_image - observed_image + bias) ** 2) / (
        2 * noise_variance
    )


def _likelihood_isotropic_gaussian_marginalized(
    computed_image: Float[Array, "n_pixels n_pixels"],
    observed_image: Float[Array, "n_pixels n_pixels"],
    _=None,
) -> Float:
    cc = jnp.mean(computed_image**2)
    co = jnp.mean(observed_image * computed_image)
    c = jnp.mean(computed_image)
    o = jnp.mean(observed_image)

    scale = (co - c * o) / (cc - c**2)
    bias = o - scale * c
    n_pixels = computed_image.size

    return (2 - n_pixels) * jnp.log(
        jnp.linalg.norm(scale * computed_image - observed_image + bias)
    )


def _compute_likelihood_image_and_walker(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStack,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    per_particle_args: PerParticleArgs,
) -> Float:
    potential = cxs.GaussianMixtureAtomicPotential(
        walker,
        gaussian_amplitudes,
        gaussian_variances,
    )
    structural_ensemble = cxs.SingleStructureEnsemble(
        potential, relion_stack.parameters.pose
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble=structural_ensemble,
        potential_integrator=cxs.GaussianMixtureProjection(),
        transfer_theory=relion_stack.parameters.transfer_theory,
    )

    imaging_pipeline = cxs.ContrastImageModel(
        relion_stack.parameters.instrument_config, scattering_theory
    )

    computed_image = imaging_pipeline.render(outputs_real_space=True)

    return image_to_walker_log_likelihood_fn(
        computed_image,
        relion_stack.images,
        per_particle_args,
    )


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, None, 0, 0, None, None), out_axes=0)
@partial(
    eqx.filter_vmap,
    in_axes=(None, eqx.if_array(0), None, None, None, eqx.if_array(0)),
    out_axes=0,
)
def _compute_likelihood_matrix(
    ensemble_walkers: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStack,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    per_particle_args: PerParticleArgs,
) -> Float[Array, "n_images n_walkers"]:
    """
    Compute the likelihood matrix for a set of walkers and a Relion stack.
    The likelihood is computed for each walker and each image in the stack.

    **Arguments:**
    - `ensemble_walkers`: The walkers of the ensemble. This is a 3D array
        with shape (n_walkers, n_atoms, 3).
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `gaussian_amplitudes`: The amplitudes for the GMM atom potential.
    - `gaussian_variances`: The variances for the GMM atom potential.
    - `image_to_walker_log_likelihood_fn`: The function to compute the likelihood
        between the computed image and the observed image.
    - `per_particle_args`: The arguments to pass to the likelihood function.
    **Returns:**
    - The likelihood matrix of the ensemble. This is a 2D array
        such that the n, m element is p(y_n | x_m), where y_n is the n-th image
        and x_m is the m-th walker (atomic model).
    """

    return _compute_likelihood_image_and_walker(
        ensemble_walkers,
        relion_stack,
        gaussian_amplitudes,
        gaussian_variances,
        image_to_walker_log_likelihood_fn,
        per_particle_args,
    )


def compute_likelihood_matrix(
    ensemble_walkers: Float[Array, "n_walkers n_atoms 3"],
    relion_stack: ParticleStack,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    per_particle_args: PerParticleArgs,
) -> Float[Array, "n_images n_walkers"]:
    """
    Compute the likelihood matrix for a set of walkers and a Relion stack.
    The likelihood is computed for each walker and each image in the stack.

    **Arguments:**
    - `ensemble_walkers`: The walkers of the ensemble. This is a 3D array
        with shape (n_walkers, n_atoms, 3).
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `gaussian_amplitudes`: The amplitudes for the GMM atom potential.
    - `gaussian_variances`: The variances for the GMM atom potential.
    - `image_to_walker_log_likelihood_fn`: The function to compute the likelihood
        between the computed image and the observed image.
    - `per_particle_args`: The arguments to pass to the likelihood function.
    **Returns:**
    - The likelihood matrix of the ensemble. This is a 2D array
        such that the n, m element is p(y_n | x_m), where y_n is the n-th image
        and x_m is the m-th walker (atomic model).
    """

    return _compute_likelihood_matrix(
        ensemble_walkers,
        relion_stack,
        gaussian_amplitudes,
        gaussian_variances,
        image_to_walker_log_likelihood_fn,
        per_particle_args,
    ).T  # order of vmaps!


@eqx.filter_jit
def compute_neg_log_likelihood_from_weights(
    weights: Float[Array, " n_walkers"],
    likelihood_matrix: Float[Array, "n_images n_walkers"],
) -> Float:
    """
    Compute the negative log likelihood from the weights and a pre-computed likelihood
    matrix. The likelihood is averaged to avoid numerical issues and dependence on the
    number of images.

    This function is used for optimizing the weights of the ensemble with fixed walkers.

    Args:
        weights: The weights of the ensemble.
        likelihood_matrix: The likelihood matrix of the ensemble. This is a 2D array
        such that the n, m element is p(y_n | x_m), where y_n is the n-th image
        and x_m is the m-th walker (atomic model).
    Returns:
        The negative log likelihood of the ensemble.
    """
    log_lklhood = jax.scipy.special.logsumexp(
        a=likelihood_matrix, b=weights[None, :], axis=1
    )
    return -jnp.mean(log_lklhood)


@eqx.filter_jit
def compute_neg_log_likelihood(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    relion_stack: ParticleStack,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    per_particle_args: PerParticleArgs,
) -> Float:
    """
    Compute the negative log likelihood from the walkers and weights. The likelihood is
    averaged to avoid numerical issues and dependence on the number of images.

    This function is used for optimizing the walkers of the ensemble with fixed weights.

    Args:
        walkers: The walkers of the ensemble. This is a 3D array
            with shape (n_walkers, n_atoms, 3).
        weights: The weights of the ensemble.
        relion_stack: A cryojax `ParticleStack` object.
        gaussian_amplitudes: The amplitudes for the GMM atom potential.
        gaussian_variances: The variances for the GMM atom potential.
        image_to_walker_log_likelihood_fn: The function to compute the likelihood
            between the computed image and the observed image.
        per_particle_args: The arguments to pass to the likelihood function.
    Returns:
        The negative log likelihood of the ensemble.
    """
    lklhood_matrix = compute_likelihood_matrix(
        walkers,
        relion_stack,
        gaussian_amplitudes,
        gaussian_variances,
        image_to_walker_log_likelihood_fn,
        per_particle_args,
    )
    return compute_neg_log_likelihood_from_weights(weights, lklhood_matrix)
