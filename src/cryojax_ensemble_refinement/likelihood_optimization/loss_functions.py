from functools import partial
from typing import Optional, Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.data import ParticleStack
from jaxtyping import Array, Float

from ..simulator._distributions import (
    VarianceMarginalizedWhiteGaussianNoise,
    WhiteGaussianNoise,
)


def _compute_likelihood_image_and_walker(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStack,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    noise_variance: Optional[Float] = None,
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

    if noise_variance is None:
        distribution = VarianceMarginalizedWhiteGaussianNoise(imaging_pipeline)
    else:
        distribution = WhiteGaussianNoise(imaging_pipeline, noise_variance)

    return distribution.log_likelihood(relion_stack.images)


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(None, eqx.if_array(0), None, None, None), out_axes=0)
@partial(eqx.filter_vmap, in_axes=(0, None, None, None, None), out_axes=0)
def compute_likelihood_matrix(
    ensemble_walkers: Float[Array, "n_walkers n_atoms 3"],
    relion_stack: ParticleStack,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    noise_variance: Optional[Float] = None,
) -> Float[Array, "n_images n_walkers"]:
    """
    Compute the likelihood matrix for a set of walkers and a Relion stack.
    The likelihood is computed for each walker and each image in the stack.

    **Arguments:**
    - `ensemble_walkers`: The walkers of the ensemble. This is a 3D array
        with shape (n_walkers, n_atoms, 3).
    - `relion_stack`: A cryojax `ParticleStack` object containing the images and parameters.
    - `gaussian_amplitudes`: The amplitudes for the GMM atom potential.
    - `gaussian_variances`: The variances for the GMM atom potential.
    - `noise_variance`: The noise variance for the imaging pipeline. If None, the
        variance is marginalized over the noise. This is a scalar.
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
        noise_variance,
    )


@eqx.filter_jit
def neg_log_likelihood_from_weights(
    weights: Float[Array, " n_walkers"],
    likelihood_matrix: Float[Array, "n_images n_walkers"],
) -> Float:
    """
    Compute the negative log likelihood from the weights and a pre-computed likelihood matrix.
    The likelihood is averaged to avoid numerical issues and dependence on the number of images.

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
def neg_log_likelihood_from_walkers_and_weights(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    relion_stack: ParticleStack,
    args: Tuple[
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float | None,
    ],
) -> Float:
    """
    Compute the negative log likelihood from the walkers and weights.
    The likelihood is averaged to avoid numerical issues and dependence on the number of images.

    This function is used for optimizing the walkers of the ensemble with fixed weights.

    Args:
        walkers: The walkers of the ensemble. This is a 3D array
            with shape (n_walkers, n_atoms, 3).
        weights: The weights of the ensemble.
        relion_stack: A cryojax `ParticleStack` object containing the images and parameters.
        args: The arguments for the likelihood function.
    Returns:
        The negative log likelihood of the ensemble.
    """
    lklhood_matrix = compute_likelihood_matrix(walkers, relion_stack, *args)
    return neg_log_likelihood_from_weights(weights, lklhood_matrix)


# @eqx.filter_jit
# @partial(jax.value_and_grad, argnums=0, has_aux=True)
# def compute_loss_weights_and_grads(atom_positions, weights, relion_stack_vmap, args):
#     lklhood_matrix = compute_likelihood_matrix(atom_positions, relion_stack_vmap, args)
#     return compute_loss(weights, lklhood_matrix), weights
