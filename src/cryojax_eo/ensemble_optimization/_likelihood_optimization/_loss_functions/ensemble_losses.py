from functools import partial
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.dataset import ParticleStackInfo
from jaxtyping import Array, Float

from ...._custom_types import ConstantT, LossFn, PerParticleT
from ....simulator._dilated_mask import DilatedMask


@partial(
    eqx.filter_vmap, in_axes=(0, None, 0, 0, None, None, None, None, None), out_axes=0
)
@partial(
    eqx.filter_vmap,
    in_axes=(None, eqx.if_array(0), None, None, None, None, None, None, eqx.if_array(0)),
    out_axes=0,
)
def _compute_likelihood_matrix(
    ensemble_walkers: Float[Array, " n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    dilated_mask: DilatedMask | None,
    estimates_pose: bool,
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
) -> Float[Array, "n_images n_walkers"]:
    """
    Compute the likelihood matrix for a set of walkers and a Relion stack.
    The likelihood is computed for each walker and each image in the stack.

    **Arguments:**
    - `ensemble_walkers`: The walkers of the ensemble. This is a 3D array
        with shape (n_walkers, n_atoms, 3).
    - `relion_stack`: A cryojax  Dict` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
    - `image_to_walker_log_likelihood_fn`: The function to compute the likelihood
        between the computed image and the observed image.
    - `per_particle_args`: The arguments to pass to the likelihood function.
    **Returns:**
    - The likelihood matrix of the ensemble. This is a 2D array
        such that the n, m element is p(y_n | x_m), where y_n is the n-th image
        and x_m is the m-th walker (atomic model).
    """

    return image_to_walker_log_likelihood_fn(
        ensemble_walkers,
        relion_stack,
        amplitudes,
        variances,
        dilated_mask,
        estimates_pose,
        constant_args=constant_args,
        per_particle_args=per_particle_args,
    )


@eqx.filter_jit
def compute_likelihood_matrix(
    ensemble_walkers: Float[Array, "n_walkers n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask] = None,
    estimates_pose: bool = False,
    *,
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
) -> Float[Array, "n_images n_walkers"]:
    """
    Compute the likelihood matrix for a set of walkers and a Relion stack.
    The likelihood is computed for each walker and each image in the stack.

    **Arguments:**
    - `ensemble_walkers`: The walkers of the ensemble. This is a 3D array
        with shape (n_walkers, n_atoms, 3).
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
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
        amplitudes,
        variances,
        image_to_walker_log_likelihood_fn,
        dilated_mask,
        estimates_pose,
        constant_args,
        per_particle_args,
    ).T


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
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask] = None,
    estimates_pose: bool = False,
    *,
    constant_args: ConstantT,
    per_particle_args: PerParticleT,
) -> Float:
    """
    Compute the negative log likelihood from the walkers and weights. The likelihood is
    averaged to avoid numerical issues and dependence on the number of images.

    This function is used for optimizing the walkers of the ensemble with fixed weights.

    Args:
        walkers: The walkers of the ensemble. This is a 3D array
            with shape (n_walkers, n_atoms, 3).
        weights: The weights of the ensemble.
        relion_stack: A cryojax  Dict` object.
        amplitudes: The amplitudes for the GMM atomic volume representation.
        variances: The variances for the GMM atomic volume representation.
        image_to_walker_log_likelihood_fn: The function to compute the likelihood
            between the computed image and the observed image.
        per_particle_args: The arguments to pass to the likelihood function.
    Returns:
        The negative log likelihood of the ensemble.
    """
    lklhood_matrix = compute_likelihood_matrix(
        walkers,
        relion_stack,
        amplitudes,
        variances,
        image_to_walker_log_likelihood_fn,
        dilated_mask,
        estimates_pose,
        constant_args=constant_args,
        per_particle_args=per_particle_args,
    )
    return compute_neg_log_likelihood_from_weights(weights, lklhood_matrix)
