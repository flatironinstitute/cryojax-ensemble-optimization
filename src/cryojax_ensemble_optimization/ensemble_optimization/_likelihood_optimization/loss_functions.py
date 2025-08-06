from functools import partial
from typing import Optional, Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from dm_pix import rotate
from jax.nn import relu
from jaxtyping import Array, Float, Int

from ..._custom_types import ConstantT, LossFn, ParticleStackInfo, PerParticleT
from ...simulator._dilated_mask import DilatedMask


def _likelihood_sliced_wasserstein(
    computed_image: Float[Array, "n_pixels n_pixels"],
    observed_image: Float[Array, "n_pixels n_pixels"],
    n_projections: Int,
) -> Float:
    """
    Compute the likelihood using the sliced Wasserstein distance.
    """
    scale, bias = _compute_optimal_scale_and_offset(computed_image, observed_image)
    angles = jnp.linspace(0, jnp.pi, n_projections, endpoint=False)

    def rotate_and_project(image, angles):
        image_channel_one = jnp.expand_dims(image, -1)
        projected_image = jax.vmap(
            lambda angle: jnp.sum(
                rotate(image_channel_one, angle) * image_channel_one, axis=1
            )
        )(angles)
        return jnp.squeeze(projected_image, -1)

    rescaled_computed_image = scale * computed_image + bias
    projections_computed_pos = rotate_and_project(relu(rescaled_computed_image), angles)
    projections_observed_pos = rotate_and_project(relu(observed_image), angles)
    projections_computed_neg = rotate_and_project(-relu(-rescaled_computed_image), angles)
    projections_observed_neg = rotate_and_project(-relu(-observed_image), angles)
    p = 2  # TODO: pass in param as 1 or 2
    w_pos = _wasserstein_1d_via_cdf(projections_computed_pos, projections_observed_pos, p)
    w_neg = _wasserstein_1d_via_cdf(projections_computed_neg, projections_observed_neg, p)
    sliced_wasserstein = w_pos + w_neg
    return sliced_wasserstein


def _wasserstein_1d_via_cdf(histograms_1: Array, histograms_2: Array, p: Int) -> Float:
    """
    Compute the 1D Sliced Wasserstein-p^p distance.

    Computes the Wasserstein distance between two sets of histograms via the c
    umulative distribution functions (CDFs). Assumes spatial bins are equally spaced.
    Histograms are normalized to sum to 1 in this function.

    Args:
        histograms_1: (n_hist, n_pix) tensor of histograms (each row will be sumed to 1)
        histograms_2: (n_hist, n_pix) tensor of histograms
        eps: numerical stability value for normalization

    Returns:
        wasserstein^p: scalar value of the Wasserstein^p distance.

    Notes:
    Eq 2 in https://openreview.net/forum?id=yPBtJ4JPwi
    """
    eps = 1e-8
    # Normalize histograms
    histograms_1 = histograms_1 / (
        histograms_1.sum(axis=1, keepdims=True) + eps
    )  # (n_hist, n_pix)
    histograms_2 = histograms_2 / (histograms_2.sum(axis=1, keepdims=True) + eps)

    # Compute CDFs
    cdf_1 = jnp.cumsum(histograms_1, axis=1)  # (n_hist, n_pix)
    cdf_2 = jnp.cumsum(histograms_2, axis=1)

    # Compute pairwise squared L2 distances between CDFs
    diff = cdf_1 - cdf_2
    if p == 1:
        wasserstein_1d = jnp.abs(diff).mean()
    elif p == 2:
        wasserstein_1d = (diff**2).mean()
    else:
        raise ValueError(f"Unsupported p value: {p}. Only p=1 and p=2 are supported.")

    return wasserstein_1d


def _make_image_model(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
) -> cxs.ContrastImageModel:
    potential = cxs.GaussianMixtureAtomicPotential(
        walker,
        gaussian_amplitudes,
        gaussian_variances,
    )

    return cxs.make_image_model(
        potential,
        relion_stack["parameters"]["config"],
        relion_stack["parameters"]["pose"],
        relion_stack["parameters"]["transfer_theory"],
    )


def _compute_optimal_scale_and_offset(
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


def likelihood_isotropic_gaussian(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    dilated_mask: Optional[DilatedMask] = None,
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

    image_model = _make_image_model(
        walker,
        relion_stack,
        gaussian_amplitudes,
        gaussian_variances,
    )
    computed_image = image_model.simulate()
    observed_image = relion_stack["images"]

    if dilated_mask is not None:
        mask2d = dilated_mask.project(relion_stack["parameters"]["pose"])
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = constant_args * observed_image * mask2d

    scale, offset = _compute_optimal_scale_and_offset(computed_image, observed_image)

    return -jnp.sum((scale * computed_image - observed_image + offset) ** 2) / (
        2 * noise_variance
    )


def likelihood_isotropic_gaussian_marginalized(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    dilated_mask: Optional[DilatedMask] = None,
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
    image_model = _make_image_model(
        walker,
        relion_stack,
        gaussian_amplitudes,
        gaussian_variances,
    )
    computed_image = image_model.simulate()
    observed_image = relion_stack["images"]

    if dilated_mask is not None:
        mask2d = dilated_mask.project(relion_stack["parameters"]["pose"])
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = constant_args * observed_image * mask2d

    scale, offset = _compute_optimal_scale_and_offset(computed_image, observed_image)
    n_pixels = computed_image.size

    return (2 - n_pixels) * jnp.log(
        jnp.linalg.norm(scale * computed_image - observed_image + offset)
    )


@partial(eqx.filter_vmap, in_axes=(0, None, 0, 0, None, None, None, None), out_axes=0)
@partial(
    eqx.filter_vmap,
    in_axes=(None, eqx.if_array(0), None, None, None, None, None, eqx.if_array(0)),
    out_axes=0,
)
def _compute_likelihood_matrix(
    ensemble_walkers: Float[Array, " n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    dilated_mask: DilatedMask | None,
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

    return image_to_walker_log_likelihood_fn(
        ensemble_walkers,
        relion_stack,
        gaussian_amplitudes,
        gaussian_variances,
        dilated_mask,
        constant_args=constant_args,
        per_particle_args=per_particle_args,
    )


def compute_likelihood_matrix(
    ensemble_walkers: Float[Array, "n_walkers n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask] = None,
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
        dilated_mask,
        constant_args,
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
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    image_to_walker_log_likelihood_fn: LossFn,
    dilated_mask: Optional[DilatedMask] = None,
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
        dilated_mask,
        constant_args=constant_args,
        per_particle_args=per_particle_args,
    )
    return compute_neg_log_likelihood_from_weights(weights, lklhood_matrix)
