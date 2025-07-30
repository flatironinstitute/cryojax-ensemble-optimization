from functools import partial
from typing import Dict

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from dm_pix import rotate
from jax.nn import relu
from jaxtyping import Array, Float, Int

from ..._custom_types import Image, LossFn, PerParticleArgs


def _optimal_scale_and_bias(
    computed_image: Image,
    observed_image: Image,
) -> Float:
    """
    Notes:
    returns NaN when compute_image is a constant image
    """
    cc = jnp.mean(computed_image**2)
    co = jnp.mean(observed_image * computed_image)
    c = jnp.mean(computed_image)
    o = jnp.mean(observed_image)

    scale = (co - c * o) / (cc - c**2)
    bias = o - scale * c

    return scale, bias, cc, co, c, o


def _likelihood_sliced_wasserstein(
    computed_image: Image,
    observed_image: Image,
    n_projections: Int,
) -> Float:
    """
    Compute the likelihood using the sliced Wasserstein distance.
    """
    scale, bias, _, _, _, _ = _optimal_scale_and_bias(computed_image, observed_image)
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
    w_pos = wasserstein_1d_torch_pairwise(
        projections_computed_pos, projections_observed_pos, p
    )
    w_neg = wasserstein_1d_torch_pairwise(
        projections_computed_neg, projections_observed_neg, p
    )
    sliced_wasserstein = w_pos + w_neg
    return sliced_wasserstein


def wasserstein_1d_torch_pairwise(a, b, p):
    """
    Compute all pairwise 1D Wasserstein-2^2 distances between two batches of histograms.
    Assumes spatial bins are equally spaced.

    Args:
        a: (N1, n) tensor of histograms (each row sums to 1)
        b: (N2, n) tensor of histograms
        eps: numerical stability value for normalization

    Returns:
        w2_matrix: (N1, N2) tensor where w2_matrix[i, j] = W2^2(a[i], b[j])

    Notes:
    Eq 2 in https://openreview.net/forum?id=yPBtJ4JPwi
    """
    eps = 1e-8
    # Normalize histograms
    a = a / (a.sum(axis=1, keepdims=True) + eps)  # (N1, n)
    b = b / (b.sum(axis=1, keepdims=True) + eps)  # (N2, n)

    # Compute CDFs
    cdf_a = jnp.cumsum(a, axis=1)  # (N1, n)
    cdf_b = jnp.cumsum(b, axis=1)  # (N2, n)

    # Compute pairwise squared L2 distances between CDFs
    diff = cdf_a - cdf_b
    if p == 1:
        w = jnp.abs(diff).mean()
    elif p == 2:
        w = (diff**2).mean()
    else:
        raise ValueError(f"Unsupported p value: {p}. Only p=1 and p=2 are supported.")

    return w


def _likelihood_isotropic_gaussian(
    computed_image: Image,
    observed_image: Image,
    noise_variance: Float,
) -> Float:
    """
    Notes:
    returns NaN when compute_image is a constant image
    """
    scale, bias, cc, co, c, o = _optimal_scale_and_bias(computed_image, observed_image)

    return -jnp.sum((scale * computed_image - observed_image + bias) ** 2) / (
        2 * noise_variance
    )


def _likelihood_isotropic_gaussian_marginalized(
    computed_image: Float[Array, "n_pixels n_pixels"],
    observed_image: Float[Array, "n_pixels n_pixels"],
    _=None,
) -> Float:
    """
    Notes:
    returns NaN when compute_image is a constant image
    returns inf when compute_image is equal to observed_image
    """
    scale, bias, cc, co, c, o = _optimal_scale_and_bias(computed_image, observed_image)
    n_pixels = computed_image.size

    return (2 - n_pixels) * jnp.log(
        jnp.linalg.norm(scale * computed_image - observed_image + bias)
    )


def _compute_likelihood_image_and_walker(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: Dict,
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

    image_model = cxs.make_image_model(
        potential,
        relion_stack["parameters"]["config"],
        relion_stack["parameters"]["pose"],
        relion_stack["parameters"]["transfer_theory"],
    )

    computed_image = image_model.simulate(outputs_real_space=True)

    return image_to_walker_log_likelihood_fn(
        computed_image,
        relion_stack["images"],
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
    ensemble_walkers: Float[Array, " n_atoms 3"],
    relion_stack: Dict,
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
    relion_stack: Dict,
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
    relion_stack: Dict,
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
        per_particle_args,
    )
    return compute_neg_log_likelihood_from_weights(weights, lklhood_matrix)
