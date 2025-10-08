from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from cryojax.dataset import ParticleStackInfo
from dm_pix import rotate
from jax.nn import relu
from jaxtyping import Array, Float, Int

from ....simulator._dilated_mask import DilatedMask
from .common_functions import compute_optimal_scale_and_offset
from .make_model_utils import make_image_model_from_gmm


def _rotate_and_project(image, angles):
    image_channel_one = jnp.expand_dims(image, -1)
    projected_image = jax.vmap(
        lambda angle: jnp.sum(
            rotate(image_channel_one, angle) * image_channel_one, axis=1
        )
    )(angles)
    return jnp.squeeze(projected_image, -1)


def likelihood_sliced_wasserstein(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    dilated_mask: Optional[DilatedMask] = None,
    estimates_pose: bool = False,
    *,
    constant_args: Tuple[Int, Int] = (18, 2),
    per_particle_args: Tuple = (),
    # n_projections: Int,
) -> Float:
    """
    Compute the likelihood using the sliced Wasserstein distance.

    **Arguments:**
    - `walker`: A `walker` that is, a point cloud representing an atomic model.
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
    - `dilated_mask`: An optional dilated mask to apply to the computed image.
    - `constant_args`: A tuple containing constant arguments for the function.
        For this function these are the number of projections and the p-norm to use.
        - n_projections: int, default 18
        - p_norm: int, default 2
    - `per_particle_args`: Not used in this function.
    """
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    n_projections, p_norm = constant_args

    image_model = make_image_model_from_gmm(
        walker, relion_stack, amplitudes, variances, estimates_pose
    )
    computed_image = image_model.simulate()
    observed_image = jnp.asarray(relion_stack["images"])
    # jax.debug.print("Variance: {variance}", variance=jnp.var(computed_image))

    if dilated_mask is not None:
        mask2d = dilated_mask.project(relion_stack["parameters"]["pose"])
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = observed_image * mask2d
    scale, offset = compute_optimal_scale_and_offset(computed_image, observed_image)
    # jax.debug.print("Computed scale: {scale}, bias: {bias}", scale=scale, bias=offset)
    # scale = 1.0
    # offset = 0.0

    angles = jnp.linspace(0, jnp.pi, n_projections, endpoint=False)
    rescaled_computed_image = scale * computed_image + offset

    projections_computed_pos = _rotate_and_project(relu(rescaled_computed_image), angles)
    projections_observed_pos = _rotate_and_project(relu(observed_image), angles)
    projections_computed_neg = _rotate_and_project(
        -relu(-rescaled_computed_image), angles
    )
    projections_observed_neg = _rotate_and_project(-relu(-observed_image), angles)

    w_pos = _wasserstein_1d_via_cdf(
        projections_computed_pos, projections_observed_pos, p_norm
    )
    w_neg = _wasserstein_1d_via_cdf(
        projections_computed_neg, projections_observed_neg, p_norm
    )
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
