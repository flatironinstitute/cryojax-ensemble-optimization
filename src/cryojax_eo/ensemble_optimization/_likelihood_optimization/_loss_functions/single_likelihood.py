from typing import Any, Literal

import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ....simulator._dilated_mask import DilatedMask
from .common_functions import compute_optimal_scale_and_offset


class AbstractImageToWalkerLogLikelihoodFn(eqx.Module, strict=True):
    amplitudes: eqx.AbstractVar[Float[Array, "n_atoms n_gaussians_per_atom"]]
    variances: eqx.AbstractVar[Float[Array, "n_atoms n_gaussians_per_atom"]]
    image_sign: eqx.AbstractVar[Float[Array, ""]]
    dilated_mask: eqx.AbstractVar[DilatedMask | None]

    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        image: Float[Array, "y x"],
        image_config: cxs.BasicImageConfig,
        pose: cxs.AbstractPose,
        transfer_theory: cxs.ContrastTransferTheory,
        per_particle_args: Any,
    ) -> Float:
        raise NotImplementedError


class MargGaussianWhiteLogLikelihoodFn(AbstractImageToWalkerLogLikelihoodFn, strict=True):
    """A log likelihood function that models the likelihood of an image given a walker
    using a Gaussian noise model where the variance has been marginalized with
    Jeffrey's prior. This is useful when the variance is not known.

    **Arguments:**
    - `amplitudes`:
        The amplitudes for the GMM atomic volume representation.
    - `variances`:
        The variances for the GMM atomic volume representation.
    - `image_sign`: Set to dark-on-light if the experimental images are dark particles on
        a light background (most common for Relion stacks). Set to light-on-dark if the
        experimental images are light particles on a dark background (most common for
        cryoJAX generated data).
    - `dilated_mask`:
        An optional dilated mask to apply to the computed image and observed image.
    """

    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"]
    variances: Float[Array, "n_atoms n_gaussians_per_atom"]
    image_sign: Float[Array, ""]
    dilated_mask: DilatedMask | None

    def __init__(
        self,
        amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_atoms n_gaussians_per_atom"],
        data_sign: Literal["dark-on-light", "light-on-dark"],
        dilated_mask: DilatedMask | None = None,
    ):
        """Init the MargGaussianWhiteLogLikelihoodFn.

        **Arguments:**
        - `amplitudes`:
            The amplitudes for the GMM atomic volume representation.
        - `variances`:
            The variances for the GMM atomic volume representation.
        - `data_sign`:
            Set to dark-on-light if the experimental images are
            dark particles on a light background (most common for Relion stacks).
            Set to light-on-dark if the experimental images are light particles on a
            dark background (most common for cryoJAX generated data).
        - `dilated_mask`:
            An optional dilated mask to apply to the computed image and observed image.
        """
        assert (amplitudes > 0).all(), "Amplitudes must be positive."
        assert (variances > 0).all(), "Variances must be positive."
        assert data_sign in [
            "dark-on-light",
            "light-on-dark",
        ], "data_sign must be either 'dark-on-light' or 'light-on-dark'."
        image_sign = -1.0 if data_sign == "dark-on-light" else 1.0

        self.variances = variances
        self.amplitudes = amplitudes
        self.image_sign = jnp.asarray(image_sign)
        self.dilated_mask = dilated_mask

    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        image: Float[Array, "y x"],
        image_config: cxs.BasicImageConfig,
        pose: cxs.AbstractPose,
        transfer_theory: cxs.ContrastTransferTheory,
        per_particle_args: Any,
    ) -> Float:
        volume = cxs.GaussianMixtureVolume(
            walker,
            self.amplitudes,
            self.variances,
        )
        return likelihood_iso_gaussian_marg(
            volume=volume,
            image=image,
            image_config=image_config,
            pose=pose,
            transfer_theory=transfer_theory,
            dilated_mask=self.dilated_mask,
            image_sign=self.image_sign,
        )


class GaussianWhiteLogLikelihoodFn(AbstractImageToWalkerLogLikelihoodFn, strict=True):
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"]
    variances: Float[Array, "n_atoms n_gaussians_per_atom"]
    image_sign: Float[Array, ""]
    dilated_mask: DilatedMask | None

    def __init__(
        self,
        amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_atoms n_gaussians_per_atom"],
        data_sign: Literal["dark-on-light", "light-on-dark"],
        dilated_mask: DilatedMask | None = None,
    ):
        """Init the GaussianWhiteLogLikelihoodFn.

        **Arguments:**
        - `amplitudes`:
            The amplitudes for the GMM atomic volume representation.
        - `variances`:
            The variances for the GMM atomic volume representation.
        - `data_sign`:
            Set to dark-on-light if the experimental images are
            dark particles on a light background (most common for Relion stacks).
            Set to light-on-dark if the experimental images are light particles on a
            dark background (most common for cryoJAX generated data).
        - `dilated_mask`:
            An optional dilated mask to apply to the computed image and observed image.
        """
        assert (amplitudes > 0).all(), "Amplitudes must be positive."
        assert (variances > 0).all(), "Variances must be positive."
        assert data_sign in [
            "dark-on-light",
            "light-on-dark",
        ], "data_sign must be either 'dark-on-light' or 'light-on-dark'."
        image_sign = -1.0 if data_sign == "dark-on-light" else 1.0

        self.variances = variances
        self.amplitudes = amplitudes
        self.image_sign = jnp.asarray(image_sign)
        self.dilated_mask = dilated_mask

    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        image: Float[Array, "y x"],
        image_config: cxs.BasicImageConfig,
        pose: cxs.AbstractPose,
        transfer_theory: cxs.ContrastTransferTheory,
        per_particle_args: Float[Array, ""],
    ) -> Float:
        volume = cxs.GaussianMixtureVolume(
            walker,
            self.amplitudes,
            self.variances,
        )
        return likelihood_gaussian_white_noise(
            volume=volume,
            image=image,
            image_config=image_config,
            pose=pose,
            transfer_theory=transfer_theory,
            dilated_mask=self.dilated_mask,
            image_sign=self.image_sign,
            per_particle_args=per_particle_args,
        )


def likelihood_gaussian_white_noise(
    volume: cxs.AbstractVolumeRepresentation,
    image: Float[Array, "y x"],
    image_config: cxs.BasicImageConfig,
    pose: cxs.AbstractPose,
    transfer_theory: cxs.ContrastTransferTheory,
    dilated_mask: DilatedMask | None = None,
    image_sign: Float[Array, ""] = jnp.array(1.0),
    per_particle_args: Float[Array, ""] = jnp.array(1.0),
) -> Float:
    """
    Compute the likelihood of a walker given a Relion stack using an isotropic Gaussian
    likelihood function.

    **Arguments:**
    - `walker`: A `walker` that is, a point cloud representing an atomic model.
    - `relion_stack`: A cryojax `ParticleStack` object.
    - `amplitudes`: The amplitudes for the GMM atomic volume representation.
    - `variances`: The variances for the GMM atomic volume representation.
    - `dilated_mask`: An optional dilated mask to apply to the computed image.
    - `image_sign`: For this particular function the constant argument
        is the sign of the observed image. For typical Relion stacks this is -1.0.
        For data generated with cryoJAX this is 1.0.
    - `per_particle_args`: The noise variance for the likelihood function.

    **Returns:**
    - The log likelihood of the walker given the Relion stack.

    """

    noise_variance = per_particle_args

    image_model = cxs.make_image_model(
        volume,
        image_config,
        pose,
        transfer_theory,
        normalizes_signal=True,
    )
    computed_image = image_model.simulate()
    observed_image = jnp.asarray(image)

    if dilated_mask is not None:
        mask2d = dilated_mask.project(pose, image_config)
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = image_sign * observed_image * mask2d

    scale, offset = compute_optimal_scale_and_offset(computed_image, observed_image)

    return -jnp.sum((scale * computed_image - observed_image + offset) ** 2) / (
        2 * noise_variance
    )


def likelihood_iso_gaussian_marg(
    volume: cxs.AbstractVolumeRepresentation,
    image: Float[Array, "y x"],
    image_config: cxs.BasicImageConfig,
    pose: cxs.AbstractPose,
    transfer_theory: cxs.ContrastTransferTheory,
    dilated_mask: DilatedMask | None = None,
    image_sign: Float[Array, ""] = jnp.array(1.0),
    per_particle_args: None = None,
) -> Float:
    """
    Compute the marginalized likelihood of a walker given a Relion stack using an
    isotropic Gaussian likelihood function where the variance has been marginalized.
    This is useful when the variance is not known or is not fixed.
    """
    assert per_particle_args is None, (
        "per_particle_args is not used in this function and should be None."
    )

    image_model = cxs.make_image_model(
        volume,
        image_config,
        pose,
        transfer_theory,
        normalizes_signal=True,
    )
    computed_image = image_model.simulate()
    observed_image = jnp.asarray(image)

    if dilated_mask is not None:
        mask2d = dilated_mask.project(pose, image_config)
    else:
        mask2d = jnp.ones_like(computed_image)

    computed_image = computed_image * mask2d
    observed_image = image_sign * observed_image * mask2d

    scale, offset = compute_optimal_scale_and_offset(computed_image, observed_image)
    n_pixels = computed_image.size
    loss = -n_pixels * jnp.log(
        jnp.linalg.norm(scale * computed_image - observed_image + offset)
    )
    return loss
