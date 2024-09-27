from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import field
from jaxtyping import Float, Complex, Array, PRNGKeyArray

from cryojax.inference.distributions import AbstractDistribution
from cryojax.simulator import AbstractImagingPipeline
from cryojax.image import rfftn

from .._errors import error_if_not_positive


class WhiteGaussianNoise(AbstractDistribution, strict=True):
    r"""A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    so that the variance to be an arbitrary noise power spectrum.
    """

    imaging_pipeline: AbstractImagingPipeline
    noise_variance: Float | Array
    is_signal_normalized: bool = field(static=True)

    def __init__(
        self,
        imaging_pipeline: AbstractImagingPipeline,
        noise_variance: Float[Array, ""],  # noqa: F722
        is_signal_normalized: bool = True,
    ):
        """**Arguments:**

        - `imaging_pipeline`: The image formation model.
        - `noise_variance`: The variance of the noise in fourier space.
        """  # noqa: E501
        self.imaging_pipeline = imaging_pipeline
        self.noise_variance = error_if_not_positive(noise_variance)
        self.is_signal_normalized = is_signal_normalized

        if not self.is_signal_normalized:
            raise NotImplementedError("A non-normalized signal is not yet supported.")

    @override
    def compute_signal(
        self, *, get_real: bool = True
    ) -> Float[
        Array,
        "{self.imaging_pipeline.instrument_config.y_dim} "  # noqa: F722
        "{self.imaging_pipeline.instrument_config.x_dim}",  # noqa: F722
    ]:
        """Render the image formation model."""

        simulated_image = self.imaging_pipeline.render(get_real=True)
        return simulated_image / jnp.linalg.norm(simulated_image)

    def compute_noise(
        self, rng_key: PRNGKeyArray, *, get_real: bool = True
    ) -> Float[
        Array,
        "{self.imaging_pipeline.instrument_config.y_dim} "  # noqa: F722
        "{self.imaging_pipeline.instrument_config.x_dim}",  # noqa: F722
    ]:
        pipeline = self.imaging_pipeline
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        noise = jr.normal(rng_key, shape=pipeline.instrument_config.shape) * jnp.sqrt(
            self.noise_variance
        )

        return noise

    @override
    def sample(
        self, rng_key: PRNGKeyArray, *, get_real: bool = True
    ) -> Float[
        Array,
        "{self.imaging_pipeline.instrument_config.y_dim} "  # noqa: F722
        "{self.imaging_pipeline.instrument_config.x_dim}",  # noqa: F722
    ]:
        """Sample from the gaussian noise model."""

        noisy_image = self.compute_signal(get_real=get_real) + self.compute_noise(
            rng_key, get_real=get_real
        )
        return noisy_image

    @override
    def log_likelihood(
        self,
        observed: Complex[
            Array,
            "{self.imaging_pipeline.instrument_config.y_dim} "  # noqa: F722
            "{self.imaging_pipeline.instrument_config.x_dim//2+1}",  # noqa: F722
        ],
    ) -> Float:
        """Evaluate the log-likelihood of the gaussian noise model.

        **Arguments:**

        - `observed` : The observed data in fourier space.
        """
        # Create simulated data
        simulated = self.compute_signal(get_real=True)

        # cc = jnp.mean(simulated**2)
        # co = jnp.mean(observed * simulated)
        # c = jnp.mean(simulated)
        # o = jnp.mean(observed)

        scale = 1.0  # (co / cc - o) / (1 - c)
        bias = 0.0  # o - scale * c

        # Compute residuals
        log_likelihood = -jnp.sum(
            (scale * simulated - observed + bias) ** 2 / (2 * self.noise_variance)
        )

        return log_likelihood


class VarianceMarginalizedWhiteGaussianNoise(AbstractDistribution, strict=True):
    imaging_pipeline: AbstractImagingPipeline
    is_signal_normalized: bool = field(static=True)

    def __init__(
        self,
        imaging_pipeline: AbstractImagingPipeline,
        is_signal_normalized: bool = True,
    ):
        """**Arguments:**

        - `imaging_pipeline`: The image formation model.
        """
        self.imaging_pipeline = imaging_pipeline
        self.is_signal_normalized = is_signal_normalized

    @override
    def compute_signal(
        self, *, get_real: bool = True
    ) -> Float[
        Array,
        "{self.imaging_pipeline.instrument_config.y_dim} "  # noqa: F722
        "{self.imaging_pipeline.instrument_config.x_dim}",  # noqa: F722
    ]:
        """Render the image formation model."""

        simulated_image = self.imaging_pipeline.render(get_real=True)

        if self.is_signal_normalized:
            simulated_image /= jnp.linalg.norm(simulated_image)

        if not get_real:
            simulated_image = rfftn(simulated_image)
        return simulated_image

    @override
    def sample(self, rng_key: PRNGKeyArray, *, get_real: bool = True) -> Array:
        raise NotImplementedError("This method is not implemented yet.")

    @override
    def log_likelihood(self, observed: Array) -> Array:
        N = observed.flatten().shape[0]
        signal = self.compute_signal(get_real=True)

        return (2 - N) * jnp.log(jnp.linalg.norm(signal - observed))
