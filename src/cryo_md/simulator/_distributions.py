from typing_extensions import override

import jax.numpy as jnp
import jax.random as jr
from equinox import field
from jaxtyping import Float, Complex, Array, PRNGKeyArray

from cryojax.inference.distributions import AbstractDistribution
from cryojax.simulator import AbstractImageModel
from cryojax.image import rfftn

from .._errors import error_if_not_positive


class WhiteGaussianNoise(AbstractDistribution, strict=True):
    r"""A gaussian noise model, where each fourier mode is independent.

    This computes the likelihood in Fourier space,
    so that the variance to be an arbitrary noise power spectrum.
    """

    image_model: AbstractImageModel
    noise_variance: Float | Array
    normalizes_signal: bool = field(static=True)

    def __init__(
        self,
        image_model: AbstractImageModel,
        noise_variance: Float[Array, ""],  # noqa: F722
        normalizes_signal: bool = True,
    ):
        """**Arguments:**

        - `image_model`: The image formation model.
        - `noise_variance`: The variance of the noise in fourier space.
        """  # noqa: E501
        self.image_model = image_model
        self.noise_variance = error_if_not_positive(noise_variance)
        self.normalizes_signal = normalizes_signal

        if not self.normalizes_signal:
            raise NotImplementedError("A non-normalized signal is not yet supported.")

    @override
    def compute_signal(
        self, *, outputs_real_space: bool = True
    ) -> Float[
        Array,
        "{self.image_model.instrument_config.y_dim} "  # noqa: F722
        "{self.image_model.instrument_config.x_dim}",  # noqa: F722
    ]:
        """Render the image formation model."""

        simulated_image = self.image_model.render(outputs_real_space=True)
        return simulated_image / jnp.linalg.norm(simulated_image)

    def compute_noise(
        self, rng_key: PRNGKeyArray, *, outputs_real_space: bool = True
    ) -> Float[
        Array,
        "{self.image_model.instrument_config.y_dim} "  # noqa: F722
        "{self.image_model.instrument_config.x_dim}",  # noqa: F722
    ]:
        pipeline = self.image_model
        # Compute the zero mean variance and scale up to be independent of the number of
        # pixels
        noise = jr.normal(rng_key, shape=pipeline.instrument_config.shape) * jnp.sqrt(
            self.noise_variance
        )

        return noise

    @override
    def sample(
        self, rng_key: PRNGKeyArray, *, outputs_real_space: bool = True
    ) -> Float[
        Array,
        "{self.image_model.instrument_config.y_dim} "  # noqa: F722
        "{self.image_model.instrument_config.x_dim}",  # noqa: F722
    ]:
        """Sample from the gaussian noise model."""

        noisy_image = self.compute_signal(outputs_real_space=outputs_real_space) + self.compute_noise(
            rng_key, outputs_real_space=outputs_real_space
        )
        return noisy_image

    @override
    def log_likelihood(
        self,
        observed: Complex[
            Array,
            "{self.image_model.instrument_config.y_dim} "  # noqa: F722
            "{self.image_model.instrument_config.x_dim//2+1}",  # noqa: F722
        ],
    ) -> Float:
        """Evaluate the log-likelihood of the gaussian noise model.

        **Arguments:**

        - `observed` : The observed data in fourier space.
        """
        # Create simulated data
        simulated = self.compute_signal(outputs_real_space=True)

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
    image_model: AbstractImageModel
    normalizes_signal: bool = field(static=True)

    def __init__(
        self,
        image_model: AbstractImageModel,
        normalizes_signal: bool = True,
    ):
        """**Arguments:**

        - `image_model`: The image formation model.
        """
        self.image_model = image_model
        self.normalizes_signal = normalizes_signal

    @override
    def compute_signal(
        self, *, outputs_real_space: bool = True
    ) -> Float[
        Array,
        "{self.image_model.instrument_config.y_dim} "  # noqa: F722
        "{self.image_model.instrument_config.x_dim}",  # noqa: F722
    ]:
        """Render the image formation model."""

        simulated_image = self.image_model.render(outputs_real_space=True)

        if self.normalizes_signal:
            simulated_image /= jnp.linalg.norm(simulated_image)

        if not outputs_real_space:
            simulated_image = rfftn(simulated_image)
        return simulated_image

    @override
    def sample(self, rng_key: PRNGKeyArray, *, outputs_real_space: bool = True) -> Array:
        raise NotImplementedError("This method is not implemented yet.")

    @override
    def log_likelihood(self, observed: Array) -> Array:
        N = observed.flatten().shape[0]
        signal = self.compute_signal(outputs_real_space=True)

        cc = jnp.mean(signal**2)
        co = jnp.mean(observed * signal)
        c = jnp.mean(signal)
        o = jnp.mean(observed)

        scale = (co - c * o) / (cc - c ** 2)
        bias = o - scale * c

        return (2 - N) * jnp.log(jnp.linalg.norm(scale * signal - observed + bias))
