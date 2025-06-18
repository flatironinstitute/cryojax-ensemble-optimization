"""
Weight and position optimizers for ensemble refinement.
"""

from functools import partial
from typing import Tuple
from typing_extensions import Literal, override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
from cryojax.data import RelionParticleStackDataset
from cryojax.internal import error_if_negative, error_if_not_positive
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex
from jaxtyping import Array, Float, Int

from ..._custom_types import LossFn, PerParticleArgs
from .base_optimizer import AbstractEnsembleParameterOptimizer
from .loss_functions import (
    _likelihood_isotropic_gaussian,
    _likelihood_isotropic_gaussian_marginalized,
    compute_likelihood_matrix,
    compute_neg_log_likelihood,
    compute_neg_log_likelihood_from_weights,
)


class ProjGradDescWeightOptimizer(AbstractEnsembleParameterOptimizer, strict=True):
    n_steps: Int
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    image_to_walker_log_likelihood_fn: LossFn

    def __init__(
        self,
        gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        image_to_walker_log_likelihood_fn: (
            Literal["iso_gaussian", "iso_gaussian_var_marg"] | LossFn
        ),
    ):
        self.n_steps = 1  # not used
        gaussian_variances = error_if_not_positive(gaussian_variances)
        gaussian_amplitudes = error_if_not_positive(gaussian_amplitudes)

        assert gaussian_variances.ndim == 3, (
            "gaussian_variances must have shape "
            + "(n_walkers, n_atoms, n_gaussians_per_atom)"
        )
        assert gaussian_amplitudes.ndim == 3, (
            "gaussian_amplitudes must have shape "
            + "(n_walkers, n_atoms, n_gaussians_per_atom)"
        )

        self.gaussian_variances = gaussian_variances
        self.gaussian_amplitudes = gaussian_amplitudes

        if image_to_walker_log_likelihood_fn == "iso_gaussian":
            self.image_to_walker_log_likelihood_fn = _likelihood_isotropic_gaussian
        elif image_to_walker_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_walker_log_likelihood_fn = (
                _likelihood_isotropic_gaussian_marginalized
            )
        else:
            self.image_to_walker_log_likelihood_fn = image_to_walker_log_likelihood_fn

    @override
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: jdl.DataLoader,
    ):
        """
        Optimize the weights of the walkers using projected gradient descent
        using all images.

        **Arguments:**
            walkers: The current positions of the walkers.
            weights: The current weights of the walkers.
            dataloader: The dataloader for the data.
            args: Additional arguments for the likelihood function.
            This should be a tuple with the following elements:
                - `gaussian_amplitudes`: The Gaussian amplitudes for each atom.
                - `gaussian_variances`: The Gaussian variances for each atom.
                - `noise_variance`: The noise variance for the data. If None, the
                    noise variance is marginalized.

        **Returns:**
            The optimized weights of the walkers.
        """
        likelihood_matrix = _compute_full_likelihood_matrix(
            walkers,
            dataloader,
            self.gaussian_amplitudes,
            self.gaussian_variances,
            image_to_walker_log_likelihood_fn=self.image_to_walker_log_likelihood_fn,
        )
        weights = _optimize_weights(weights, likelihood_matrix)
        return weights


class SteepestDescWalkerPositionsOptimizer(
    AbstractEnsembleParameterOptimizer, strict=True
):
    step_size: Float
    n_steps: Int
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    image_to_walker_log_likelihood_fn: LossFn

    def __init__(
        self,
        n_steps: Int,
        step_size: Float,
        gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        image_to_walker_log_likelihood_fn: (
            Literal["iso_gaussian", "iso_gaussian_var_marg"] | LossFn
        ),
    ):
        assert n_steps > 0, "n_steps must be positive"
        self.n_steps = n_steps
        self.gaussian_variances = error_if_not_positive(gaussian_variances)
        self.gaussian_amplitudes = error_if_not_positive(gaussian_amplitudes)

        if image_to_walker_log_likelihood_fn == "iso_gaussian":
            self.image_to_walker_log_likelihood_fn = _likelihood_isotropic_gaussian
        elif image_to_walker_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_walker_log_likelihood_fn = (
                _likelihood_isotropic_gaussian_marginalized
            )

        else:
            self.image_to_walker_log_likelihood_fn = image_to_walker_log_likelihood_fn

        self.step_size = error_if_negative(step_size)

    @override
    def __call__(self, walkers, weights, dataloader):
        for _ in range(self.n_steps):
            batch = next(iter(dataloader))
            walkers = _optimize_walkers_positions(
                walkers,
                weights,
                batch["particle_stack"],
                self.step_size,
                self.gaussian_amplitudes,
                self.gaussian_variances,
                image_to_walker_log_likelihood_fn=self.image_to_walker_log_likelihood_fn,
                per_particle_args=batch["per_particle_args"],
            )

        return walkers


class IterativeEnsembleLikelihoodOptimizer(AbstractEnsembleParameterOptimizer):
    step_size: Float
    n_steps: Int
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    image_to_walker_log_likelihood_fn: LossFn

    def __init__(
        self,
        step_size: Float,
        n_steps: Int,
        gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        image_to_walker_log_likelihood_fn: Literal[
            "iso_gaussian", "iso_gaussian_var_marg"
        ]
        | LossFn,
    ):
        self.step_size = step_size
        self.n_steps = n_steps
        self.gaussian_variances = error_if_not_positive(gaussian_variances)
        self.gaussian_amplitudes = error_if_not_positive(gaussian_amplitudes)
        if image_to_walker_log_likelihood_fn == "iso_gaussian":
            self.image_to_walker_log_likelihood_fn = _likelihood_isotropic_gaussian
        elif image_to_walker_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_walker_log_likelihood_fn = (
                _likelihood_isotropic_gaussian_marginalized
            )
        else:
            assert callable(image_to_walker_log_likelihood_fn), (
                "If `image_to_walker_log_likelihood_fn` is not 'iso_gaussian' or "
                + "'iso_gaussian_var_marg', it must be a callable function."
            )
            self.image_to_walker_log_likelihood_fn = image_to_walker_log_likelihood_fn

    @override
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: jdl.DataLoader,
    ):
        for _ in range(self.n_steps):
            batch = next(iter(dataloader))
            walkers, weights = _optimize_ensemble(
                walkers,
                weights,
                batch["particle_stack"],
                self.step_size,
                self.gaussian_amplitudes,
                self.gaussian_variances,
                image_to_walker_log_likelihood_fn=self.image_to_walker_log_likelihood_fn,
                per_particle_args=batch["per_particle_args"],
            )
        return walkers, weights


@eqx.filter_jit
def _optimize_weights(
    weights: Float[Array, " n_walkers"],
    likelihood_matrix: Float[Array, "n_images n_walkers"],
) -> Float[Array, " n_walkers"]:
    pg = ProjectedGradient(
        fun=compute_neg_log_likelihood_from_weights, projection=projection_simplex
    )
    return pg.run(weights, likelihood_matrix=likelihood_matrix).params


@eqx.filter_jit
def _optimize_walkers_positions(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    relion_stack: RelionParticleStackDataset,
    step_size: Float,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    *,
    image_to_walker_log_likelihood_fn: LossFn,
    per_particle_args: PerParticleArgs,
) -> Float[Array, "n_walkers n_atoms 3"]:
    gradients = jax.grad(
        compute_neg_log_likelihood,
        argnums=0,
    )(
        walkers,
        weights,
        relion_stack,
        gaussian_amplitudes,
        gaussian_variances,
        image_to_walker_log_likelihood_fn,
        per_particle_args,
    )

    norms = jnp.linalg.norm(gradients, axis=(2), keepdims=True)
    # set small norms to 1 (avoid making small gradients large!)
    norms = jnp.where(norms < 1e-12, 1.0, norms)
    gradients /= norms

    return walkers - step_size * gradients


@eqx.filter_jit
def _optimize_ensemble(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    relion_stack: RelionParticleStackDataset,
    step_size: Float,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    *,
    image_to_walker_log_likelihood_fn: LossFn,
    per_particle_args: PerParticleArgs,
) -> Tuple[
    Float[Array, "n_walkers n_atoms 3"],
    Float[Array, " n_walkers"],
]:
    """
    Optimize the walkers and weights of the ensemble. First, the weights
    are optimized through projected gradient descent, and then the walkers
    are optimized with steepest descent.

    **Arguments:**
        walkers: The current positions of the walkers.
        weights: The current weights of the walkers.
        relion_stack: The data to optimize against.
        step_size: The step size for the optimization.
        args: Additional arguments for the likelihood function.

    **Returns:**
        The optimized walkers and weights of the ensemble.
    """

    @partial(jax.grad, argnums=0, has_aux=True)
    def _loss_fn(walkers, weights):
        likelihood_matrix = compute_likelihood_matrix(
            walkers,
            relion_stack,
            gaussian_amplitudes,
            gaussian_variances,
            image_to_walker_log_likelihood_fn,
            per_particle_args,
        )
        weights = _optimize_weights(weights, likelihood_matrix)
        weights = jax.nn.softmax(weights)
        loss = compute_neg_log_likelihood_from_weights(weights, likelihood_matrix)
        return loss, weights

    gradients, weights = _loss_fn(walkers, weights)

    norms = jnp.linalg.norm(gradients, axis=(2), keepdims=True)
    # set small norms to 1 (avoid making small gradients large!)
    norms = jnp.where(norms < 1e-12, 1.0, norms)
    gradients /= norms
    walkers = walkers - step_size * gradients

    return walkers, weights


def _compute_full_likelihood_matrix(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    dataloader: jdl.DataLoader,
    gaussian_amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    *,
    image_to_walker_log_likelihood_fn: LossFn,
) -> Array:
    """
    Compute the full likelihood matrix for the given walkers and dataloader.
    """

    shuffle = dataloader.dataloader.shuffle  # save the original shuffle state
    dataloader.dataloader.shuffle = False
    # Compute the likelihood matrix for each batch in the dataloader
    likelihood_matrix = []
    for batch in dataloader:
        lklhood_matrix = compute_likelihood_matrix(
            walkers,
            batch["particle_stack"],
            gaussian_amplitudes,
            gaussian_variances,
            image_to_walker_log_likelihood_fn=image_to_walker_log_likelihood_fn,
            per_particle_args=batch["per_particle_args"],
        )
        likelihood_matrix.append(lklhood_matrix)

    # restore the original shuffle state
    dataloader.dataloader.shuffle = shuffle

    # Concatenate the likelihood matrices from all batches
    return jnp.concatenate(likelihood_matrix, axis=0)
