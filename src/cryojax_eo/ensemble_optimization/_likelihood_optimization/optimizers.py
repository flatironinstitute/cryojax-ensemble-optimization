"""
Weight and position optimizers for ensemble refinement.
"""

from typing import Tuple
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
from jaxtyping import Array, Float, Int

from cryojax_eo.typing import ParticleStackInfo, PerParticleT

from ._loss_functions.ensemble_losses import compute_likelihood_matrix
from ._loss_functions.likelihood_wrappers import (
    _optimize_weights,
    AbstractLikelihoodFn,
    LikelihoodFn,
    LikelihoodOptimalWeightsFn,
)
from .base_optimizer import AbstractEnsembleParameterOptimizer


class ProjGradDescWeightOptimizer(AbstractEnsembleParameterOptimizer):
    n_steps: Int
    likelihood_fn: LikelihoodFn

    def __init__(
        self,
        n_steps: Int,
        likelihood_fn: LikelihoodFn,
    ):
        self.n_steps = n_steps
        self.likelihood_fn = likelihood_fn

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
                - `amplitudes`: The Gaussian amplitudes for each atom.
                - `variances`: The Gaussian variances for each atom.
                - `noise_variance`: The noise variance for the data. If None, the
                    noise variance is marginalized.

        **Returns:**
            The optimized weights of the walkers.
        """
        likelihood_matrix = _compute_full_likelihood_matrix(
            walkers, dataloader, self.likelihood_fn
        )
        weights = _optimize_weights(weights, likelihood_matrix, self.n_steps)
        return weights


class IterativeEnsembleLikelihoodOptimizer(AbstractEnsembleParameterOptimizer):
    step_size: Float
    n_steps: Int
    n_batches_per_step: Int
    likelihood_fn: LikelihoodOptimalWeightsFn

    def __init__(
        self,
        step_size: Float,
        n_steps: Int,
        n_batches_per_step: Int,
        likelihood_fn: LikelihoodOptimalWeightsFn,
    ):
        self.step_size = step_size
        self.n_steps = n_steps
        self.n_batches_per_step = n_batches_per_step
        self.likelihood_fn = likelihood_fn

    @override
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: jdl.DataLoader,
    ):
        for _ in range(self.n_steps):
            gradients = jnp.zeros_like(walkers)
            weights = jnp.ones_like(weights) / weights.shape[0]

            for _ in range(self.n_batches_per_step):
                batch = next(iter(dataloader))
                tmp_grads, tmp_weights = _compute_ensemble_gradients(
                    walkers,
                    weights,
                    batch["particle_stack"],
                    batch["per_particle_args"],
                    self.likelihood_fn,
                )
                gradients += tmp_grads
                weights += tmp_weights
            gradients /= self.n_batches_per_step
            weights /= self.n_batches_per_step

            norms = jnp.linalg.norm(gradients, axis=(2), keepdims=True)
            norms = jnp.where(norms < 1e-12, 1.0, norms)

            # norms = (
            #     jnp.linalg.norm(gradients, axis=(1, 2), keepdims=True)
            #     / gradients.shape[1]
            # )
            gradients /= norms

            walkers = walkers - self.step_size * gradients
        return walkers, weights


@eqx.filter_jit
def _compute_ensemble_gradients(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    relion_stack: ParticleStackInfo,
    per_particle_args: PerParticleT,
    likelihood_fn: LikelihoodOptimalWeightsFn,
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

    def _loss_fn(walkers, weights, relion_stack, per_particle_args):
        return likelihood_fn(walkers, weights, relion_stack, per_particle_args)

    return jax.grad(_loss_fn, argnums=0, has_aux=True)(
        walkers, weights, relion_stack, per_particle_args
    )


def _compute_full_likelihood_matrix(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    dataloader: jdl.DataLoader,
    likelihood_fn: AbstractLikelihoodFn,
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
            likelihood_fn.amplitudes,
            likelihood_fn.variances,
            likelihood_fn.image_to_walker_log_likelihood_fn,
            likelihood_fn.dilated_mask,
            likelihood_fn.estimates_pose,
            constant_args=likelihood_fn.loss_fn_constant_args,
            per_particle_args=batch["per_particle_args"],
        )
        likelihood_matrix.append(lklhood_matrix)

    # restore the original shuffle state
    dataloader.dataloader.shuffle = shuffle

    # Concatenate the likelihood matrices from all batches
    return jnp.concatenate(likelihood_matrix, axis=0)
