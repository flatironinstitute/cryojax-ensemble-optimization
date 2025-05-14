"""
Weight and position optimizers for ensemble refinement.
"""

from functools import partial
from typing import Tuple
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
from cryojax.data import ParticleStack
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex
from jaxtyping import Array, Float

from .base_optimizer import AbstractEnsembleParameterOptimizer
from .loss_functions import (
    compute_likelihood_matrix,
    neg_log_likelihood_from_walkers_and_weights,
    neg_log_likelihood_from_weights,
)


class ProjGradDescWeightOptimizer(AbstractEnsembleParameterOptimizer, strict=True):
    @override
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: jdl.DataLoader,
        args: Tuple[
            Float[Array, "n_atoms n_gaussians_per_atom"],
            Float[Array, "n_atoms n_gaussians_per_atom"],
            Float | None,
        ],
    ):
        """
        Optimize the weights of the walkers using projected gradient descent using all images.

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
        likelihood_matrix = _compute_full_likelihood_matrix(walkers, dataloader, args)
        weights = _optimize_weights(weights, likelihood_matrix)
        return weights


class SteepestDescWalkerPositionsOptimizer(
    AbstractEnsembleParameterOptimizer, strict=True
):
    step_size: Float
    n_steps: int

    def __init__(self, step_size: Float, n_steps: int):
        assert step_size > 0, "Step size must be greater than 0"
        assert n_steps > 0, "Number of steps must be greater than 0"

        self.step_size = step_size
        self.n_steps = n_steps

    @override
    def __call__(self, walkers, weights, dataloader, args):
        for i in range(self.n_steps):
            batch = next(iter(dataloader))
            walkers = _optimize_walkers_positions(
                walkers, weights, batch, self.step_size, args
            )

        return walkers


class IterativeEnsembleOptimizer(AbstractEnsembleParameterOptimizer):
    step_size: Float
    n_steps: int

    def __init__(self, step_size: Float, n_steps: int):
        self.step_size = step_size
        self.n_steps = n_steps

    @override
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: jdl.DataLoader,
        args,
    ):
        for _ in range(self.n_steps):
            batch = next(iter(dataloader))
            walkers, weights = _optimize_ensemble(
                walkers, weights, batch, self.step_size, args
            )
        return walkers, weights


@eqx.filter_jit
def _optimize_weights(
    weights: Float[Array, " n_walkers"],
    likelihood_matrix: Float[Array, "n_images n_walkers"],
) -> Float[Array, " n_walkers"]:
    pg = ProjectedGradient(
        fun=neg_log_likelihood_from_weights, projection=projection_simplex
    )
    return pg.run(weights, likelihood_matrix=likelihood_matrix).params


@eqx.filter_jit
def _optimize_walkers_positions(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    relion_stack: ParticleStack,
    step_size: Float,
    args: Tuple[
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float | None,
    ],
) -> Float[Array, "n_walkers n_atoms 3"]:
    gradients = jax.grad(
        neg_log_likelihood_from_walkers_and_weights,
        argnums=0,
    )(walkers, weights, relion_stack, args)

    norms = jnp.linalg.norm(gradients, axis=(2), keepdims=True)
    # set small norms to 1 (avoid making small gradients large!)
    norms = jnp.where(norms < 1e-12, 1.0, norms)
    gradients /= norms

    return walkers - step_size * gradients


@eqx.filter_jit
def _optimize_ensemble(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    relion_stack: ParticleStack,
    step_size: Float,
    args: Tuple[
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float | None,
    ],
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
        likelihood_matrix = compute_likelihood_matrix(walkers, relion_stack, *args)
        weights = _optimize_weights(weights, likelihood_matrix)
        weights = jax.nn.softmax(weights)
        loss = neg_log_likelihood_from_weights(weights, likelihood_matrix)
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
    args: Tuple[
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float | None,
    ],
) -> Array:
    """
    Compute the full likelihood matrix for the given walkers and dataloader.
    """

    shuffle = dataloader.dataloader.shuffle # save the original shuffle state
    dataloader.dataloader.shuffle = False
    # Compute the likelihood matrix for each batch in the dataloader
    likelihood_matrix = []
    for batch in dataloader:
        lklhood_matrix = compute_likelihood_matrix(walkers, batch, *args)
        likelihood_matrix.append(lklhood_matrix)

    # restore the original shuffle state
    dataloader.dataloader.shuffle = shuffle
    
    # Concatenate the likelihood matrices from all batches
    return jnp.concatenate(likelihood_matrix, axis=0)
