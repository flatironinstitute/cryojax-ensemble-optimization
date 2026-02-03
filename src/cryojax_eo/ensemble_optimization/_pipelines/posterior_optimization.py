import os
from collections.abc import Callable
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
import mdtraj
import optax
from jax_dataloader import DataLoader
from jaxtyping import Array, Float
from tqdm import tqdm

from .._likelihood_optimization import LikelihoodOptimalWeightsFn
from .._likelihood_optimization.optimizers import (
    ProjGradDescWeightOptimizer,
)
from .._prior_projection import AbstractForceField


class PosteriorOptimizer(eqx.Module):
    """
    A class for posterior optimization. This class optimizes the log-posterior
    defined as

        $$log P(X|D) ~ \\gamma log P(D|X) + log P(X)$$

    where $\\gamma$ is a weighting factor between the likelihood and the prior.

    The optimization is performed using the Adabelief optimizer from Optax.
    """

    prior_fn: Callable
    likelihood_fn: LikelihoodOptimalWeightsFn
    runs_postprocessing: bool = True

    def __init__(
        self,
        forcefield: AbstractForceField,
        likelihood_fn: LikelihoodOptimalWeightsFn,
        runs_postprocessing: bool = True,
    ):
        """
        Initialize the PosteriorOptimizer.

        **Arguments:**
        - `forcefield`: An instance of `AbstractForceField` representing the prior.
        - `likelihood_fn`: A callable that computes the likelihood and its gradient.
        - `runs_postprocessing`: Whether to run postprocessing after optimization.
            The postprocessing step optimizes the weights of the ensemble
            members using all the data.
        """
        self.prior_fn = lambda x: forcefield(x)
        self.likelihood_fn = likelihood_fn
        self.runs_postprocessing = runs_postprocessing

    def _compute_gradient_and_weights(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weight_likelihood_parameter: float,
        dataloader: jdl.DataLoader,
        n_batches: int,
    ):
        likelihood_grad, weights = _compute_likelihood_gradient_and_weights(
            walkers,
            weights,
            dataloader,
            n_batches,
            self.likelihood_fn,
        )

        prior_grad = _compute_prior_gradient(walkers, self.prior_fn)
        prior_grad /= jnp.linalg.norm(prior_grad, axis=(1, 2), keepdims=True)

        gradients = weight_likelihood_parameter * likelihood_grad + prior_grad
        norms = jnp.linalg.norm(gradients, axis=(2), keepdims=True)
        norms = jnp.where(norms < 1e-12, 1.0, norms)

        return gradients / norms, weights

    @eqx.filter_jit
    def _make_step(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        gradients: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        opt_state: optax.OptState,
        solver: optax.GradientTransformation,
    ):
        updates, opt_state = solver.update(gradients, opt_state, walkers)
        walkers = optax.apply_updates(walkers, updates)

        return walkers, opt_state

    def optimize(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        dataloader: jdl.DataLoader,
        n_steps: int,
        n_batches_per_step: int,
        step_size: optax.Schedule,
        weight_likelihood_parameter: optax.Schedule,
        *,
        output_directory: str,
    ):
        """
        Optimize the given walkers and weights with the data from the dataloader.

        **Arguments:**
        - `walkers`: The initial walkers to optimize.
        - `weights`: The initial weights of the walkers.
        - `dataloader`: A dataloader that provides the data for likelihood computation.
        - `n_steps`: The number of optimization steps to perform.
        - `n_batches_per_step`: The number of batches to use per optimization step.
        - `step_size`: The step size for the optimizer.
        - `weight_likelihood_parameter`: The weighting factor for the likelihood term.
        - `output_directory`: The directory to save the trajectories and final ensemble.
        """
        solver = optax.adabelief(learning_rate=step_size, nesterov=True)
        opt_state = solver.init(walkers)

        writers = [
            mdtraj.formats.XYZTrajectoryFile(
                os.path.join(output_directory, f"traj_walker_{i}.xyz"), "w"
            )
            for i in range(walkers.shape[0])
        ]
        for i in range(walkers.shape[0]):
            writers[i].write(walkers[i][None, ...] / 10.0)

        progress_bar = tqdm(range(n_steps), desc="Posterior Optimization", leave=True)
        for i in progress_bar:
            gradients, weights = self._compute_gradient_and_weights(
                walkers,
                weights,
                weight_likelihood_parameter(i),
                dataloader,
                n_batches_per_step,
            )

            walkers, opt_state = self._make_step(
                walkers,
                gradients,
                opt_state,
                solver,
            )
            for i in range(walkers.shape[0]):
                writers[i].write(walkers[i][None, ...] / 10.0)

        for writer in writers:
            writer.close()

        if self.runs_postprocessing:
            weight_optimizer = ProjGradDescWeightOptimizer(
                n_steps=500,
                likelihood_fn=self.likelihood_fn,
            )
            walkers, weights = self.postprocess(
                walkers, weights, dataloader, weight_optimizer
            )

        jnp.savez(
            os.path.join(output_directory, "optimized_ensemble.npz"),
            walkers=walkers,
            weights=weights,
        )

        return walkers, weights

    def postprocess(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        weight_optimizer: ProjGradDescWeightOptimizer,
    ):
        """
        Postprocess the walkers and weights.
        """
        # Project the weights
        weights = weight_optimizer(
            walkers,
            weights,
            dataloader,
        )

        return walkers, weights


def _compute_likelihood_gradient_and_weights(
    walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    dataloader,
    n_batches,
    likelihood_fn: LikelihoodOptimalWeightsFn,
):
    gradients = jnp.zeros_like(walkers)
    opt_weights = jnp.zeros_like(weights)
    for _ in range(n_batches):
        relion_batch = next(iter(dataloader))
        batch_gradients, batch_weights = _compute_likelihood_gradient_and_weights_batch(
            walkers, weights, relion_batch, likelihood_fn
        )
        gradients += batch_gradients
        opt_weights += batch_weights
    return gradients / n_batches, opt_weights / n_batches


@eqx.filter_jit
def _compute_likelihood_gradient_and_weights_batch(
    walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    relion_batch,
    likelihood_fn: LikelihoodOptimalWeightsFn,
):
    @partial(jax.grad, has_aux=True)
    def _compute_likelihood_grad(walkers, weights, relion_batch):
        return likelihood_fn(walkers, weights, relion_batch)

    """
    Compute the likelihood gradient and weights for a batch of walkers.
    """
    batch_gradients, batch_weights = _compute_likelihood_grad(
        walkers, weights, relion_batch
    )
    return batch_gradients, batch_weights


@eqx.filter_jit
def _compute_prior_gradient(walkers, prior_fn):
    prior_grad = jax.vmap(jax.grad(prior_fn))(walkers)
    return prior_grad
