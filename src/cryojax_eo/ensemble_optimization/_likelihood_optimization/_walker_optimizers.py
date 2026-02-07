"""
Weight and position optimizers for ensemble refinement.
"""

from typing import Any, cast
from typing_extensions import override

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
from jaxtyping import Array, Float

from .._pose_search import HierarchicalSO3GridSearch
from . import ImagesToEnsembleLikelihoodFn
from ._utils import _estimate_poses
from ._weights_optimizer import _optimize_weights
from .base_optimizer import AbstractEnsembleParameterOptimizer


class IterativeEnsembleLikelihoodOptimizer(AbstractEnsembleParameterOptimizer):
    step_size: Float
    n_steps: int
    n_batches_per_step: int
    ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn
    pose_search: HierarchicalSO3GridSearch | None

    def __init__(
        self,
        step_size: Float,
        n_steps: int,
        n_batches_per_step: int,
        ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn,
        pose_search: HierarchicalSO3GridSearch | None = None,
    ):
        assert step_size > 0, "Step size must be positive."
        assert n_steps > 0, "Number of steps must be positive."
        assert n_batches_per_step > 0, "Number of batches per step must be positive."

        self.step_size = step_size
        self.n_steps = n_steps
        self.n_batches_per_step = n_batches_per_step
        self.ensemble_likelihood_fn = ensemble_likelihood_fn
        self.pose_search = pose_search

    @override
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: jdl.DataLoader,
    ) -> tuple[float, Float[Array, "n_walkers n_atoms 3"], Float[Array, " n_walkers"]]:
        loss = 0.0
        for _ in range(self.n_steps):
            gradients = jnp.zeros_like(walkers)
            weights = jnp.ones_like(weights) / weights.shape[0]
            loss = 0.0
            for _ in range(self.n_batches_per_step):
                batch = next(iter(dataloader))
                if self.pose_search is None:
                    poses_per_walker = jax.tree.map(
                        lambda x: jnp.repeat(
                            x[None, :], repeats=walkers.shape[0], axis=0
                        ),
                        batch["particle_stack"]["parameters"]["pose"],
                    )
                    poses_per_walker = cast(cxs.EulerAnglePose, poses_per_walker)
                else:
                    poses_per_walker = _estimate_poses_per_walker(
                        walkers,
                        batch,
                        self.ensemble_likelihood_fn,
                        self.pose_search,
                    )

                tmp_loss, tmp_grads, tmp_weights = _compute_ensemble_gradients(
                    walkers,
                    weights,
                    batch["particle_stack"]["images"],
                    batch["particle_stack"]["parameters"]["image_config"],
                    poses_per_walker,
                    batch["particle_stack"]["parameters"]["transfer_theory"],
                    batch["per_particle_args"],
                    self.ensemble_likelihood_fn,
                )
                gradients += tmp_grads
                weights += tmp_weights
                loss += tmp_loss

            gradients /= self.n_batches_per_step
            weights /= self.n_batches_per_step
            loss /= self.n_batches_per_step

            norms = jnp.linalg.norm(gradients, axis=(2), keepdims=True)
            norms = jnp.where(norms < 1e-12, 1.0, norms)

            # norms = (
            #     jnp.linalg.norm(gradients, axis=(1, 2), keepdims=True)
            #     / gradients.shape[1]
            # )
            gradients /= norms

            walkers = walkers - self.step_size * gradients
        return loss, walkers, weights


def _estimate_poses_per_walker(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    batch: Any,
    ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn,
    pose_search: HierarchicalSO3GridSearch,
):
    return _estimate_poses(
        walkers,
        ensemble_likelihood_fn.image_to_walker_likelihood_fn.amplitudes,
        ensemble_likelihood_fn.image_to_walker_likelihood_fn.variances,
        batch["particle_stack"]["images"],
        batch["particle_stack"]["parameters"]["image_config"],
        batch["particle_stack"]["parameters"]["transfer_theory"],
        pose_search,
        n_walkers_in_parallel=ensemble_likelihood_fn.n_walkers_in_parallel,
        n_images_in_parallel=ensemble_likelihood_fn.n_images_in_parallel,
    )


@eqx.filter_jit
def _compute_ensemble_gradients(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    weights: Float[Array, " n_walkers"],
    images: Float[Array, "n_images y x"],
    image_config: cxs.BasicImageConfig,
    poses_per_walker: cxs.AbstractPose,
    transfer_theories: cxs.ContrastTransferTheory,
    per_particle_args: Any,
    ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn,
) -> tuple[
    float,
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
        images: The images to optimize against.
        image_config: Configuration for the images.
        poses_per_walker: The poses associated with each walker.
        transfer_theories: The contrast transfer theories.
        per_particle_args: Additional arguments for each particle.
        ensemble_likelihood_fn: The likelihood function for the ensemble.

    **Returns:**
        The optimized walkers and weights of the ensemble.
    """

    def _loss_fn(walkers, weights):
        log_likelihood_matrix = ensemble_likelihood_fn.compute_log_likelihood_matrix(
            walkers,
            images,
            image_config,
            poses_per_walker,
            transfer_theories,
            per_particle_args,
        )
        weights = jax.nn.softmax(_optimize_weights(weights, log_likelihood_matrix))
        log_lklhood = jax.scipy.special.logsumexp(
            a=log_likelihood_matrix, b=weights[None, :], axis=1
        )
        return -jnp.mean(log_lklhood), weights

    (loss, weights), grads = jax.value_and_grad(_loss_fn, argnums=0, has_aux=True)(
        walkers,
        weights,
    )
    return loss, grads, weights
