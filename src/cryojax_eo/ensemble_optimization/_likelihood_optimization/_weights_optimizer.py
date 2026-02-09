"""
Weight and position optimizers for ensemble refinement.
"""

from typing import cast
from typing_extensions import override

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import jax_dataloader as jdl
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex
from jaxtyping import Array, Float, Int

from .._pose_search import HierarchicalSO3GridSearch
from . import ImagesToEnsembleLikelihoodFn
from ._utils import _estimate_poses
from .base_optimizer import AbstractEnsembleParameterOptimizer


class ProjGradDescWeightOptimizer(AbstractEnsembleParameterOptimizer):
    n_steps: Int
    ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn
    pose_search: HierarchicalSO3GridSearch | None

    def __init__(
        self,
        n_steps: Int,
        ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn,
        pose_search: HierarchicalSO3GridSearch | None = None,
    ):
        self.n_steps = n_steps
        self.ensemble_likelihood_fn = ensemble_likelihood_fn
        self.pose_search = pose_search

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
            walkers, dataloader, self.ensemble_likelihood_fn, self.pose_search
        )
        weights = _optimize_weights(weights, likelihood_matrix, self.n_steps)
        return weights


@eqx.filter_jit
def _optimize_weights(
    weights: Float[Array, " n_walkers"],
    likelihood_matrix: Float[Array, "n_images n_walkers"],
    n_steps: Int = 500,
) -> Float[Array, " n_walkers"]:
    pg = ProjectedGradient(
        fun=_compute_neg_log_likelihood,
        projection=projection_simplex,
        maxiter=n_steps,
    )
    return pg.run(weights, likelihood_matrix=likelihood_matrix).params


def _compute_full_likelihood_matrix(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    dataloader: jdl.DataLoader,
    ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn,
    pose_search: HierarchicalSO3GridSearch | None,
) -> Array:
    """
    Compute the full likelihood matrix for the given walkers and dataloader.
    """

    compute_likelihood_matrix_fn = eqx.filter_jit(
        ensemble_likelihood_fn.compute_log_likelihood_matrix
    )
    estimate_pose_fn = (
        eqx.filter_jit(_estimate_poses) if pose_search is not None else None
    )

    shuffle = dataloader.dataloader.shuffle  # save the original shuffle state
    dataloader.dataloader.shuffle = False
    # Compute the likelihood matrix for each batch in the dataloader
    likelihood_matrix = []
    for batch in dataloader:
        if pose_search is None:
            poses_per_walker = jax.tree.map(
                lambda x: jnp.repeat(x[None, :], repeats=walkers.shape[0], axis=0),
                batch["particle_stack"]["parameters"]["pose"],
            )
            poses_per_walker = cast(cxs.EulerAnglePose, poses_per_walker)
        else:
            assert estimate_pose_fn is not None

            poses_per_walker = estimate_pose_fn(
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

        lklhood_matrix = compute_likelihood_matrix_fn(
            walkers,
            batch["particle_stack"]["images"],
            batch["particle_stack"]["parameters"]["image_config"],
            poses_per_walker,
            batch["particle_stack"]["parameters"]["transfer_theory"],
            batch["per_particle_args"],
        )
        likelihood_matrix.append(lklhood_matrix)

    # restore the original shuffle state
    dataloader.dataloader.shuffle = shuffle

    # Concatenate the likelihood matrices from all batches
    return jnp.concatenate(likelihood_matrix, axis=0)


def _compute_neg_log_likelihood(
    weights: Float[Array, " n_walkers"],
    likelihood_matrix: Float[Array, "n_images n_walkers"],
) -> Float:
    """
    Compute the negative log likelihood of the data given the weights and the
    likelihood matrix.

    **Arguments:**
        weights: The weights of the walkers.
        likelihood_matrix: The likelihood matrix for each image and walker.

    **Returns:**
        The negative log likelihood of the data given the weights and the
        likelihood matrix.
    """
    log_lklhood = jax.scipy.special.logsumexp(
        a=likelihood_matrix, b=weights[None, :], axis=1
    )
    return -jnp.mean(log_lklhood)
