"""
Weight and position optimizers for ensemble refinement.
"""

import logging
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import jax_dataloader as jdl
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex
from jaxtyping import Array, Float, Int

from .._pose_search import HierarchicalSO3GridSearch
from . import ImagesToEnsembleLikelihoodFn
from ._mult_grad_weight_opt import multiplicative_gradient
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
        likelihood_matrix = (
            self.ensemble_likelihood_fn.compute_full_log_likelihood_matrix(
                walkers, dataloader, prints_progress=True
            )
        )
        return self.optimize_with_precomputed_likelihood_matrix(
            likelihood_matrix, weights
        )

    def optimize_with_precomputed_likelihood_matrix(
        self,
        likelihood_matrix: Float[Array, "n_images n_walkers"],
        init_weights: None | Float[Array, " n_walkers"] = None,
    ):
        """
        Optimize the weights of the walkers using projected gradient descent
        using a precomputed likelihood matrix.

        **Arguments:**
            weights: The current weights of the walkers.
            likelihood_matrix: The precomputed likelihood matrix for each image
                and walker.

        **Returns:**
            The optimized weights of the walkers.
        """
        if init_weights is None:
            init_weights = (
                jnp.ones(likelihood_matrix.shape[1]) / likelihood_matrix.shape[1]
            )
        weights = _optimize_weights(
            init_weights, likelihood_matrix, self.ensemble_likelihood_fn, self.n_steps
        )
        return weights


@eqx.filter_jit
def _optimize_weights(
    weights: Float[Array, " n_walkers"],
    log_likelihood_matrix: Float[Array, "n_images n_walkers"],
    ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn,
    n_steps: Int = 500,
) -> Float[Array, " n_walkers"]:
    def loss_fn(w, llm):
        return -ensemble_likelihood_fn.compute_from_log_likelihood_matrix(llm, w)

    pg = ProjectedGradient(
        fun=loss_fn,
        projection=projection_simplex,
        maxiter=n_steps,
    )
    return pg.run(weights, llm=log_likelihood_matrix).params


def optimize_weights(
    log_likelihood_matrix: Float[Array, "n_images n_walkers"],
    max_iter: int = 500,
    tol: float = 1e-2,
) -> Float[Array, " n_walkers"]:
    # if init_weights is None:
    #     init_weights = (
    #         jnp.ones(log_likelihood_matrix.shape[1]) / log_likelihood_matrix.shape[1]
    #     )

    # def _loss_fn(w, llm):
    #     return -jnp.mean(jax.scipy.special.logsumexp(a=llm, b=w[None, :], axis=1))

    # pg = ProjectedGradient(
    #     fun=_loss_fn,
    #     projection=projection_simplex,
    #     maxiter=max_iter,
    # )
    # return pg.run(init_weights, llm=log_likelihood_matrix).params

    weights, n_iter, final_gap = multiplicative_gradient(
        log_likelihood_matrix,
        max_iter=jnp.asarray(max_iter),
        tol=jnp.asarray(tol),
    )
    if n_iter == max_iter:
        logging.info(
            f"Optimization did not converge after {max_iter} iterations. "
            f"Final gap: {final_gap:.4f}. Consider increasing max_iter or tol."
        )
    else:
        logging.info(
            f"Optimization converged after {n_iter} iterations. "
            f"Final gap: {final_gap:.4f}."
        )
    return weights
