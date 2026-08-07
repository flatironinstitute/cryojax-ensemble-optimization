"""
Weight and position optimizers for ensemble refinement.
"""

import logging
from typing_extensions import override

import equinox as eqx
import jax.numpy as jnp
import jax_dataloader as jdl
from jaxtyping import Array, Float, Int

from .._pose_search import HierarchicalSO3GridSearch
from . import ImagesToEnsembleLikelihoodFn
from ._mult_grad_weight_opt import multiplicative_gradient


class MultGradWeightOptimizer(eqx.Module):
    max_iter: int
    tol: float
    ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn
    pose_search: HierarchicalSO3GridSearch | None

    def __init__(
        self,
        ensemble_likelihood_fn: ImagesToEnsembleLikelihoodFn,
        pose_search: HierarchicalSO3GridSearch | None = None,
        max_iter: int = 500,
        tol: float = 1e-4,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.ensemble_likelihood_fn = ensemble_likelihood_fn  # type: ignore
        self.pose_search = pose_search

    @override
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        dataloader: jdl.DataLoader,
    ) -> Float[Array, " n_walkers"]:
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
            likelihood_matrix,
        )

    def optimize_with_precomputed_likelihood_matrix(
        self,
        likelihood_matrix: Float[Array, "n_images n_walkers"],
    ) -> Float[Array, " n_walkers"]:
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
        weights, _, _ = _optimize_weights(
            likelihood_matrix, jnp.asarray(self.max_iter), jnp.asarray(self.tol)
        )
        return weights


@eqx.filter_jit
def _optimize_weights(
    log_likelihood_matrix: Float[Array, "n_images n_walkers"],
    max_iter: Int[Array, ""],
    tol: Float[Array, ""],
) -> tuple[Float[Array, " n_walkers"], Int[Array, ""], Float[Array, ""]]:
    weights, n_iter, final_gap = multiplicative_gradient(
        log_likelihood_matrix,
        max_iter=jnp.asarray(max_iter),
        tol=jnp.asarray(tol),
    )
    return weights, n_iter, final_gap


def optimize_weights(
    log_likelihood_matrix: Float[Array, "n_images n_walkers"],
    max_iter: int = 500,
    tol: float = 1e-2,
) -> Float[Array, " n_walkers"]:

    weights, n_iter, final_gap = _optimize_weights(
        log_likelihood_matrix, max_iter=jnp.asarray(max_iter), tol=jnp.asarray(tol)
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
