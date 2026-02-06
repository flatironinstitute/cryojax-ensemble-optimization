"""
Base optimizer object for ensemble refinement.
"""

from abc import abstractmethod

from equinox import AbstractVar, Module
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int

from .._pose_search import HierarchicalSO3GridSearch
from ._loss_functions import AbstractImagesToEnsembleLikelihoodFn


class AbstractEnsembleParameterOptimizer(Module):
    """Abstract interface for objects that optimize parameters
    of an ensemble of structures.
    """

    n_steps: AbstractVar[Int]
    ensemble_likelihood_fn: AbstractVar[AbstractImagesToEnsembleLikelihoodFn]
    pose_search: AbstractVar[HierarchicalSO3GridSearch | None]

    @abstractmethod
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
    ) -> Float:
        raise NotImplementedError
