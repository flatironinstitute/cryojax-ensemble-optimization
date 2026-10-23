"""
Base optimizer object for ensemble refinement.
"""

from abc import abstractmethod

from equinox import AbstractVar, Module
from jaxtyping import Array, Float

from .._pose_search import HierarchicalSO3GridSearch
from ..dataset import ParticleDataLoader
from ._loss_functions import AbstractImagesToEnsembleLikelihoodFn


class AbstractEnsembleParameterOptimizer(Module):
    """Abstract interface for objects that optimize parameters
    of an ensemble of structures.
    """

    n_steps: AbstractVar[int]
    ensemble_likelihood_fn: AbstractVar[AbstractImagesToEnsembleLikelihoodFn]
    pose_search: AbstractVar[HierarchicalSO3GridSearch | None]

    @abstractmethod
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: ParticleDataLoader,
    ) -> Float:
        raise NotImplementedError
