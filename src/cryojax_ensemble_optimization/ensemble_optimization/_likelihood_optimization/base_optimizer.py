"""
Base optimizer object for ensemble refinement.
"""

from abc import abstractmethod

from equinox import AbstractVar, Module
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int

from ._loss_functions import AbstractLikelihoodFn


class AbstractEnsembleParameterOptimizer(Module):
    """Abstract interface for objects that optimize parameters
    of an ensemble of structures.
    """

    n_steps: AbstractVar[Int]
    likelihood_fn: AbstractVar[AbstractLikelihoodFn]

    @abstractmethod
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
    ) -> Float:
        raise NotImplementedError
