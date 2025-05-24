"""
Base optimizer object for ensemble refinement.
"""

from abc import abstractmethod

from equinox import AbstractVar, Module
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int

from ..._custom_types import LossFn


class AbstractEnsembleParameterOptimizer(Module, strict=True):
    """Abstract interface for objects that optimize parameters
    of an ensemble of structures.
    """

    gaussian_variances: AbstractVar[
        Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    ]
    gaussian_amplitudes: AbstractVar[
        Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    ]
    n_steps: AbstractVar[Int]
    image_to_walker_log_likelihood_fn: AbstractVar[LossFn]

    @abstractmethod
    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
    ) -> Float:
        raise NotImplementedError
