from abc import abstractmethod
from typing import Any, Tuple, Optional
import pathlib

from equinox import AbstractVar, Module
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .._likelihood_optimization.base_optimizer import AbstractEnsembleParameterOptimizer
from .._prior_projection.base_prior_projector import AbstractEnsemblePriorProjector

class AbstractEnsembleRefinementPipeline(Module, strict=True):
    """
    Abstract class for ensemble refinement pipelines.
    """

    prior_projector: AbstractVar[AbstractEnsemblePriorProjector]
    likelihood_optimizer: AbstractVar[AbstractEnsembleParameterOptimizer]
    n_steps: AbstractVar[Int]

    @abstractmethod
    def run(
        self,
        key: PRNGKeyArray,
        initial_walkers: Float[Array, "n_walkers n_atoms 3"],
        initial_weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        *,
        output_directory: str | pathlib.Path,
        initial_state_for_projector: Any = None,
        
    ) -> Tuple[
        Float[Array, "n_steps n_walkers n_atoms 3"],
        Float[Array, "n_steps n_walkers"],
    ]:
        raise NotImplementedError
