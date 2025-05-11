from abc import abstractmethod
from typing import Any, Tuple

from equinox import AbstractVar, Module
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int, PRNGKeyArray

from ..likelihood_optimization.base_optimizer import AbstractEnsembleParameterOptimizer
from ..prior_projection.base_prior_projector import AbstractPriorProjector


class AbstractEnsembleRefinementPipeline(Module):
    """
    Abstract class for ensemble refinement pipelines.
    """

    prior_projector: AbstractVar[AbstractPriorProjector]
    optimizer: AbstractVar[AbstractEnsembleParameterOptimizer]
    n_steps: AbstractVar[Int]

    @abstractmethod
    def __call__(
        self,
        key: PRNGKeyArray,
        initial_walkers: Float[Array, "n_walkers n_atoms 3"],
        initial_weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        args_for_optimizer: Any,
        args_for_prior_projector: Any,
    ) -> Tuple[
        Float[Array, "n_steps n_walkers n_atoms 3"],
        Float[Array, "n_steps n_walkers"],
    ]:
        raise NotImplementedError
