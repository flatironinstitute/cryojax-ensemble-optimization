from abc import abstractmethod
from typing import Any, List, Optional, Tuple

import equinox as eqx
from jaxtyping import Array, Float, PRNGKeyArray


class AbstractPriorProjector(eqx.Module, strict=True):
    """
    Abstract class for prior projectors.
    """

    @abstractmethod
    def initialize(
        self,
        init_state: Optional[Any] = None,
    ) -> Any:
        raise NotImplementedError(
            "Abstract method not implemented. "
            "Please implement the initialize method in the subclass."
        )

    @abstractmethod
    def __call__(
        self,
        key: PRNGKeyArray,
        ref_positions: Float[Array, "n_atoms 3"] | Float[Array, "n_walkers n_atoms 3"],
        state: Any,
    ) -> Tuple[Float[Array, "n_atoms 3"] | Float[Array, "n_walkers n_atoms 3"], Any]:
        raise NotImplementedError("Abstract method not implemented.")


class AbstractEnsemblePriorProjector(eqx.Module, strict=True):
    """
    Abstract class for ensemble prior projectors.
    """

    projectors: eqx.AbstractVar[List[AbstractPriorProjector]]

    def initialize(
        self,
        init_states: Optional[List[Any]] = None,
    ) -> List[Any]:
        if init_states is None:
            init_states = [None] * len(self.projectors)
        states = []
        for i, projector in enumerate(self.projectors):
            states.append(projector.initialize(init_states[i]))
        return states

    @abstractmethod
    def __call__(
        self,
        key: PRNGKeyArray,
        ref_positions: Float[Array, "n_walkers n_atoms 3"],
        states: List[Any],
    ) -> Tuple[Float[Array, "n_walkers n_atoms 3"], List[Any]]:
        raise NotImplementedError("Abstract method not implemented.")
