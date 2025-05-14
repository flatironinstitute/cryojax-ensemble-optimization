from abc import abstractmethod
from typing import Any, Tuple

import equinox as eqx
from jaxtyping import Array, Float


class AbstractPriorProjector(eqx.Module, strict=True):
    """
    Abstract class for prior projectors.
    """

    @abstractmethod
    def __call__(
        self,
        ref_positions: Float[Array, "n_atoms 3"] | Float[Array, "n_walkers n_atoms 3"],
    ) -> Float[Array, "n_atoms 3"] | Float[Array, "n_walkers n_atoms 3"]:
        raise NotImplementedError("Abstract method not implemented.")
