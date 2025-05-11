from abc import abstractmethod
from typing import overload

import equinox as eqx
from jaxtyping import Array, Float


class AbstractPriorProjector(eqx.Module, strict=True):
    """
    Abstract class for prior projectors.
    """

    @overload
    @abstractmethod
    def __call__(self) -> Float[Array, "n_walkers n_atoms 3"]: ...

    @abstractmethod
    def __call__(self) -> Float[Array, "n_atoms 3"]:
        """
        Project the walkers onto the prior.

        Args:
            key: Random key.
            walkers: Walkers to be projected.
            weights: Weights of the walkers.
            dataloader: DataLoader object.
            args: Additional arguments.

        Returns:
            Projected walkers and updated weights.
        """
        raise NotImplementedError("Abstract method not implemented.")
