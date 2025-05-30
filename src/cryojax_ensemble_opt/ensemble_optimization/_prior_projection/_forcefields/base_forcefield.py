"""
Base class for force fields used in custom prior projection methods.
"""

from abc import abstractmethod
from typing import Callable, TypeVar

import equinox as eqx
from jaxtyping import Array, Float


EnergyFuncArgs = TypeVar("EnergyFuncArgs")


class AbstractForceField(eqx.Module, strict=True):
    """
    Abstract base class for a force field.
    """

    energy_fn: Callable[[Float[Array, "n_atoms 3"], EnergyFuncArgs], Float]
    energy_fn_args: EnergyFuncArgs

    @abstractmethod
    def __call__(
        self,
        positions: Float[Array, "n_atoms 3"],
    ):
        """
        Compute the energy of the system.
        """
        raise NotImplementedError
