"""
Base optimizer object for ensemble refinement.
"""

from abc import abstractmethod

from equinox import Module


class AbstractEnsembleParameterOptimizer(Module, strict=True):
    """Abstract interface for objects that optimize parameters
    of an ensemble of structures.
    """

    @abstractmethod
    def __call__(self, walkers, weights, dataloader, args):
        raise NotImplementedError
