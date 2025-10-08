from typing import Callable, Optional, TypeVar
from typing_extensions import TypeAlias

from cryojax.dataset import ParticleStackInfo
from jaxtyping import Array, Float

from .simulator._dilated_mask import DilatedMask


PerParticleT = TypeVar("PerParticleT")
ConstantT = TypeVar("ConstantT")

LossFn: TypeAlias = Callable[
    [
        Float[Array, "n_atoms 3"],
        ParticleStackInfo,
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Optional[DilatedMask],
        Optional[bool],
        ConstantT,
        PerParticleT,
    ],
    Float,
]
