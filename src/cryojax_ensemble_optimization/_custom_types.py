from typing import Callable, Optional, TypedDict, TypeVar
from typing_extensions import TypeAlias

import pandas as pd
from cryojax.simulator import BasicConfig, ContrastTransferTheory, EulerAnglePose
from jaxtyping import Array, Float

from .simulator._dilated_mask import DilatedMask


class ParticleParameterInfo(TypedDict):
    """Parameters for a particle stack from RELION."""

    config: BasicConfig
    pose: EulerAnglePose
    transfer_theory: ContrastTransferTheory

    metadata: Optional[pd.DataFrame]


class ParticleStackInfo(TypedDict):
    """Particle stack info from RELION."""

    parameters: ParticleParameterInfo
    images: Float[Array, "... y_dim x_dim"]


PerParticleT = TypeVar("PerParticleT")
ConstantT = TypeVar("ConstantT")

LossFn: TypeAlias = Callable[
    [
        Float[Array, "n_atoms 3"],
        ParticleStackInfo,
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Optional[DilatedMask],
        ConstantT,
        PerParticleT,
    ],
    Float,
]
