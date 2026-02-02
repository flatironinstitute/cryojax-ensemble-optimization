from typing import Any, Callable, Optional, TypedDict, TypeVar
from typing_extensions import TypeAlias

import numpy as np
from cryojax.simulator import BasicImageConfig, ContrastTransferTheory, EulerAnglePose
from jaxtyping import Array, Float


class ParticleParameterInfo(TypedDict):
    """Parameters for a particle stack from RELION."""

    image_config: BasicImageConfig
    pose: EulerAnglePose
    transfer_theory: ContrastTransferTheory


class ParticleStackInfo(TypedDict):
    """Particle stack info from RELION."""

    parameters: ParticleParameterInfo | None
    images: Float[np.ndarray, "... y_dim x_dim"]


PerParticleT = TypeVar("PerParticleT")
ConstantT = TypeVar("ConstantT")

LossFn: TypeAlias = Callable[
    [
        Float[Array, "n_atoms 3"],
        ParticleStackInfo,
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Float[Array, "n_atoms n_gaussians_per_atom"],
        Optional[Any],
        Optional[bool],
        ConstantT,
        PerParticleT,
    ],
    Float,
]
