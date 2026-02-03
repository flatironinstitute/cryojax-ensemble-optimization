from collections.abc import Callable
from typing import Any, TypeAlias, TypedDict, TypeVar

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
        Any | None,
        bool | None,
        ConstantT,
        PerParticleT,
    ],
    Float,
]
