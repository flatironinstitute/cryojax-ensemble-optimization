from typing import TypedDict

import numpy as np
from cryojax.simulator import BasicImageConfig, ContrastTransferTheory, EulerAnglePose
from jaxtyping import Float


class ParticleParameterInfo(TypedDict):
    """Parameters for a particle stack from RELION."""

    image_config: BasicImageConfig
    pose: EulerAnglePose
    transfer_theory: ContrastTransferTheory


class ParticleStackInfo(TypedDict):
    """Particle stack info from RELION."""

    parameters: ParticleParameterInfo | None
    images: Float[np.ndarray, "... y_dim x_dim"]
