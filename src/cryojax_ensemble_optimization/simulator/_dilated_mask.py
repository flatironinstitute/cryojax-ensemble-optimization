import cryojax.simulator as cxs
import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Float


class DilatedMask(Module):
    potential: cxs.FourierVoxelGridPotential
    config: cxs.AbstractConfig

    def __init__(self, density: Float[Array, "z y x"], config: cxs.AbstractConfig):
        self.config = config
        self.potential = cxs.FourierVoxelGridPotential.from_real_voxel_grid(
            density, config.pixel_size
        )

    def project(self, pose: cxs.AbstractPose):
        mask2d = cxs.make_image_model(self.potential, self.config, pose).simulate()
        mask2d /= mask2d.max()
        mask2d = jnp.where(jnp.abs(mask2d) > 0.1, 1.0, 0.0)
        return mask2d
