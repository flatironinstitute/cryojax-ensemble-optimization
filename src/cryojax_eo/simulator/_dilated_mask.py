import cryojax.simulator as cxs
import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Float


class DilatedMask(Module):
    volume: cxs.FourierVoxelGridVolume
    config: cxs.AbstractImageConfig

    def __init__(
        self, real_voxel_grid: Float[Array, "z y x"], config: cxs.AbstractImageConfig
    ):
        self.config = config
        self.volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)

    def project(self, pose: cxs.AbstractPose):
        mask2d = cxs.make_image_model(self.volume, self.config, pose).simulate()
        mask2d /= mask2d.max()
        mask2d = jnp.where(jnp.abs(mask2d) > 0.1, 1.0, 0.0)
        return mask2d
