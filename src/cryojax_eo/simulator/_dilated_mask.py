import cryojax.simulator as cxs
import jax.numpy as jnp
from equinox import Module
from jaxtyping import Array, Float


class DilatedMask(Module):
    volume: cxs.FourierVoxelGridVolume
    image_config: cxs.AbstractImageConfig

    def __init__(
        self,
        real_voxel_grid: Float[Array, "z y x"],
        image_config: cxs.AbstractImageConfig,
    ):
        self.image_config = image_config
        self.volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(real_voxel_grid)

    def project(self, pose: cxs.AbstractPose):
        mask2d = cxs.make_image_model(
            self.volume, self.image_config, pose.to_inverse_rotation()
        ).simulate()
        mask2d /= mask2d.max()
        mask2d = jnp.where(jnp.abs(mask2d) > 0.2, 1.0, 0.0)
        return mask2d
