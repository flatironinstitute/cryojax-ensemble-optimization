import cryojax.simulator as cxs
import equinox as eqx
from cryojax.jax_util import filter_bmap
from jaxtyping import Array, Float

from .._pose_search import HierarchicalSO3GridSearch


@eqx.filter_jit
def estimate_poses(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians"],
    variances: Float[Array, "n_atoms n_gaussians"],
    images: Float[Array, "n_images y x"],
    image_config: cxs.BasicImageConfig,
    transfer_theories: cxs.ContrastTransferTheory,
    pose_search: HierarchicalSO3GridSearch,
    integrator: cxs.AbstractVolumeIntegrator,
    n_walkers_in_parallel: int,
    n_images_in_parallel: int,
) -> cxs.QuaternionPose:
    """
    Estimate the best poses for each walker and image using the pose search.

    **Arguments:**
    - walkers:
        The current positions of the walkers as Gaussian mixture volumes.
    - amplitudes:
        The amplitudes of the Gaussian mixtures.
    - variances:
        The variances of the Gaussian mixtures.
    - images:
        The images to estimate the poses for.
    - image_config:
        The configuration of the images.
    - transfer_theories:
        The transfer theories for the images.
    - pose_search:
        The pose search object.

    **Returns:**
        The estimated poses for each walker and image.
    """
    return filter_bmap(
        lambda x: _estimate_poses_for_walker(
            x,
            amplitudes,
            variances,
            images,
            image_config,
            transfer_theories,
            pose_search,
            integrator,
            n_images_in_parallel,
        ),
        xs=walkers,
        batch_size=n_walkers_in_parallel,
    )


@eqx.filter_vmap(in_axes=(0, None, None, None, None, None, None, None, None))
def _estimate_poses_for_walker(
    walker: Float[Array, "n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians"],
    variances: Float[Array, "n_atoms n_gaussians"],
    images: Float[Array, "n_images y x"],
    image_config: cxs.BasicImageConfig,
    transfer_theories: cxs.ContrastTransferTheory,
    pose_search: HierarchicalSO3GridSearch,
    integrator: cxs.AbstractVolumeIntegrator,
    n_images_in_parallel: int,
) -> cxs.QuaternionPose:
    volume = cxs.GaussianMixtureVolume(walker, amplitudes, variances)

    return filter_bmap(
        lambda x: _estimate_poses_for_single_image(
            volume,
            x[0],
            image_config,
            x[1],
            pose_search,
            integrator,
        ),
        xs=(images, transfer_theories),
        batch_size=n_images_in_parallel,
    )


@eqx.filter_vmap(in_axes=(None, 0, None, 0, None, None))
def _estimate_poses_for_single_image(
    volume: cxs.GaussianMixtureVolume,
    image: Float[Array, "y x"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    pose_search: HierarchicalSO3GridSearch,
    integrator: cxs.AbstractVolumeIntegrator,
) -> cxs.QuaternionPose:
    return pose_search(volume, image, image_config, transfer_theory, integrator)
