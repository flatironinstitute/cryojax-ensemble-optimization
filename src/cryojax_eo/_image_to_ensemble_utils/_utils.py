import cryojax.simulator as cxs
import equinox as eqx
import jax
from cryojax.jax_util import filter_bmap
from cryojax.rotations import SO3
from jaxtyping import Array, Float

from .._pose_search import HierarchicalSO3GridSearch
from ..utils import rigid_align_positions


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
        The pose search object. If its `mask_frame_positions` is set, each walker
        is first rigidly aligned to those positions (once, before any batching),
        the search runs on the aligned walkers, and the alignment is composed into
        the returned poses so that they apply to the original walkers.

    **Returns:**
        The estimated poses for each walker and image.
    """
    if pose_search.mask_frame_positions is not None:
        assert pose_search.mask_frame_positions.shape == walkers.shape[1:], (
            "The model aligned to the pose search mask must have the same atoms "
            f"as the walkers. Got {pose_search.mask_frame_positions.shape[0]} atoms "
            f"for the aligned model and {walkers.shape[1]} for the walkers."
        )
        walkers, rotation_matrices, displacements = jax.vmap(
            rigid_align_positions, in_axes=(0, None)
        )(walkers, pose_search.mask_frame_positions)

    poses = filter_bmap(
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

    if pose_search.mask_frame_positions is not None:
        poses = _compose_poses_with_alignment(poses, rotation_matrices, displacements)
    return poses


@eqx.filter_vmap(in_axes=(0, 0, 0))
@eqx.filter_vmap(in_axes=(0, None, None))
def _compose_poses_with_alignment(
    pose: cxs.QuaternionPose,
    rotation_matrix: Float[Array, "3 3"],
    displacement: Float[Array, "1 3"],
) -> cxs.QuaternionPose:
    """
    Given a pose `P'` estimated for an aligned walker `x' = x @ R.T + d`, return
    the pose `P` that produces the same image from the original walker `x`.

    Atom positions are rotated by `pose.rotation` (`A`), so
    `A' (R x + d) = (A' R) x + A' d`: the rotation becomes `A' R`, and the
    in-plane part of `A' d` is added to the offset.
    """
    rotation = pose.rotation.compose(SO3.from_matrix(rotation_matrix))
    offset = pose.offset_in_angstroms[:2] + pose.rotation.apply(displacement[0])[:2]
    return cxs.QuaternionPose.from_rotation_and_translation(rotation, offset)


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
