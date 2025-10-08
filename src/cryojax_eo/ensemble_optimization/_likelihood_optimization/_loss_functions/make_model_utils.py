import cryojax.simulator as cxs
from cryojax.dataset import ParticleStackInfo
from jaxtyping import Array, Float

from ..._pose_search import global_SO3_hier_search


def _make_image_model_with_stack_poses(
    volume: cxs.AbstractVolumeRepresentation,
    relion_stack: ParticleStackInfo,
):
    """
    Create an image model using the poses from the Relion stack.
    This is a helper function to avoid passing poses explicitly.
    """
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    return cxs.make_image_model(
        volume,
        relion_stack["parameters"]["image_config"],
        relion_stack["parameters"]["pose"],
        relion_stack["parameters"]["transfer_theory"],
        normalizes_signal=True,
        simulates_quantity=False,
    )


def _make_image_model_from_gmm_estimate_poses(
    gmm_volume: cxs.GaussianMixtureVolume,
    relion_stack: ParticleStackInfo,
) -> cxs.AbstractImageModel:
    if relion_stack["parameters"] is None:
        raise ValueError("relion_stack must have non None 'parameters' field.")

    pose = global_SO3_hier_search(gmm_volume, relion_stack, 1, 5, 40)
    return cxs.make_image_model(
        gmm_volume,
        relion_stack["parameters"]["image_config"],
        pose,
        relion_stack["parameters"]["transfer_theory"],
        normalizes_signal=True,
        simulates_quantity=False,
    )


def make_image_model_from_gmm(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    estimates_pose: bool = False,
) -> cxs.AbstractImageModel:
    volume = cxs.GaussianMixtureVolume(
        walker,
        amplitudes,
        variances,
    )
    if estimates_pose:
        return _make_image_model_from_gmm_estimate_poses(volume, relion_stack)
    else:
        return _make_image_model_with_stack_poses(volume, relion_stack)
