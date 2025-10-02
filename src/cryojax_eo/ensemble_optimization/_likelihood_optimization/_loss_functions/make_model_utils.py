import cryojax.simulator as cxs
from jaxtyping import Array, Float

from ...._custom_types import ParticleStackInfo
from ..._pose_search import global_SO3_hier_search


def _make_image_model_with_stack_poses(
    potential: cxs.AbstractAtomicPotential,
    relion_stack: ParticleStackInfo,
):
    """
    Create an image model using the poses from the Relion stack.
    This is a helper function to avoid passing poses explicitly.
    """
    return cxs.make_image_model(
        potential,
        relion_stack["parameters"]["config"],
        relion_stack["parameters"]["pose"],
        relion_stack["parameters"]["transfer_theory"],
        normalizes_signal=True,
        physical_units=False,
    )


def _make_image_model_from_gmm_estimate_poses(
    potential: cxs.GaussianMixtureAtomicPotential,
    relion_stack: ParticleStackInfo,
) -> cxs.AbstractImageModel:
    pose = global_SO3_hier_search(potential, relion_stack, 1, 5, 40)
    return cxs.make_image_model(
        potential,
        relion_stack["parameters"]["config"],
        pose,
        relion_stack["parameters"]["transfer_theory"],
        normalizes_signal=True,
        physical_units=False,
    )


def make_image_model_from_gmm(
    walker: Float[Array, "n_atoms 3"],
    relion_stack: ParticleStackInfo,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    estimates_pose: bool = False,
) -> cxs.AbstractImageModel:
    potential = cxs.GaussianMixtureAtomicPotential(
        walker,
        gaussian_amplitudes,
        gaussian_variances,
    )
    if estimates_pose:
        return _make_image_model_from_gmm_estimate_poses(potential, relion_stack)
    else:
        return _make_image_model_with_stack_poses(potential, relion_stack)
