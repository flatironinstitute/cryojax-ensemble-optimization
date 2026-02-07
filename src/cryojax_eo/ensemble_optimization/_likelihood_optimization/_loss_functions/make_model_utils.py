import cryojax.simulator as cxs
from jaxtyping import Array, Float


def make_image_model_from_gmm(
    walker: Float[Array, "n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    image_config: cxs.BasicImageConfig,
    pose: cxs.AbstractPose,
    transfer_theory: cxs.ContrastTransferTheory,
) -> cxs.AbstractImageModel:
    volume = cxs.GaussianMixtureVolume(
        walker,
        amplitudes,
        variances,
    )
    return cxs.make_image_model(
        volume,
        image_config,
        pose,
        transfer_theory,
        normalizes_signal=True,
    )
