import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
from cryojax.internal import error_if_not_positive
from jaxtyping import Array, Float, Int


class ModelToVolumeLikelihoodFn(eqx.Module):
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"]
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"]
    voxel_size: Float
    batch_size_for_z_planes: Int
    n_batches_of_atoms: Int

    def __init__(
        self,
        gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
        gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
        voxel_size: Float,
        *,
        batch_size_for_z_planes: Int = 1,
        n_batches_of_atoms: Int = 1,
    ):
        self.gaussian_variances = error_if_not_positive(gaussian_variances)
        self.gaussian_amplitudes = error_if_not_positive(gaussian_amplitudes)

        self.voxel_size = error_if_not_positive(voxel_size)
        self.batch_size_for_z_planes = int(error_if_not_positive(batch_size_for_z_planes))
        self.n_batches_of_atoms = int(error_if_not_positive(n_batches_of_atoms))

    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
    ) -> Float:
        # Compute the model-to-volume loss
        return 1 - _model_to_volume_crosscorrelation(
            walker,
            self.gaussian_amplitudes,
            self.gaussian_variances,
            reference_volume,
            self.voxel_size,
            batch_size_for_z_planes=self.batch_size_for_z_planes,
            n_batches_of_atoms=self.n_batches_of_atoms,
        )


def _model_to_volume_crosscorrelation(
    walker: Float[Array, "n_atoms 3"],
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
    voxel_size: Float,
    *,
    batch_size_for_z_planes: int = 1,
    n_batches_of_atoms: int = 1,
) -> Float:
    volume_shape = reference_volume.shape

    comp_volume = cxs.GaussianMixtureAtomicPotential(
        walker,
        gaussian_amplitudes,
        gaussian_variances,
    ).as_real_voxel_grid(
        volume_shape,
        voxel_size,
        batch_size_for_z_planes=batch_size_for_z_planes,
        n_batches_of_atoms=n_batches_of_atoms,
    )

    cross_correlation = (
        jnp.sum(comp_volume * reference_volume)
        / jnp.linalg.norm(comp_volume)
        / jnp.linalg.norm(reference_volume)
    )

    return cross_correlation
