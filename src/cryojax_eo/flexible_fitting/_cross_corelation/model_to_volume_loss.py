import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class ModelToVolumeLikelihoodFn(eqx.Module):
    variances: Float[Array, "n_atoms n_gaussians_per_atom"]
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"]
    voxel_size: Float
    batch_size_for_z_planes: Int
    n_batches_of_atoms: Int

    def __init__(
        self,
        amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_atoms n_gaussians_per_atom"],
        voxel_size: Float,
        *,
        batch_size_for_z_planes: Int = 1,
        n_batches_of_atoms: Int = 1,
    ):
        assert (amplitudes > 0).all(), "Amplitudes must be positive."
        assert (variances > 0).all(), "Variances must be positive."
        assert voxel_size > 0, "Voxel size must be positive."
        assert n_batches_of_atoms > 0, "n_batches_of_atoms must be positive."
        assert batch_size_for_z_planes > 0, "batch_size_for_z_planes must be positive."

        self.variances = variances
        self.amplitudes = amplitudes

        self.voxel_size = voxel_size
        self.batch_size_for_z_planes = int(batch_size_for_z_planes)
        self.n_batches_of_atoms = int(n_batches_of_atoms)

    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "dim dim dim"],
    ) -> Float:
        # Compute the model-to-volume loss
        return 1 - _model_to_volume_crosscorrelation(
            walker,
            self.amplitudes,
            self.variances,
            reference_volume,
            self.voxel_size,
            batch_size_for_z_planes=self.batch_size_for_z_planes,
            n_batches_of_atoms=self.n_batches_of_atoms,
        )


def _model_to_volume_crosscorrelation(
    walker: Float[Array, "n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    reference_volume: Float[Array, "dim dim dim"],
    voxel_size: Float,
    *,
    batch_size_for_z_planes: int = 1,
    n_batches_of_atoms: int = 1,
) -> Float:
    volume_shape = reference_volume.shape

    comp_volume = cxs.GaussianMixtureVolume(
        walker,
        amplitudes,
        variances,
    ).to_real_voxel_grid(
        volume_shape,  # type: ignore
        voxel_size,
        batch_options={
            "batch_size": batch_size_for_z_planes,
            "n_batches": n_batches_of_atoms,
        },
    )

    cross_correlation = (
        jnp.sum(comp_volume * reference_volume)
        / jnp.linalg.norm(comp_volume)
        / jnp.linalg.norm(reference_volume)
    )

    return cross_correlation
