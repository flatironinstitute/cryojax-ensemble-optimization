import cryojax.simulator as cxs
import jax.numpy as jnp
from jaxtyping import Array, Float


def model_to_volume_crosscorrelation(
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

    return 1 - cross_correlation
