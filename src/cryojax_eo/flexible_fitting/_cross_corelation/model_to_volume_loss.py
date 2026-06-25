import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class ModelToVolumeCorrelationLossFn(eqx.Module):
    variances: Float[Array, "n_atoms n_gaussians_per_atom"]
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"]
    render_fn: cxs.GaussianMixtureRenderFn
    vol_mask: Float[Array, "dim_z dim_y dim_x"]

    def __init__(
        self,
        amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_atoms n_gaussians_per_atom"],
        voxel_size: Float,
        volume_shape: tuple[int, int, int],
        vol_mask: Float[Array, "dim_z dim_y dim_x"] | None = None,
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

        self.vol_mask = jnp.ones(volume_shape) if vol_mask is None else vol_mask

        self.render_fn = cxs.GaussianMixtureRenderFn(
            shape=volume_shape,
            voxel_size=voxel_size,
            batch_options=dict(
                batch_size=batch_size_for_z_planes, n_batches=n_batches_of_atoms
            ),
        )

    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "dim dim dim"],
    ) -> float:
        # Compute the model-to-volume loss
        return 1 - _model_to_volume_crosscorrelation(
            walker,
            self.amplitudes,
            self.variances,
            reference_volume,
            self.render_fn,
            self.vol_mask,
        )


def _model_to_volume_crosscorrelation(
    walker: Float[Array, "n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    reference_volume: Float[Array, "dim dim dim"],
    render_fn: cxs.GaussianMixtureRenderFn,
    vol_mask: Float[Array, "dim dim dim"],
) -> float:
    comp_volume = render_fn(
        volume_representation=cxs.GaussianMixtureVolume(
            walker,
            amplitudes,
            variances,
        ),
    )

    comp_volume = comp_volume * vol_mask
    reference_volume = reference_volume * vol_mask

    cross_correlation = (
        jnp.sum(comp_volume * reference_volume)
        / jnp.linalg.norm(comp_volume)
        / jnp.linalg.norm(reference_volume)
    )

    return cross_correlation
