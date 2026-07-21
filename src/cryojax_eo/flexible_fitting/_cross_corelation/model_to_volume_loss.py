import abc

import cryojax.ndimage as im
import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class AbstractModelToVolumeLossFn(eqx.Module):
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
            n_batches=n_batches_of_atoms,
        )

    @abc.abstractmethod
    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "dim dim dim"],
    ) -> float:
        raise NotImplementedError


class ModelToVolumeCorrelationLossFn(AbstractModelToVolumeLossFn):
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


class ModelToVolumeWeightedMSELossFn(AbstractModelToVolumeLossFn):
    mse_weights: Float[Array, "dim dim dim//2+1"]

    def __init__(
        self,
        amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "dim dim dim"],
        voxel_size: Float,
        volume_shape: tuple[int, int, int],
        vol_mask: Float[Array, "dim_z dim_y dim_x"] | None = None,
        *,
        batch_size_for_z_planes: Int = 1,
        n_batches_of_atoms: Int = 1,
    ):
        super().__init__(
            amplitudes,
            variances,
            voxel_size,
            volume_shape,
            vol_mask=vol_mask,
            batch_size_for_z_planes=batch_size_for_z_planes,
            n_batches_of_atoms=n_batches_of_atoms,
        )

        weights = weights
        upsampling_factor = weights.shape[0] / volume_shape[0]
        if not float(upsampling_factor).is_integer():
            raise ValueError(
                f"weights shape {weights.shape[0]} is not an integer multiple "
                f"of volume_shape {volume_shape[0]}"
            )
        upsampling_factor = int(upsampling_factor)

        upsampled_shape = tuple(v * upsampling_factor for v in volume_shape)
        rfftn_weights = _rfftn_weights(upsampled_shape, axes=(2,))
        # TODO: Ideally reference volume should be passed to constructor
        # so we can pre-compute and cache its upsampled RFFT
        self.mse_weights = (rfftn_weights * weights) ** 0.5

    def __call__(
        self,
        walker: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "dim dim dim"],
    ) -> float:
        return _model_to_volume_weighted_mse(
            walker,
            self.amplitudes,
            self.variances,
            reference_volume,
            self.render_fn,
            self.mse_weights,
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


def _model_to_volume_weighted_mse(
    walker: Float[Array, "n_atoms 3"],
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    reference_volume: Float[Array, "dim dim dim"],
    render_fn: cxs.GaussianMixtureRenderFn,
    fourier_weights: Float[Array, "dim dim dim"],
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

    # fourier_weights are already in upsampled RFFT space
    # pad the volumes and compute the RFFT (of the same shape)
    pad_size = (fourier_weights.shape[0],) * 3
    comp_volume_fourier = (
        im.rfftn(im.pad_to_shape(comp_volume, pad_size)) * fourier_weights
    )
    reference_volume_fourier = (
        im.rfftn(im.pad_to_shape(reference_volume, pad_size)) * fourier_weights
    )

    optimal_scale = jnp.sum(comp_volume_fourier * reference_volume_fourier) / jnp.sum(
        comp_volume_fourier**2
    )

    return (
        jnp.linalg.norm(optimal_scale * comp_volume_fourier - reference_volume_fourier)
        ** 2
    )


def _rfftn_weights(shape, axes=None):
    """
    Construct weights for computing inner products in RFFT space.

    Args:
        shape: tuple, original real-space shape
        axes: tuple or None, axes used in rfftn (same convention as jnp.fft.rfftn)

    Returns:
        weights: array with shape equal to rfftn output
    """
    if axes is None:
        axes = tuple(range(len(shape)))
    axes = tuple(axes)

    # Output shape after rfftn
    out_shape = list(shape)
    last_axis = axes[-1]
    out_shape[last_axis] = shape[last_axis] // 2 + 1

    weights = jnp.ones(out_shape)

    for ax in axes:
        n = shape[ax]

        if ax == last_axis:
            # rfft axis: truncated
            freq_size = n // 2 + 1
            w = jnp.ones(freq_size)

            if n % 2 == 0:
                # even: Nyquist exists
                w = w.at[1:-1].set(2.0)
            else:
                # odd: no Nyquist
                w = w.at[1:].set(2.0)
        else:
            # full fft axis
            freq_size = n
            w = jnp.ones(freq_size)

            if n % 2 == 0:
                w = w.at[1 : n // 2].set(2.0)
                w = w.at[n // 2 + 1 :].set(2.0)
            else:
                w = w.at[1 : (n + 1) // 2].set(2.0)
                w = w.at[(n + 1) // 2 :].set(2.0)

        # reshape for broadcasting
        reshape_dims = [1] * len(out_shape)
        reshape_dims[ax] = freq_size
        w = w.reshape(reshape_dims)

        weights = weights * w

    return weights
