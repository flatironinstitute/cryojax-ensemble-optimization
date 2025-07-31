"""
Weight and position optimizers for ensemble refinement.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.internal import error_if_negative, error_if_not_positive
from jaxtyping import Array, Float, Int

from .model_to_volume_loss import model_to_volume_crosscorrelation


class SteepestDescWalkerFlexibleFitting(eqx.Module):
    step_size: Float
    n_steps: Int
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"]
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"]
    voxel_size: Float

    def __init__(
        self,
        n_steps: Int,
        step_size: Float,
        gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
        gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
        voxel_size: Float,
    ):
        assert n_steps > 0, "n_steps must be positive"
        self.n_steps = n_steps
        self.gaussian_variances = error_if_not_positive(gaussian_variances)
        self.gaussian_amplitudes = error_if_not_positive(gaussian_amplitudes)

        self.voxel_size = error_if_not_positive(voxel_size)
        self.step_size = error_if_negative(step_size)

    def __call__(
        self,
        walkers,
        reference_volume,
        *,
        n_batches_of_atoms: int = 1,
        batch_size_for_z_planes: int = 1,
    ):
        for _ in range(self.n_steps):
            loss, walkers = _optimize_walkers_positions(
                walkers,
                reference_volume,
                self.step_size,
                self.gaussian_amplitudes,
                self.gaussian_variances,
                self.voxel_size,
                batch_size_for_z_planes=batch_size_for_z_planes,
                n_batches_of_atoms=n_batches_of_atoms,
            )

        return loss, walkers


# @eqx.filter_jit
def _optimize_walkers_positions(
    walkers: Float[Array, "n_atoms 3"],
    reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
    step_size: Float,
    gaussian_amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    gaussian_variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    voxel_size: Float,
    *,
    batch_size_for_z_planes: int = 1,
    n_batches_of_atoms: int = 1,
) -> Float[Array, "n_atoms 3"]:
    loss, gradients = jax.value_and_grad(
        model_to_volume_crosscorrelation,
        argnums=0,
    )(
        walkers,
        gaussian_amplitudes,
        gaussian_variances,
        reference_volume,
        voxel_size,
        batch_size_for_z_planes=batch_size_for_z_planes,
        n_batches_of_atoms=n_batches_of_atoms,
    )

    norms = jnp.linalg.norm(gradients, axis=(1), keepdims=True)
    # set small norms to 1 (avoid making small gradients large!)
    norms = jnp.where(norms < 1e-12, 1.0, norms)
    gradients /= norms

    return loss, walkers - step_size * gradients
