"""
Weight and position optimizers for ensemble refinement.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .model_to_volume_loss import ModelToVolumeLikelihoodFn


class SteepestDescWalkerFlexibleFitting(eqx.Module):
    step_size: Float
    n_steps: Int
    model_to_vol_likelihood_fn: ModelToVolumeLikelihoodFn

    def __init__(
        self,
        n_steps: Int,
        step_size: Float,
        model_to_vol_likelihood_fn: ModelToVolumeLikelihoodFn,
    ):
        assert n_steps > 0, "n_steps must be positive"
        assert step_size >= 0, "step_size must be non-negative."

        self.n_steps = n_steps
        self.model_to_vol_likelihood_fn = model_to_vol_likelihood_fn
        self.step_size = step_size

    def __call__(
        self,
        walkers,
        reference_volume,
    ) -> Tuple[Float, Float[Array, "n_atoms 3"]]:
        loss = jnp.inf
        for _ in range(self.n_steps):
            loss, walkers = _optimize_walkers_positions(
                walkers,
                reference_volume,
                self.step_size,
                self.model_to_vol_likelihood_fn,
            )

        return loss, walkers


@eqx.filter_jit
def _optimize_walkers_positions(
    walkers: Float[Array, "n_atoms 3"],
    reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
    step_size: Float,
    likelihood_fn: ModelToVolumeLikelihoodFn,
) -> Tuple[Float, Float[Array, "n_atoms 3"]]:
    def _lklhood_fn(walker, ref_volume):
        return likelihood_fn(walker, ref_volume)

    loss, gradients = jax.value_and_grad(
        _lklhood_fn,
        argnums=0,
    )(
        walkers,
        reference_volume,
    )

    norms = jnp.linalg.norm(gradients, axis=(1), keepdims=True)
    # set small norms to 1 (avoid making small gradients large!)
    norms = jnp.where(norms < 1e-12, 1.0, norms)
    gradients /= norms

    return loss, walkers - step_size * gradients
