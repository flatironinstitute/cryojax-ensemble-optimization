"""
Weight and position optimizers for ensemble refinement.
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

from .model_to_volume_loss import AbstractModelToVolumeLossFn


class AbstractWalkerOptimizer(eqx.Module):
    step_size: Float
    n_steps: Int = eqx.field(static=True)
    model_to_vol_loss_fn: AbstractModelToVolumeLossFn

    @eqx.filter_jit
    def __call__(
        self,
        walkers: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
        opt_state: optax.OptState,
    ) -> tuple[Float, Float[Array, "n_atoms 3"], optax.OptState]:
        def step_fn(carry, _):
            walkers, opt_state = carry
            loss, gradients = _compute_walker_gradients(
                walkers, reference_volume, self.model_to_vol_loss_fn
            )
            walkers, opt_state = self._optimizer_step(walkers, gradients, opt_state)
            return (walkers, opt_state), loss

        (walkers, opt_state), losses = jax.lax.scan(
            step_fn, (walkers, opt_state), xs=None, length=self.n_steps
        )
        return losses[-1], walkers, opt_state

    @abc.abstractmethod
    def _initalize_opt_state(self, walkers: Float[Array, "n_atoms 3"]) -> optax.OptState:
        raise NotImplementedError

    @abc.abstractmethod
    def _optimizer_step(
        self,
        walkers: Float[Array, "n_atoms 3"],
        gradients: Float[Array, "n_atoms 3"],
        opt_state: optax.OptState,
    ) -> tuple[Float[Array, "n_atoms 3"], optax.OptState]:
        raise NotImplementedError


class SteepestDescWalkerFlexibleFitting(AbstractWalkerOptimizer):
    def __init__(
        self,
        n_steps: Int,
        step_size: Float,
        model_to_vol_loss_fn: AbstractModelToVolumeLossFn,
    ):
        assert n_steps > 0, "n_steps must be positive"
        assert step_size >= 0, "step_size must be non-negative."

        self.n_steps = n_steps
        self.model_to_vol_loss_fn = model_to_vol_loss_fn
        self.step_size = step_size

    def _optimizer_step(
        self,
        walkers: Float[Array, "n_atoms 3"],
        gradients: Float[Array, "n_atoms 3"],
        opt_state: optax.OptState,
    ) -> tuple[Float[Array, "n_atoms 3"], optax.OptState]:
        norms = jnp.linalg.norm(gradients, axis=(1), keepdims=True)
        # set small norms to 1 (avoid making small gradients large!)
        norms = jnp.where(norms < 1e-12, 1.0, norms)
        gradients = gradients / norms

        new_walkers = walkers - self.step_size * gradients
        return new_walkers, opt_state

    def _initalize_opt_state(self, walkers: Float[Array, "n_atoms 3"]) -> optax.OptState:
        return None


class AdamWalkerFlexibleFitting(AbstractWalkerOptimizer):
    optimizer: optax.GradientTransformationExtraArgs

    def __init__(
        self,
        n_steps: Int,
        step_size: Float,
        model_to_vol_loss_fn: AbstractModelToVolumeLossFn,
    ):
        assert n_steps > 0, "n_steps must be positive"
        assert step_size >= 0, "step_size must be non-negative."

        self.n_steps = n_steps
        self.model_to_vol_loss_fn = model_to_vol_loss_fn
        self.step_size = step_size
        self.optimizer = optax.adam(learning_rate=self.step_size)

    def _optimizer_step(
        self,
        walkers: Float[Array, "n_atoms 3"],
        gradients: Float[Array, "n_atoms 3"],
        opt_state: optax.OptState,
    ) -> tuple[Float[Array, "n_atoms 3"], optax.OptState]:
        updates, opt_state = self.optimizer.update(gradients, opt_state)
        new_walkers = optax.apply_updates(walkers, updates)
        return new_walkers, opt_state

    def _initalize_opt_state(self, walkers: Float[Array, "n_atoms 3"]) -> optax.OptState:
        return self.optimizer.init(walkers)


def _compute_walker_gradients(
    walkers: Float[Array, "n_atoms 3"],
    reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
    model_to_vol_loss_fn: AbstractModelToVolumeLossFn,
) -> tuple[Float, Float[Array, "n_atoms 3"]]:
    def _loss_fn(walker, ref_volume):
        return model_to_vol_loss_fn(walker, ref_volume)

    loss, gradients = jax.value_and_grad(
        _loss_fn,
        argnums=0,
    )(
        walkers,
        reference_volume,
    )

    return loss, gradients
