import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Bool, Float, Int


class EarlyStoppingState(eqx.Module):
    best_loss: Float
    best_loss_step: Int
    current_step: Int


class EarlyStopping(eqx.Module):
    patience: Int
    rtol: Float
    atol: Float

    def __init__(self, patience: Int, rtol: Float, atol: Float):
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if rtol < 0:
            raise ValueError("rtol must be non-negative.")
        if atol < 0:
            raise ValueError("atol must be non-negative.")

        self.patience = patience
        self.rtol = rtol
        self.atol = atol

    def init(self) -> EarlyStoppingState:
        """Create a fresh early-stopping state."""
        return EarlyStoppingState(
            best_loss=jnp.inf,
            best_loss_step=0,
            current_step=0,
        )

    def update(self, state: EarlyStoppingState, loss: Float) -> EarlyStoppingState:
        """
        Return an updated state from the previous state and latest loss.
        """
        step = state.current_step + 1
        has_improved = (
            loss < (state.best_loss - self.atol - self.rtol * state.best_loss)
            if state.best_loss != jnp.inf
            else True
        )

        return EarlyStoppingState(
            best_loss=jnp.where(has_improved, loss, state.best_loss),
            best_loss_step=jnp.where(has_improved, step, state.best_loss_step),
            current_step=step,
        )

    def should_stop(self, state: EarlyStoppingState) -> Bool:
        """
        Return True when no sufficient improvement has happened for `patience` steps.
        """
        return (state.current_step - state.best_loss_step) >= self.patience

    def step(
        self, state: EarlyStoppingState, loss: Float
    ) -> tuple[EarlyStoppingState, Bool]:
        """
        Convenience wrapper: update state and return stop flag.
        """
        new_state = self.update(state, loss)
        return new_state, self.should_stop(new_state)
