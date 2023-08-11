import jax.numpy as jnp
import logging

from cryo_md.likelihood.calc_lklhood import (
    calc_lkl_and_grad_wts,
    calc_lkl_and_grad_struct,
)


class WeightOptimizer:
    def __init__(self, n_steps, step_size):
        assert n_steps > 0, "Number of steps must be greater than 0"
        assert step_size > 0, "Step size must be greater than 0"

        self.n_steps = n_steps
        self.step_size = step_size

        return

    def run(self, positions, weights, image_stack, struct_info):
        logging.info(f"Running weight optimization for {self.n_steps} steps...")
        for _ in range(self.n_steps):
            loss, grad_wts = calc_lkl_and_grad_wts(
                positions, weights, image_stack, struct_info
            )

            weights = weights + self.step_size * grad_wts
            weights /= jnp.sum(weights)

        logging.info(f"Optimization done. Final loss: {loss}.")

        return weights


class PositionOptimizer:
    def __init__(self, step_size, batch_size):
        assert step_size > 0, "Step size must be greater than 0"
        assert batch_size > 0, "Batch size must be greater than 0"

        self.step_size = step_size
        self.batch_size = batch_size

        return

    def run(self, positions, weights, image_stack, struct_info):
        logging.info("Running position optimization...")
        loss, grad_str = calc_lkl_and_grad_struct(
            positions, weights, image_stack, struct_info, self.batch_size
        )

        norms = jnp.max(jnp.abs(grad_str), axis=(1))[:, None, :]
        grad_str /= norms #jnp.maximum(norms, jnp.ones_like(norms))

        positions = positions + self.step_size * grad_str
        logging.info(f"Optimization done. Final loss: {loss}.")

        return positions, loss
