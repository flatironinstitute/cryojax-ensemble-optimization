import logging

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from cryojax import get_filter_spec
from cryojax.image.operators import FourierGaussian

from .loss_and_gradients import (
    compute_loss_and_grads_positions,
    compute_loss_and_grads_weights,
)


class WeightOptimizer:
    def __init__(self, rng_key, n_steps, step_size, batch_size, dataset):
        assert n_steps >= 0, "Number of steps must be greater than 0"
        assert step_size > 0, "Step size must be greater than 0"

        self.n_steps = n_steps
        self.step_size = step_size
        self.filter_spec_for_vmap = _get_particle_stack_filter_spec(dataset[0:2])
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_size = len(dataset)

        self.key = rng_key
        return

    def run(self, atom_positions, weights, structural_info, noise_variance):
        logging.info(f"Running weight optimization for {self.n_steps} steps...")

        if self.n_steps == 0:
            logging.info("No optimization steps requested. Returning initial weights.")
            return weights

        else:
            loss = None
            for _ in range(self.n_steps):
                self.key, subkey = jax.random.split(self.key)

                subset_idx = jax.random.choice(
                    subkey, self.dataset_size, (self.batch_size,), replace=False
                )
                relion_stack = self.dataset[np.asarray(subset_idx)]

                relion_stack_vmap, relion_stack_novmap = eqx.partition(
                    relion_stack, self.filter_spec_for_vmap
                )
                loss, grad_wts = compute_loss_and_grads_weights(
                    atom_positions=atom_positions,
                    model_weights=weights,
                    relion_stack_vmap=relion_stack_vmap,
                    args=(
                        structural_info["atom_identities"],
                        structural_info["b_factors"],
                        noise_variance,
                        relion_stack_novmap,
                    ),
                )

                # weights = jax.nn.sigmoid(weights - self.step_size * grad_wts)
                weights = weights - self.step_size * grad_wts
                weights = jnp.maximum(weights, jnp.zeros_like(weights))
                weights /= jnp.sum(weights)

                logging.debug(f"Weights: {weights}; Loss: {loss}")

            logging.info(f"Optimization done. Final loss: {loss}.")

        return weights


class PositionOptimizer:
    def __init__(self, rng_key, step_size, batch_size, dataset, n_steps=1):
        assert step_size > 0, "Step size must be greater than 0"
        assert n_steps > 0, "Number of steps must be greater than 0"

        self.n_steps = n_steps
        self.step_size = step_size
        self.filter_spec_for_vmap = _get_particle_stack_filter_spec(dataset[0:2])
        self.batch_size = batch_size
        self.dataset = dataset
        self.dataset_size = len(dataset)

        self.key = rng_key

        return

    def run(self, positions, weights, structural_info, noise_variance):
        # get batch_size n random indices from self.dataset_size
        logging.info("Running position optimization...")
        loss = None
        for i in range(self.n_steps):
            self.key, subkey = jax.random.split(self.key)
            subset_idx = jax.random.choice(
                subkey, self.dataset_size, (self.batch_size,), replace=False
            )
            relion_stack = self.dataset[np.asarray(subset_idx)]

            relion_stack_vmap, relion_stack_novmap = eqx.partition(
                relion_stack, self.filter_spec_for_vmap
            )

            loss, grads = compute_loss_and_grads_positions(
                atom_positions=positions,
                model_weights=weights,
                relion_stack_vmap=relion_stack_vmap,
                args=(
                    structural_info["atom_identities"],
                    structural_info["b_factors"],
                    noise_variance,
                    relion_stack_novmap,
                ),
            )

            # print(grads.shape)
            # print(jnp.linalg.norm(positions, axis=(1, 2)))
            # print(jnp.linalg.norm(grads, axis=(1, 2)))

            norms = jnp.max(jnp.abs(grads), axis=(2), keepdims=True)
            grads /= norms  # jnp.maximum(norms, jnp.ones_like(norms))
            # grads /= jnp.linalg.norm(grads, axis=(1, 2), keepdims=True)

            # print(jnp.linalg.norm(grads, axis=(1, 2)))

            positions = positions - self.step_size * grads

            print(f"i={i}, loss={loss}")

        logging.info(f"Optimization done. Final loss: {loss}.")

        return positions, loss


def _get_particle_stack_filter_spec(particle_stack):
    return get_filter_spec(particle_stack, _pointer_to_vmapped_parameters)


def _pointer_to_vmapped_parameters(particle_stack):
    if isinstance(particle_stack.transfer_theory.envelope, FourierGaussian):
        output = (
            particle_stack.transfer_theory.ctf.defocus_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_angle,
            particle_stack.transfer_theory.ctf.phase_shift,
            particle_stack.transfer_theory.envelope.b_factor,
            particle_stack.transfer_theory.envelope.amplitude,
            particle_stack.pose.offset_x_in_angstroms,
            particle_stack.pose.offset_y_in_angstroms,
            particle_stack.pose.view_phi,
            particle_stack.pose.view_theta,
            particle_stack.pose.view_psi,
            particle_stack.image_stack,
        )
    else:
        output = (
            particle_stack.transfer_theory.ctf.defocus_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_in_angstroms,
            particle_stack.transfer_theory.ctf.astigmatism_angle,
            particle_stack.transfer_theory.ctf.phase_shift,
            particle_stack.pose.offset_x_in_angstroms,
            particle_stack.pose.offset_y_in_angstroms,
            particle_stack.pose.view_phi,
            particle_stack.pose.view_theta,
            particle_stack.pose.view_psi,
            particle_stack.image_stack,
        )
    return output
