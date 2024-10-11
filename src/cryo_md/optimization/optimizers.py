import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import jax_dataloader as jdl

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex

from cryojax import get_filter_spec
from cryojax.image.operators import FourierGaussian
from cryojax.data import RelionDataset, RelionParticleStack


from .loss_and_gradients import (
    compute_loss_and_grads_positions,
    compute_lklhood_matrix,
)


@eqx.filter_jit
def compute_loss(weights, lklhood_matrix):
    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=weights[None, :], axis=1
    )
    return -jnp.mean(log_lklhood)


class CustomJaxDataset(jdl.Dataset):
    def __init__(self, cryojax_dataset: RelionDataset):
        self.cryojax_dataset = cryojax_dataset

    def __getitem__(self, index) -> RelionParticleStack:
        return self.cryojax_dataset[index]

    def __len__(self) -> int:
        return len(self.cryojax_dataset)


class WeightOptimizer:
    def __init__(self, rng_key, n_steps, step_size, batch_size, dataset):
        assert n_steps >= 0, "Number of steps must be greater than 0"
        assert step_size > 0, "Step size must be greater than 0"

        self.n_steps = n_steps
        self.step_size = step_size
        self.filter_spec_for_vmap = _get_particle_stack_filter_spec(dataset[0:2])
        self.batch_size = batch_size
        self.dataset_size = len(dataset)

        self.dataloader = jdl.DataLoader(
            CustomJaxDataset(
                dataset
            ),  # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
            backend="jax",  # Use 'jax' backend for loading data
            batch_size=self.batch_size,  # Batch size
            shuffle=True,  # Shuffle the dataloader every iteration or not
            drop_last=False,  # Drop the last batch or not
        )

        self.key = rng_key
        return

    def run(self, atom_positions, weights, structural_info, noise_variance):
        logging.info(f"Running weight optimization for {self.n_steps} steps...")

        if self.n_steps == 0:
            logging.info("No optimization steps requested. Returning initial weights.")

        else:
            for batch in self.dataloader:
                relion_stack_vmap, relion_stack_novmap = eqx.partition(
                    batch, self.filter_spec_for_vmap
                )
                lklhood_matrix = compute_lklhood_matrix(
                    atom_positions,
                    relion_stack_vmap,
                    (
                        structural_info["atom_identities"],
                        structural_info["b_factors"],
                        noise_variance,
                        relion_stack_novmap,
                    ),
                )
                break

            pg = ProjectedGradient(fun=compute_loss, projection=projection_simplex)
            weights = pg.run(weights, lklhood_matrix=lklhood_matrix).params
            loss = compute_loss(weights, lklhood_matrix)

            logging.info(
                f"Optimization done. Final loss: {loss}. Final weights: {weights}."
            )

        return weights


class PositionOptimizer:
    def __init__(self, rng_key, step_size, batch_size, dataset, n_steps=1):
        assert step_size > 0, "Step size must be greater than 0"
        assert n_steps > 0, "Number of steps must be greater than 0"

        self.n_steps = n_steps
        self.step_size = step_size
        self.filter_spec_for_vmap = _get_particle_stack_filter_spec(dataset[0:2])
        self.batch_size = batch_size

        self.dataset_size = len(dataset)
        self.dataloader = jdl.DataLoader(
            CustomJaxDataset(
                dataset
            ),  # Can be a jdl.Dataset or pytorch or huggingface or tensorflow dataset
            backend="jax",  # Use 'jax' backend for loading data
            batch_size=self.batch_size,  # Batch size
            shuffle=True,  # Shuffle the dataloader every iteration or not
            drop_last=False,  # Drop the last batch or not
        )

        self.key = rng_key

        return

    def run(self, positions, weights, structural_info, noise_variance):
        # traj = np.zeros((self.n_steps, *positions.shape))
        # get batch_size n random indices from self.dataset_size
        logging.info("Running position optimization...")
        loss = None
        for i in range(self.n_steps):
            batch = next(iter(self.dataloader))
            relion_stack_vmap, relion_stack_novmap = eqx.partition(
                batch, self.filter_spec_for_vmap
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

            # logging.info(grads.shape)
            # logging.info(jnp.linalg.norm(positions, axis=(1, 2)))
            # logging.info(jnp.linalg.norm(grads, axis=(1, 2)))

            norms = jnp.max(jnp.abs(grads), axis=(2), keepdims=True)
            grads /= norms  # jnp.maximum(norms, jnp.ones_like(norms))
            # grads /= jnp.linalg.norm(grads, axis=(1, 2), keepdims=True)

            # logging.info(jnp.linalg.norm(grads, axis=(1, 2)))

            positions = positions - self.step_size * grads
            # traj[i] = positions

            logging.info(f"i={i}, loss={loss}")
            # print(f"i={i}, loss={loss}")

        logging.info(f"Optimization done. Final loss: {loss}.")

        return positions, loss  # , traj


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
