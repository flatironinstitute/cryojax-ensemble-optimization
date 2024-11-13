import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import jax_dataloader as jdl


from cryojax import get_filter_spec
from cryojax.image.operators import FourierGaussian
from cryojax.data import RelionDataset, RelionParticleStack


from .loss_and_gradients import compute_loss_weights_and_grads


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


class EnsembleOptimizer:
    def __init__(
        self,
        step_size,
        batch_size,
        dataset,
        init_weights,
        structural_info,
        noise_variance,
        n_steps=1,
    ):
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

        self.weights = init_weights
        self.loss = None
        self.structural_info = structural_info
        self.noise_variance = noise_variance

        return

    def run(self, positions):
        # traj = np.zeros((self.n_steps, *positions.shape))
        # get batch_size n random indices from self.dataset_size
        logging.info("Running position optimization...")

        for i in range(self.n_steps):
            batch = next(iter(self.dataloader))
            relion_stack_vmap, relion_stack_novmap = eqx.partition(
                batch, self.filter_spec_for_vmap
            )

            outputs, grads = compute_loss_weights_and_grads(
                atom_positions=positions,
                weights=self.weights,
                relion_stack_vmap=relion_stack_vmap,
                args=(
                    self.structural_info["atom_identities"],
                    self.structural_info["b_factors"],
                    self.noise_variance,
                    relion_stack_novmap,
                ),
            )

            self.loss, self.weights = outputs

            norms = jnp.linalg.norm(grads, axis=(2), keepdims=True)
            grads /= norms

            positions = positions - self.step_size * grads

            logging.info(f"i={i}, loss={self.loss}, weights={self.weights}")

        logging.info(f"Optimization done. Final loss: {self.loss}.")

        return positions, self.weights, self.loss


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
