import logging

import jax
import jax.numpy as jnp
import equinox as eqx
import jax_dataloader as jdl

from cryojax.data import RelionParticleParameterDataset, RelionParticleStackDataset
from cryojax.constants import get_tabulated_scattering_factor_parameters

from .loss_and_gradients import compute_loss_weights_and_grads


@eqx.filter_jit
def compute_loss(weights, lklhood_matrix):
    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=weights[None, :], axis=1
    )
    return -jnp.mean(log_lklhood)


class CustomJaxDataset(jdl.Dataset):
    def __init__(self, cryojax_dataset: RelionParticleParameterDataset):
        self.cryojax_dataset = cryojax_dataset

    def __getitem__(self, index) -> RelionParticleStackDataset:
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
        self.parameter_table = get_tabulated_scattering_factor_parameters(
            structural_info["atom_identities"]
        )
        self.noise_variance = noise_variance

        return

    def run(self, positions):
        # traj = np.zeros((self.n_steps, *positions.shape))
        # get batch_size n random indices from self.dataset_size
        logging.info("Running position optimization...")

        for i in range(self.n_steps):
            batch = next(iter(self.dataloader))

            outputs, grads = compute_loss_weights_and_grads(
                atom_positions=positions,
                weights=self.weights,
                relion_stack=batch,
                args=(
                    self.structural_info["atom_identities"],
                    self.structural_info["b_factors"],
                    self.parameter_table,
                    self.noise_variance,
                ),
            )

            self.loss, self.weights = outputs

            norms = jnp.linalg.norm(grads, axis=(2), keepdims=True)
            grads /= norms

            positions = positions - self.step_size * grads

            logging.info(f"i={i}, loss={self.loss}, weights={self.weights}")

        logging.info(f"Optimization done. Final loss: {self.loss}.")

        return positions, self.weights, self.loss


