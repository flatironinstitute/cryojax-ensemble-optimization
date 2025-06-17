from typing import Dict, Optional

import jax_dataloader as jdl
from cryojax.data import (
    AbstractParticleStackDataset,
    RelionParticleStackDataset,
)
from jaxtyping import PRNGKeyArray

from .._custom_types import PerParticleArgs


class CustomJaxDataset(jdl.Dataset):
    cryojax_dataset: RelionParticleStackDataset
    per_particle_args: Optional[PerParticleArgs]

    def __init__(
        self,
        cryojax_dataset: RelionParticleStackDataset,
        per_particle_args: Optional[PerParticleArgs] = None,
    ):
        self.cryojax_dataset = cryojax_dataset
        self.per_particle_args = per_particle_args

    def __getitem__(self, index) -> Dict[str, AbstractParticleStackDataset | PerParticleArgs]:
        if self.per_particle_args is None:
            per_particle_args = None
        else:
            per_particle_args = self.per_particle_args[index]

        data = {
            "particle_stack": self.cryojax_dataset[index],
            "per_particle_args": per_particle_args,
        }
        return data

    def __len__(self) -> int:
        return len(self.cryojax_dataset)


def create_dataloader(
    relion_stack_dataset: RelionParticleStackDataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    *,
    per_particle_args: Optional[PerParticleArgs] = None,
    jax_prng_key: Optional[PRNGKeyArray] = None,
):
    """
    Create a Jax DataLoader for a RelionParticleStackDataset.
    **Arguments:**
        relion_stack_dataset: A RelionParticleStackDataset object.
        batch_size: The size of each batch.
        shuffle: Whether to shuffle the dataset.
        drop_last: Whether to drop the last batch if it is smaller than batch_size.
        per_particle_args: Optional per-particle arguments to be passed to the dataset.
        jax_prng_key: JAX PRNG key for shuffling. Required if shuffle is True.
    """

    if shuffle is True:
        if jax_prng_key is None:
            raise ValueError("jax_prng_key must be provided when shuffle is True.")
        dataloader = jdl.DataLoader(
            CustomJaxDataset(relion_stack_dataset, per_particle_args),
            backend="jax",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=jax_prng_key,
        )
    else:
        dataloader = jdl.DataLoader(
            CustomJaxDataset(relion_stack_dataset, per_particle_args),
            backend="jax",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return dataloader
