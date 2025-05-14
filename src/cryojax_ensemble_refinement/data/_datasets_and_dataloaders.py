from typing import Optional

import jax_dataloader as jdl
from cryojax.data import (
    ParticleStack,
    RelionParticleStackDataset,
)
from jaxtyping import PRNGKeyArray


class CustomJaxDataset(jdl.Dataset):
    def __init__(self, cryojax_dataset: RelionParticleStackDataset):
        self.cryojax_dataset = cryojax_dataset

    def __getitem__(self, index) -> ParticleStack:
        return self.cryojax_dataset[index]

    def __len__(self) -> int:
        return len(self.cryojax_dataset)


def create_dataloader(
    relion_stack_dataset: RelionParticleStackDataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    *,
    jax_prng_key: Optional[PRNGKeyArray] = None,
):
    """
    Create a Jax DataLoader for a RelionParticleStackDataset.
    **Arguments:**
        relion_stack_dataset: A RelionParticleStackDataset object.
        batch_size: The size of each batch.
        shuffle: Whether to shuffle the dataset.
        drop_last: Whether to drop the last batch if it is smaller than batch_size.
        jax_prng_key: JAX PRNG key for shuffling. Required if shuffle is True.
    """

    if shuffle is True:
        if jax_prng_key is None:
            raise ValueError("jax_prng_key must be provided when shuffle is True.")
        dataloader = jdl.DataLoader(
            CustomJaxDataset(relion_stack_dataset),
            backend="jax",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            generator=jax_prng_key,
        )
    else:
        dataloader = jdl.DataLoader(
            CustomJaxDataset(relion_stack_dataset),
            backend="jax",
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    return dataloader
