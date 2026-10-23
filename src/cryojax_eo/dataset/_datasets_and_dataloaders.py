from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import jax
import numpy as np
from cryospax import (
    RelionParticleDataset,
)
from jaxtyping import PRNGKeyArray

from cryojax_eo.typing import ParticleStackInfo


class CustomJaxDataset:
    cryojax_dataset: RelionParticleDataset
    per_particle_args: Any | None

    def __init__(
        self,
        cryojax_dataset: RelionParticleDataset,
        per_particle_args: Any | None = None,
    ):
        self.cryojax_dataset = cryojax_dataset
        self.per_particle_args = per_particle_args

    def __getitem__(self, index) -> dict[str, ParticleStackInfo | Any]:
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


class ParticleDataLoader:
    """
    Minimal dataloader that yields batches by indexing the dataset with an
    array of indices, i.e. `dataset[indices]`.

    If `shuffle` is True, the indices are permuted at the start of every epoch
    (every call to `iter`), splitting `jax_prng_key` each time.

    If `prefetch > 0`, up to `prefetch` batches are loaded ahead in a background
    thread while the current batch is being processed. Batches are still yielded
    in order. Prefetching only pays off when iterating over full epochs; with
    the `next(iter(dataloader))` pattern the prefetched batches are discarded,
    so leave it at 0 there.
    """

    def __init__(
        self,
        dataset: Any,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        *,
        jax_prng_key: PRNGKeyArray | None = None,
        prefetch: int = 0,
    ):
        if shuffle and jax_prng_key is None:
            raise ValueError("jax_prng_key must be provided when shuffle is True.")
        if prefetch < 0:
            raise ValueError("prefetch must be non-negative.")

        self.dataset = dataset
        self.indices = np.arange(len(dataset))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.key = jax_prng_key
        self.prefetch = prefetch

    def __len__(self) -> int:
        n_complete_batches, remainder = divmod(len(self.indices), self.batch_size)
        if self.drop_last:
            return n_complete_batches
        return n_complete_batches + int(remainder > 0)

    def __iter__(self) -> Iterator[Any]:
        if self.shuffle:
            if self.key is None:
                raise ValueError("jax_prng_key must be provided when shuffle is True.")
            self.key, subkey = jax.random.split(self.key)
            indices = np.asarray(jax.random.permutation(subkey, self.indices))
        else:
            indices = self.indices

        if self.drop_last:
            indices = indices[: len(indices) - len(indices) % self.batch_size]

        batch_indices = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        if self.prefetch == 0:
            return (self.dataset[idx] for idx in batch_indices)
        return self._prefetching_iterator(batch_indices)

    def _prefetching_iterator(self, batch_indices: list[np.ndarray]) -> Iterator[Any]:
        # A single worker so that reads from the dataset never run concurrently
        executor = ThreadPoolExecutor(max_workers=1)
        futures = deque()
        try:
            for idx in batch_indices:
                futures.append(executor.submit(self.dataset.__getitem__, idx))
                if len(futures) > self.prefetch:
                    yield futures.popleft().result()
            while futures:
                yield futures.popleft().result()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)


def create_dataloader(
    relion_stack_dataset: RelionParticleDataset,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    *,
    per_particle_args: Any | None = None,
    jax_prng_key: PRNGKeyArray | None = None,
    prefetch: int = 0,
) -> ParticleDataLoader:
    """
    Create a DataLoader for a `RelionParticleDataset`.

    **Arguments:**
        - relion_stack_dataset: A `RelionParticleDataset` object.
        - batch_size: The size of each batch.
        - shuffle: Whether to shuffle the dataset.
        - drop_last: Whether to drop the last batch if it is smaller than batch_size.
        - per_particle_args: Optional per-particle arguments to be passed to the dataset.
        - jax_prng_key: JAX PRNG key for shuffling. Required if shuffle is True.
        - prefetch: Number of batches to load ahead in a background thread.
          0 disables prefetching.

    **Returns:**
        - A `ParticleDataLoader` instance, where the dataset is a
          `cryoJAX` `RelionParticleStackDataset`.
    """
    return ParticleDataLoader(
        CustomJaxDataset(relion_stack_dataset, per_particle_args),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        jax_prng_key=jax_prng_key,
        prefetch=prefetch,
    )
