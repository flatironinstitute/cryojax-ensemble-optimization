import logging
from typing import List, Tuple

import cryojax.simulator as cxs

from ..io._atomic_model_reader import read_atomic_models


def load_atomic_models_as_volumes(
    atomic_models_filenames: List[str],
    *,
    selection_string: str = "all",
    loads_b_factors: bool = False,
) -> Tuple[cxs.GaussianMixtureVolume]:
    """
    Load atomic models from files and convert them to Gaussian mixture volumes.

    TODO: More general atomic model formats!
    **Arguments:**
        atomic_models_filenames: List of filenames containing atomic models.
            The atomic models are expected to be in pdb format.
        selection_string: Selection string for the atomic models in mdtraj format.
        loads_b_factors: If True, loads b factors from the atomic models.
    **Returns:**
        A tuple of Gaussian mixture volumes.
    """
    volumes = []

    logging.info("Reading atomic models")
    atomic_models_scattering_params = read_atomic_models(
        atomic_models_filenames,
        selection_string=selection_string,
        loads_b_factors=loads_b_factors,
    )
    for atomic_model in atomic_models_scattering_params.values():
        volume = cxs.GaussianMixtureVolume(
            positions=atomic_model["atom_positions"],
            amplitudes=atomic_model["amplitudes"],
            variances=atomic_model["variances"],
        )
        volumes.append(volume)

    volumes = tuple(volumes)
    logging.info("Potentials generated.")
    return volumes
