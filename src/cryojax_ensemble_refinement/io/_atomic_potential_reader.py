import logging
from typing import List, Tuple

import cryojax.simulator as cxs

from ..io._atomic_model_reader import read_atomic_models


def load_atomic_models_as_potentials(
    atomic_models_filenames: List[str],
    *,
    select: str = "all",
    loads_b_factors: bool = False,
) -> Tuple[cxs.GaussianMixtureAtomicPotential]:
    """
    Load atomic models from files and convert them to Gaussian mixture potentials.

    TODO: More general atomic model formats!
    **Arguments:**
        atomic_models_filenames: List of filenames containing atomic models.
            The atomic models are expected to be in pdb format.
        select: Selection string for the atomic models in mdtraj format.
        loads_b_factors: If True, loads b factors from the atomic models.
    **Returns:**
        A tuple of Gaussian mixture potentials.
    """
    potentials = []

    logging.info("Reading atomic models")
    atomic_models_scattering_params = read_atomic_models(
        atomic_models_filenames, select=select, loads_b_factors=loads_b_factors
    )
    for atomic_model in atomic_models_scattering_params.values():
        potential = cxs.GaussianMixtureAtomicPotential(
            atom_positions=atomic_model["atom_positions"],
            gaussian_amplitudes=atomic_model["gaussian_amplitudes"],
            gaussian_variances=atomic_model["gaussian_variances"],
        )
        potentials.append(potential)

    potentials = tuple(potentials)
    logging.info("Potentials generated.")
    return potentials
