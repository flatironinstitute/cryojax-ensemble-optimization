from pathlib import Path
from typing import Dict, List

import jax.numpy as jnp
import mdtraj
from cryojax.constants import (
    b_factor_to_variance,
)
from cryojax.io import read_atoms_from_pdb
from cryojax.simulator import PengScatteringFactorParameters
from jaxtyping import Array, Float


def read_atomic_models(
    atomic_models_filenames: List[str],
    *,
    selection_string: str = "all",
    loads_b_factors: bool = False,
) -> Dict[int, Dict[str, Float[Array, ""]]]:
    """
    **Arguments:**
        atomic_models_filenames: List of filenames of the atomic models.
        selection_string: Selection string for the atomic models in mdtraj format.
        loads_b_factors: Whether to load B-factors from the PDB files.
    **Returns:**
        atomic_models_scattering_params: Dictionary of atomic model scattering parameters.
        The dictionary has the following structure:
        {
            i: {
                "atom_positions": atom_positions,
                "amplitudes": amplitudes,
                "variances": variances,
            }
        }
        where i is the index of the atomic model, and atom_positions, amplitudes,
        and variances are numpy arrays of shape (n_atoms, 3),
        (n_atoms, n_gaussians_per_atom), and (n_atoms,), respectively.
    """

    # Doing checks here again
    # In case people don't use the config validator
    file_extension = Path(atomic_models_filenames[0]).suffix
    assert all(
        [Path(file).suffix == file_extension for file in atomic_models_filenames]
    ), "All files must have the same extension."

    assert all(
        [Path(file).exists() for file in atomic_models_filenames]
    ), "Some files do not exist."

    if file_extension == ".pdb":
        atomic_models_scattering_params = _read_atomic_models_from_pdb(
            atomic_models_filenames,
            selection_string=selection_string,
            loads_b_factors=loads_b_factors,
        )
    elif file_extension == ".npz":
        atomic_models_scattering_params = _read_atomic_models_from_npz(
            atomic_models_filenames,
        )
    else:
        raise NotImplementedError(f"File extension {file_extension} not supported.")

    return atomic_models_scattering_params


def _read_atomic_models_from_npz(
    atomic_models_filenames: List[str],
) -> Dict[int, Dict[str, Float[Array, ""]]]:
    atomic_models_scattering_params = {}

    for i, filename in enumerate(atomic_models_filenames):
        data = jnp.load(filename)

        try:
            atomic_models_scattering_params[i] = {
                "atom_positions": data["bead_positions"],
                "amplitudes": data["amplitudes"],
                "variances": data["variances"],
            }
        except KeyError as e:
            raise ValueError(
                f"Missing key in npz file {filename}: {e}. "
                + "Keys should be 'bead_positions', 'amplitudes', "
                + "and 'variances'."
            )

    return atomic_models_scattering_params


def _read_atomic_models_from_pdb(
    atomic_models_filenames: List[str],
    selection_string: str = "all",
    loads_b_factors: bool = False,
) -> Dict[int, Dict[str, Float[Array, ""]]]:
    atomic_models_scattering_params = {}

    atoms_for_alignment = mdtraj.load(atomic_models_filenames[0])
    atoms_for_alignment = atoms_for_alignment.center_coordinates()
    atom_indices = atoms_for_alignment.topology.select(selection_string)

    for i in range(len(atomic_models_filenames)):
        if loads_b_factors:
            _, atom_types, b_factors = read_atoms_from_pdb(
                atomic_models_filenames[i],
                center=True,
                loads_b_factors=True,
                selection_string=selection_string,
            )

            scattering_factors = PengScatteringFactorParameters(atom_types)
            amplitudes = scattering_factors.a
            variances = b_factor_to_variance(scattering_factors.b + b_factors[:, None])

        else:
            _, atom_types = read_atoms_from_pdb(
                atomic_models_filenames[i],
                center=True,
                loads_b_factors=False,
                selection_string=selection_string,
            )

            scattering_factors = PengScatteringFactorParameters(atom_types)
            amplitudes = scattering_factors.a
            variances = b_factor_to_variance(scattering_factors.b)

        atom_positions = mdtraj.load(
            atomic_models_filenames[i],
        )

        atom_positions = atom_positions.superpose(
            atoms_for_alignment,
            frame=0,
            atom_indices=atom_positions.topology.select("name CA"),
        )

        atomic_models_scattering_params[i] = {
            "atom_positions": atom_positions.xyz[0][atom_indices] * 10.0,
            "amplitudes": amplitudes,
            "variances": variances,
        }

    return atomic_models_scattering_params
