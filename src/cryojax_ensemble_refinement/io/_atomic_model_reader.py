from pathlib import Path
from typing import Dict, List

from cryojax.constants import (
    convert_b_factor_to_variance,
    get_tabulated_scattering_factor_parameters,
)
from cryojax.io import read_atoms_from_pdb
from jaxtyping import Array, Float


def read_atomic_models(
    atomic_models_filenames: List[str],
    *,
    loads_b_factors: bool = False,
) -> Dict[int, Dict[str, Float[Array, ""]]]:
    """
    **Arguments:**
        atomic_models_filenames: List of filenames of the atomic models.
        loads_b_factors: Whether to load B-factors from the PDB files.
    **Returns:**
        atomic_models_scattering_params: Dictionary of atomic model scattering parameters.
        The dictionary has the following structure:
        {
            i: {
                "atom_positions": atom_positions,
                "gaussian_amplitudes": gaussian_amplitudes,
                "gaussian_variances": gaussian_variances,
            }
        }
        where i is the index of the atomic model, and atom_positions, gaussian_amplitudes,
        and gaussian_variances are numpy arrays of shape (n_atoms, 3),
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
            atomic_models_filenames, loads_b_factors=loads_b_factors
        )
    else:
        raise NotImplementedError(f"File extension {file_extension} not supported.")

    return atomic_models_scattering_params


def _read_atomic_models_from_pdb(
    atomic_models_filenames: List[str],
    loads_b_factors: bool = False,
) -> Dict[int, Dict[str, Float[Array, ""]]]:
    atomic_models_scattering_params = {}
    for i in range(len(atomic_models_filenames)):
        if loads_b_factors:
            atom_positions, atom_identities, b_factors = read_atoms_from_pdb(
                atomic_models_filenames[i],
                center=True,
                loads_b_factors=True,
                select="not element H",
            )

            scattering_factors = get_tabulated_scattering_factor_parameters(
                atom_identities
            )
            gaussian_amplitudes = scattering_factors["a"]
            gaussian_variances = convert_b_factor_to_variance(
                scattering_factors["b"] + b_factors[:, None]
            )

        else:
            atom_positions, atom_identities = read_atoms_from_pdb(
                atomic_models_filenames[i],
                center=True,
                loads_b_factors=False,
                select="not element H",
            )

            scattering_factors = get_tabulated_scattering_factor_parameters(
                atom_identities
            )
            gaussian_amplitudes = scattering_factors["a"]
            gaussian_variances = convert_b_factor_to_variance(scattering_factors["b"])

        atomic_models_scattering_params[i] = {
            "atom_positions": atom_positions,
            "gaussian_amplitudes": gaussian_amplitudes,
            "gaussian_variances": gaussian_variances,
        }

    return atomic_models_scattering_params
