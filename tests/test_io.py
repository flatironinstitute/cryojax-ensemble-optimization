import mdtraj
import pytest

# from jaxtyping import install_import_hook
# with install_import_hook("cryojax_ensemble_optimization", "typeguard.typechecked"):
from cryojax_eo.io import (
    read_walkers_from_pdbs,
)


@pytest.fixture
def path_to_atomic_models(sample_path_to_pdb1, sample_path_to_pdb2):
    return [sample_path_to_pdb1, sample_path_to_pdb2]


@pytest.mark.parametrize(
    "loads_b_factors",
    [
        True,
        False,
    ],
)
def test_load_atomic_models_from_pdb(path_to_atomic_models, loads_b_factors):
    """
    Test loading all atoms from the atomic models.
    """
    walkers, amplitudes, variances = read_walkers_from_pdbs(
        path_to_atomic_models,
        loads_b_factors=loads_b_factors,
    )

    atom_positions = mdtraj.load(
        path_to_atomic_models[0],
    )
    atom_positions = atom_positions.xyz[0]

    assert walkers.shape == (len(path_to_atomic_models), atom_positions.shape[0], 3)
    assert amplitudes.shape == (atom_positions.shape[0], 5)
    assert variances.shape == (atom_positions.shape[0], 5)


def test_inconsistent_pdbs(sample_path_groel_pdb, sample_path_to_pdb1):
    """
    Test that an error is raised when loading atomic models from PDB files with
    inconsistent atomic composition.
    """
    with pytest.raises(ValueError):
        read_walkers_from_pdbs(
            [sample_path_groel_pdb, sample_path_to_pdb1],
            loads_b_factors=False,
        )
