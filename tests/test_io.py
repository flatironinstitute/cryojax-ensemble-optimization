import jax.numpy as jnp
import mdtraj
import pytest

# from jaxtyping import install_import_hook
# with install_import_hook("cryojax_ensemble_optimization", "typeguard.typechecked"):
from cryojax_eo.io import (
    load_gmm_volume_parametrization,
    read_atomic_models,
)


@pytest.fixture
def path_to_atomic_models(sample_path_to_pdb1, sample_path_to_pdb2):
    return [sample_path_to_pdb1, sample_path_to_pdb2]


@pytest.mark.parametrize(
    "selection_string",
    [
        "all",
        "name CA",
    ],
)
def test_load_atomic_models_from_pdb(path_to_atomic_models, selection_string):
    """
    Test loading all atoms from the atomic models.
    """
    atomic_models = read_atomic_models(
        path_to_atomic_models, selection_string=selection_string, loads_b_factors=True
    )

    atom_positions = mdtraj.load(
        path_to_atomic_models[0],
    )
    atom_indices = atom_positions.topology.select(selection_string)
    atom_positions = atom_positions.xyz[0][atom_indices]

    for model in atomic_models:
        assert atomic_models[model]["atom_positions"].shape == atom_positions.shape
        assert atomic_models[model]["amplitudes"].shape[0] == atom_positions.shape[0]
        assert atomic_models[model]["variances"].shape[0] == atom_positions.shape[0]

        assert atomic_models[model]["atom_positions"].shape[1] == 3
        assert atomic_models[model]["amplitudes"].shape[1] == 5
        assert atomic_models[model]["variances"].shape[1] == 5

    assert len(atomic_models) == len(path_to_atomic_models)


def test_load_atomic_models_from_npz(sample_path_gmm_model):
    """
    Test loading atomic models from npz files.
    """
    atomic_models = read_atomic_models(
        [sample_path_gmm_model], selection_string="all", loads_b_factors=False
    )

    npz_data = jnp.load(sample_path_gmm_model)

    for model in atomic_models:
        assert atomic_models[model]["atom_positions"].shape == npz_data["positions"].shape
        assert (
            atomic_models[model]["amplitudes"].shape[0] == npz_data["positions"].shape[0]
        )
        assert (
            atomic_models[model]["variances"].shape[0] == npz_data["positions"].shape[0]
        )

        assert atomic_models[model]["atom_positions"].shape[1] == 3

    assert len(atomic_models) == 1


@pytest.mark.parametrize(
    "selection_string",
    [
        "all",
        "name CA",
    ],
)
def test_load_as_volumes(path_to_atomic_models, selection_string):
    """
    Test loading all atoms from the atomic models.
    """
    volumes = load_gmm_volume_parametrization(
        path_to_atomic_models, selection_string=selection_string, loads_b_factors=True
    )

    atom_positions = mdtraj.load(
        path_to_atomic_models[0],
    )
    atom_indices = atom_positions.topology.select(selection_string)
    atom_positions = atom_positions.xyz[0][atom_indices]

    for i in range(len(volumes)):
        volume = volumes[i]
        assert volume.positions.shape == atom_positions.shape
        assert volume.amplitudes.shape[0] == atom_positions.shape[0]
        assert volume.variances.shape[0] == atom_positions.shape[0]

        assert volume.positions.shape[1] == 3
        assert volume.amplitudes.shape[1] == 5
        assert volume.variances.shape[1] == 5

    assert len(volumes) == len(path_to_atomic_models)
