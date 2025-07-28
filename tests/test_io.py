import mdtraj
import pytest
import jax.numpy as jnp

# from jaxtyping import install_import_hook
# with install_import_hook("cryojax_ensemble_optimization", "typeguard.typechecked"):
from cryojax_ensemble_optimization.io import (
    load_atomic_models_as_potentials,
    read_atomic_models,
    read_rna_pair_string,
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
def test_load_atomic_models(path_to_atomic_models, selection_string):
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
        assert (
            atomic_models[model]["gaussian_amplitudes"].shape[0]
            == atom_positions.shape[0]
        )
        assert (
            atomic_models[model]["gaussian_variances"].shape[0] == atom_positions.shape[0]
        )

        assert atomic_models[model]["atom_positions"].shape[1] == 3
        assert atomic_models[model]["gaussian_amplitudes"].shape[1] == 5
        assert atomic_models[model]["gaussian_variances"].shape[1] == 5

    assert len(atomic_models) == len(path_to_atomic_models)


@pytest.mark.parametrize(
    "selection_string",
    [
        "all",
        "name CA",
    ],
)
def test_load_as_potentials(path_to_atomic_models, selection_string):
    """
    Test loading all atoms from the atomic models.
    """
    potentials = load_atomic_models_as_potentials(
        path_to_atomic_models, selection_string=selection_string, loads_b_factors=True
    )

    atom_positions = mdtraj.load(
        path_to_atomic_models[0],
    )
    atom_indices = atom_positions.topology.select(selection_string)
    atom_positions = atom_positions.xyz[0][atom_indices]

    for i in range(len(potentials)):
        potential = potentials[i]
        assert potential.atom_positions.shape == atom_positions.shape
        assert potential.gaussian_amplitudes.shape[0] == atom_positions.shape[0]
        assert potential.gaussian_variances.shape[0] == atom_positions.shape[0]

        assert potential.atom_positions.shape[1] == 3
        assert potential.gaussian_amplitudes.shape[1] == 5
        assert potential.gaussian_variances.shape[1] == 5

    assert len(potentials) == len(path_to_atomic_models)


def test_read_rna_pair_string():
    
    def _sort_rows(a):
        return jnp.sort(a, axis=1)

    def _sort_rows_then_whole(a):
        return jnp.sort(_sort_rows(a), axis=0)
    
    dotbracket_list = ["()", "(.)", "((..))"]
    expected_pairs_list = [jnp.array([[0,1]]), jnp.array([[0,2]]), jnp.array([[0, 5], [1, 4]]), jnp.array([])]

    for dotbracket, expected_pairs in zip(dotbracket_list, expected_pairs_list):
        
        pairs = read_rna_pair_string(dotbracket)
        assert jnp.array_equal(
            _sort_rows_then_whole(pairs),
            _sort_rows_then_whole(expected_pairs)
        )