"""Smoke tests for the `simulate_data` command."""

import os

import pytest
from cryospax import RelionParticleParameterFile

from cryojax_eo.commands._simulate_data import main

from .conftest import REMOVE, deep_update, run_command


@pytest.fixture
def base_config(sample_path_to_pdb1, sample_path_to_pdb2, tmp_path):
    project = str(tmp_path / "relion_project")
    return {
        "number_of_images": 4,
        "data_sign": "dark-on-light",
        "pixel_size": 2.0,
        "box_size": 16,
        "pad_scale": 1,
        "voltage_in_kilovolts": 300.0,
        "offset_x_in_angstroms": 0.0,
        "offset_y_in_angstroms": 0.0,
        "rotations": {"rotation_distribution": "uniform"},
        "defocus_in_angstroms": 100.0,
        "astigmatism_in_angstroms": 0.0,
        "astigmatism_angle_in_degrees": 0.0,
        "phase_shift": 0.0,
        "amplitude_contrast_ratio": 0.1,
        "spherical_aberration_in_mm": 2.7,
        "ctf_scale_factor": 1.0,
        "envelope_b_factor": 0.0,
        "noise_snr": 0.1,
        "mask_radius": None,
        "mask_rolloff_width": 1.0,
        "rng_seed": 0,
        "atomic_models_params": {
            "path_to_atomic_models": [sample_path_to_pdb1, sample_path_to_pdb2],
            "atomic_models_probabilities": [0.5, 0.5],
            "loads_b_factors": False,
            "atom_selection": "all",
        },
        "path_to_relion_project": project,
        "path_to_starfile": os.path.join(project, "particles.star"),
        "images_per_file": 2,
        "batch_size_for_generation": 2,
        "overwrite": True,
    }


def assert_dataset_written(config):
    starfile = config["path_to_starfile"]
    assert os.path.exists(starfile), f"{starfile} was not written"
    parameter_file = RelionParticleParameterFile(starfile)
    assert len(parameter_file) == config["number_of_images"]
    return parameter_file


@pytest.mark.parametrize("rotation_distribution", ["uniform", "non-uniform"])
def test_all_options(base_config, tmp_path, rotation_distribution):
    """Every option this command accepts, set explicitly in one run."""
    config = deep_update(
        base_config,
        {
            "data_sign": "light-on-dark",
            "pad_scale": 2,
            "voltage_in_kilovolts": 200.0,
            "rotations": {
                "rotation_distribution": rotation_distribution,
                "vmf_kappa": 3.0,
                "vmf_mu": [0.0, 0.0, 1.0],
                "vmf_alpha": 0.5,
            },
            "amplitude_contrast_ratio": 0.07,
            "spherical_aberration_in_mm": 2.0,
            "ctf_scale_factor": 0.9,
            "mask_radius": 6.0,
            "mask_rolloff_width": 2.0,
            "rng_seed": 3,
            "atomic_models_params": {
                "loads_b_factors": True,
                "atom_selection": "not element H",
            },
        },
    )
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_minimal_options(base_config, tmp_path):
    """Only the required fields, every optional field left at its default."""
    config = deep_update(
        base_config,
        {
            "data_sign": REMOVE,
            "pad_scale": REMOVE,
            "voltage_in_kilovolts": REMOVE,
            "offset_x_in_angstroms": REMOVE,
            "offset_y_in_angstroms": REMOVE,
            "rotations": REMOVE,
            "defocus_in_angstroms": REMOVE,
            "astigmatism_in_angstroms": REMOVE,
            "astigmatism_angle_in_degrees": REMOVE,
            "phase_shift": REMOVE,
            "amplitude_contrast_ratio": REMOVE,
            "spherical_aberration_in_mm": REMOVE,
            "ctf_scale_factor": REMOVE,
            "envelope_b_factor": REMOVE,
            "mask_radius": REMOVE,
            "mask_rolloff_width": REMOVE,
            "rng_seed": REMOVE,
            "batch_size_for_generation": REMOVE,
            "atomic_models_params": {
                "loads_b_factors": REMOVE,
                "atom_selection": REMOVE,
            },
        },
    )
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_range_valued_options(base_config, tmp_path):
    """Every field that accepts `[min, max]` instead of a single value."""
    config = deep_update(
        base_config,
        {
            "offset_x_in_angstroms": [-2.0, 2.0],
            "offset_y_in_angstroms": [-2.0, 2.0],
            "defocus_in_angstroms": [100.0, 200.0],
            "astigmatism_in_angstroms": [-1.0, 1.0],
            "astigmatism_angle_in_degrees": [-10.0, 10.0],
            "phase_shift": [0.0, 0.1],
            "envelope_b_factor": [0.0, 10.0],
            "noise_snr": [0.1, 0.5],
        },
    )
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


@pytest.mark.parametrize("data_sign", ["dark-on-light", "light-on-dark"])
def test_data_sign(base_config, tmp_path, data_sign):
    config = deep_update(base_config, {"data_sign": data_sign})
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


@pytest.mark.parametrize("pad_scale", [1, 2])
def test_pad_scale(base_config, tmp_path, pad_scale):
    config = deep_update(base_config, {"pad_scale": pad_scale})
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_mask_radius_defaults_to_box_over_three(base_config, tmp_path):
    config = deep_update(base_config, {"mask_radius": None})
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_mask_radius_larger_than_box_warns(base_config, tmp_path):
    """An oversized mask radius is clamped to the box size, with a warning."""
    config = deep_update(base_config, {"mask_radius": 100.0})
    with pytest.warns(UserWarning, match="Noise radius mask is greater than box size"):
        run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_images_split_across_files(base_config, tmp_path):
    """`images_per_file` controls how many `.mrcs` stacks are written."""
    config = deep_update(base_config, {"number_of_images": 4, "images_per_file": 2})
    run_command(main, tmp_path, config)
    assert_dataset_written(config)
    stacks = [
        name
        for name in os.listdir(config["path_to_relion_project"])
        if name.endswith(".mrcs")
    ]
    assert len(stacks) == 2


def test_atomic_models_from_glob(base_config, tmp_path, pdb_glob_pattern):
    config = deep_update(
        base_config,
        {
            "atomic_models_params": {
                "path_to_atomic_models": pdb_glob_pattern,
                "atomic_models_probabilities": [0.5, 0.5],
            }
        },
    )
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_atomic_model_from_mrc(base_config, tmp_path, sample_path_mrc_file):
    """A real-space voxel grid can be used in place of an atomic model."""
    config = deep_update(
        base_config,
        {
            "box_size": 32,
            "atomic_models_params": {
                "path_to_atomic_models": [sample_path_mrc_file],
                "atomic_models_probabilities": [1.0],
            },
        },
    )
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_atomic_model_from_npz(base_config, tmp_path, sample_path_gmm_model):
    """A fitted gaussian mixture, as written by `fit_gmm_to_atoms`."""
    config = deep_update(
        base_config,
        {
            "atomic_models_params": {
                "path_to_atomic_models": [sample_path_gmm_model],
                "atomic_models_probabilities": [1.0],
            }
        },
    )
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_scalar_model_probability(base_config, tmp_path, sample_path_to_pdb1):
    """A single model may give its probability as a scalar rather than a list."""
    config = deep_update(
        base_config,
        {
            "atomic_models_params": {
                "path_to_atomic_models": [sample_path_to_pdb1],
                "atomic_models_probabilities": 1.0,
            }
        },
    )
    run_command(main, tmp_path, config)
    assert_dataset_written(config)


def test_overwrite_false_on_existing_dataset(base_config, tmp_path):
    config = deep_update(base_config, {"overwrite": True})
    run_command(main, tmp_path, config)
    assert_dataset_written(config)

    config_no_overwrite = deep_update(base_config, {"overwrite": False})
    with pytest.raises(Exception):
        run_command(main, tmp_path, config_no_overwrite, name="no_overwrite.yaml")


def test_config_copy_is_written(base_config, tmp_path):
    """The command copies its config next to the simulated data."""
    run_command(main, tmp_path, base_config)
    assert os.path.exists(
        os.path.join(base_config["path_to_relion_project"], "config.yaml")
    )
