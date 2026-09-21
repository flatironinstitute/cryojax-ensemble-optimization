"""Smoke tests for the `run_ensemble_reweighting` command.

This command needs no OpenMM: it only evaluates likelihoods and optimizes
weights, so it stays importable without the optional MD dependency.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from cryojax_eo.commands._run_ensemble_reweighting import main

from .conftest import REMOVE, deep_update, run_command


@pytest.fixture
def base_config(distinct_pdb_copies, simulated_particle_stack, tmp_path):
    return {
        "path_to_structural_files": list(distinct_pdb_copies),
        "path_to_output_dir": str(tmp_path / "reweighting_output"),
        "atom_selection": "not element H",
        "loads_b_factors": False,
        "data_params": {
            "path_to_starfile": simulated_particle_stack["path_to_starfile"],
            "path_to_relion_project": simulated_particle_stack["path_to_relion_project"],
            "loads_envelope": False,
            "path_to_volumetric_mask": None,
            "data_sign": "light-on-dark",
        },
        "max_iter": 50,
        "tol": 1e-4,
        "n_images_in_parallel": 1,
        "max_volume_repr_resolution": None,
        "estimates_poses": False,
    }


def assert_weights_written(config, n_models):
    out_dir = config["path_to_output_dir"]
    weights_path = os.path.join(out_dir, "optimized_weights.yaml")
    assert os.path.exists(weights_path), f"{weights_path} was not written"
    with open(weights_path) as f:
        weights = yaml.safe_load(f)
    assert len(weights) == n_models
    assert sum(weights.values()) == pytest.approx(1.0)
    return weights


def test_all_options(
    base_config, tmp_path, sample_path_mrc_file, simulated_particle_stack
):
    """Every option this command accepts, set explicitly in one run."""
    config = deep_update(
        base_config,
        {
            "loads_b_factors": True,
            "data_params": {
                "loads_envelope": True,
                "path_to_volumetric_mask": sample_path_mrc_file,
                "data_sign": "dark-on-light",
            },
            "max_iter": 20,
            "tol": 1e-6,
            "n_images_in_parallel": 2,
            "max_volume_repr_resolution": 8.0,
        },
    )
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)

    matrix_path = os.path.join(config["path_to_output_dir"], "log_likelihood_matrix.npz")
    assert os.path.exists(matrix_path)
    with np.load(matrix_path) as matrix:
        assert len(matrix.files) == 3
        for key in matrix.files:
            assert matrix[key].shape == (simulated_particle_stack["number_of_images"],)


def test_minimal_options(base_config, tmp_path):
    """Every optional field left at its default."""
    config = deep_update(
        base_config,
        {
            "atom_selection": REMOVE,
            "loads_b_factors": REMOVE,
            "data_params": {
                "loads_envelope": REMOVE,
                "path_to_volumetric_mask": REMOVE,
                "data_sign": REMOVE,
            },
            "tol": REMOVE,
            "n_images_in_parallel": REMOVE,
            "max_volume_repr_resolution": REMOVE,
            "estimates_poses": REMOVE,
        },
    )
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)


def test_structural_files_from_mrc(base_config, tmp_path, sample_path_mrc_file):
    """A real-space voxel grid can be scored directly, with no atomic model."""
    config = deep_update(
        base_config, {"path_to_structural_files": [sample_path_mrc_file]}
    )
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=1)


def test_structural_files_mixed_formats(
    base_config, tmp_path, distinct_pdb_copies, sample_path_mrc_file
):
    config = deep_update(
        base_config,
        {"path_to_structural_files": [distinct_pdb_copies[0], sample_path_mrc_file]},
    )
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=2)

    # The `.pdb` branch renders a voxel grid and writes it out; `.mrc` does not.
    out_dir = config["path_to_output_dir"]
    stem = Path(distinct_pdb_copies[0]).stem
    assert os.path.exists(os.path.join(out_dir, f"{stem}_voxel_grid.mrc"))


def test_unsupported_structural_file(base_config, tmp_path, sample_path_gmm_model):
    """A file type the config does not accept is rejected before any work."""
    config = deep_update(
        base_config, {"path_to_structural_files": [sample_path_gmm_model]}
    )
    with pytest.raises(Exception):
        run_command(main, tmp_path, config, from_likelihoods=None)


@pytest.mark.parametrize("data_sign", ["dark-on-light", "light-on-dark"])
def test_data_sign(base_config, tmp_path, data_sign):
    config = deep_update(base_config, {"data_params": {"data_sign": data_sign}})
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)


def test_loads_envelope(base_config, tmp_path):
    config = deep_update(base_config, {"data_params": {"loads_envelope": True}})
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)


def test_volumetric_mask(base_config, tmp_path, sample_path_mrc_file):
    config = deep_update(
        base_config, {"data_params": {"path_to_volumetric_mask": sample_path_mrc_file}}
    )
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)


def test_max_volume_repr_resolution(base_config, tmp_path):
    """A resolution cutoff low-pass filters the volume before scoring."""
    config = deep_update(base_config, {"max_volume_repr_resolution": 8.0})
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)


def test_n_images_in_parallel(base_config, tmp_path):
    config = deep_update(base_config, {"n_images_in_parallel": 2})
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)


def test_loads_b_factors(base_config, tmp_path):
    config = deep_update(base_config, {"loads_b_factors": True})
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=3)


def test_estimates_poses(
    base_config, tmp_path, distinct_pdb_copies, simulated_particle_stack
):
    """Pose search writes the estimated poses to a new starfile per structure."""
    config = deep_update(
        base_config,
        {
            "path_to_structural_files": distinct_pdb_copies[:1],
            "estimates_poses": True,
            "pose_search_params": {
                "n_rounds": 0,
                "initial_resolution": 1,
                "n_candidates": 2,
                "n_angles_in_parallel": 2,
                "shift_search_range_in_angstroms": 2.0,
            },
        },
    )
    run_command(main, tmp_path, config, from_likelihoods=None)
    assert_weights_written(config, n_models=1)

    stem = Path(distinct_pdb_copies[0]).stem
    starfile = os.path.join(config["path_to_output_dir"], f"{stem}_starfile.star")
    assert os.path.exists(starfile), f"{starfile} was not written"


@pytest.mark.parametrize("explicit_path", [False, True])
def test_from_likelihoods(base_config, tmp_path, explicit_path):
    """`--from-likelihoods` re-optimizes without recomputing the likelihoods."""
    run_command(main, tmp_path, base_config, from_likelihoods=None)
    first = assert_weights_written(base_config, n_models=3)

    matrix_path = os.path.join(
        base_config["path_to_output_dir"], "log_likelihood_matrix.npz"
    )
    from_likelihoods = matrix_path if explicit_path else True
    run_command(
        main,
        tmp_path,
        base_config,
        name="config_rerun.yaml",
        from_likelihoods=from_likelihoods,
    )
    second = assert_weights_written(base_config, n_models=3)

    # Weight optimization is deterministic given the same likelihood matrix.
    assert second == pytest.approx(first)


def test_from_likelihoods_missing_file(base_config, tmp_path):
    with pytest.raises(FileNotFoundError, match="No pre-computed likelihoods found"):
        run_command(main, tmp_path, base_config, from_likelihoods=True)
