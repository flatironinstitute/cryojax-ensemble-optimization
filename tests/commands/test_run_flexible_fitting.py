"""Smoke tests for the `run_flexible_fitting` command."""

import os

import numpy as np
import pytest

from cryojax_eo.commands._run_flexible_fitting import main

from .conftest import (
    REMOVE,
    Replace,
    deep_update,
    openmm_platform_available,
    run_command,
    skip_if_unavailable,
)


pytest.importorskip(
    "openmm",
    reason="OpenMM is an optional dependency required for the flexible fitting pipeline",
)


@pytest.fixture
def base_config(sample_path_to_pdb1, sample_path_mrc_file, tmp_path):
    return {
        "path_to_atomic_model": sample_path_to_pdb1,
        "path_to_prealigned_atomic_model": sample_path_to_pdb1,
        "path_to_output": str(tmp_path / "flexible_fitting_output"),
        "atom_selection": "not element H",
        "loads_b_factors": False,
        "n_steps": 2,
        "rng_seed": 0,
        "reference_volume_params": {
            "path_to_reference_volume": sample_path_mrc_file,
            "path_to_weights": None,
            "flexible_fitting_box_size": 32,
            "rigid_alignment_box_size": 16,
            "reference_volume_voxel_size": None,
            "path_to_volumetric_mask": None,
        },
        "projector_params": {
            "n_steps": 10,
            "bias_constant_in_kjpermol": 1000.0,
            "platform": "CPU",
            "platform_properties": {"Threads": "1"},
            "path_to_initial_state": None,
        },
        "walker_optimizer_params": {
            "type": "steepest_desc",
            "n_steps": 2,
            "step_size": 1.0,
            "n_batches_of_atoms": 1,
        },
    }


def assert_walker_written(config):
    path = os.path.join(config["path_to_output"], "final_walker.npy")
    assert os.path.exists(path), f"{path} was not written"
    walker = np.load(path)
    assert walker.ndim == 2 and walker.shape[1] == 3
    assert np.isfinite(walker).all()


def test_all_options(
    base_config,
    tmp_path,
    sample_path_mrc_file,
    mse_weights_mrc,
    openmm_state_xml,
    atom_indices_txt,
):
    """Every option this command accepts, set explicitly in one run."""
    config = deep_update(
        base_config,
        {
            "atom_selection": atom_indices_txt,
            "loads_b_factors": True,
            "rng_seed": 7,
            "reference_volume_params": {
                "path_to_weights": mse_weights_mrc,
                "reference_volume_voxel_size": 4.0,
                "path_to_volumetric_mask": sample_path_mrc_file,
            },
            "projector_params": {
                "bias_constant_in_kjpermol": [1000.0, 2000.0],
                "path_to_initial_state": openmm_state_xml,
                "md_params": {
                    "forcefield": "amber14-all.xml",
                    "water_model": "amber14/tip3p.xml",
                    "nonbonded_method": "CutoffNonPeriodic",
                    "nonbonded_cutoff_nm": 1.0,
                    "constraints": "HBonds",
                    "temperature_K": 310.0,
                    "friction_per_ps": 2.0,
                    "timestep_ps": 0.001,
                },
            },
            "walker_optimizer_params": {
                "type": "adam",
                "n_batches_of_atoms": 2,
            },
            "early_stopping": {"patience": 2, "rtol": 1e-4, "atol": 1e-4},
        },
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_minimal_options(base_config, tmp_path):
    """Every optional field left at its default."""
    config = deep_update(
        base_config,
        {
            "atom_selection": REMOVE,
            "loads_b_factors": REMOVE,
            "rng_seed": REMOVE,
            "reference_volume_params": {
                "path_to_weights": REMOVE,
                "reference_volume_voxel_size": REMOVE,
                "path_to_volumetric_mask": REMOVE,
                "flexible_fitting_box_size": 32,
                "rigid_alignment_box_size": REMOVE,
            },
            "projector_params": {
                "platform_properties": REMOVE,
                "path_to_initial_state": REMOVE,
            },
            "walker_optimizer_params": {
                "type": REMOVE,
                "n_batches_of_atoms": REMOVE,
            },
        },
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


@pytest.mark.parametrize("optimizer_type", ["steepest_desc", "adam"])
def test_optimizer_type(base_config, tmp_path, optimizer_type):
    config = deep_update(
        base_config, {"walker_optimizer_params": {"type": optimizer_type}}
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_weighted_mse_loss(base_config, tmp_path, mse_weights_mrc):
    """`path_to_weights` selects `ModelToVolumeWeightedMSELossFn`."""
    config = deep_update(
        base_config, {"reference_volume_params": {"path_to_weights": mse_weights_mrc}}
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_volumetric_mask(base_config, tmp_path, sample_path_mrc_file):
    config = deep_update(
        base_config,
        {"reference_volume_params": {"path_to_volumetric_mask": sample_path_mrc_file}},
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_reference_volume_voxel_size_override(base_config, tmp_path):
    config = deep_update(
        base_config, {"reference_volume_params": {"reference_volume_voxel_size": 4.0}}
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_box_sizes(base_config, tmp_path):
    config = deep_update(
        base_config,
        {
            "reference_volume_params": {
                "flexible_fitting_box_size": 16,
                "rigid_alignment_box_size": 8,
            }
        },
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_early_stopping(base_config, tmp_path):
    config = deep_update(
        base_config,
        {"n_steps": 4, "early_stopping": {"patience": 1, "rtol": 1e-4, "atol": 1e-4}},
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_early_stopping_explicit_null(base_config, tmp_path):
    """An explicit `early_stopping: null` must behave like omitting the key."""
    config = deep_update(base_config, {"early_stopping": None})
    run_command(main, tmp_path, config)
    assert_walker_written(config)


@pytest.mark.parametrize("suffix", ["txt", "npy"])
def test_atom_selection_from_file(
    base_config, tmp_path, suffix, atom_indices_txt, atom_indices_npy
):
    indices_file = atom_indices_txt if suffix == "txt" else atom_indices_npy
    config = deep_update(base_config, {"atom_selection": indices_file})
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_initial_state(base_config, tmp_path, openmm_state_xml):
    config = deep_update(
        base_config, {"projector_params": {"path_to_initial_state": openmm_state_xml}}
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_bias_constant_schedule(base_config, tmp_path):
    config = deep_update(
        base_config,
        {"projector_params": {"bias_constant_in_kjpermol": [1000.0, 2000.0]}},
    )
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_loads_b_factors(base_config, tmp_path):
    config = deep_update(base_config, {"loads_b_factors": True})
    run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_rng_seed(base_config, tmp_path):
    """A non-zero seed fixes the OpenMM thermostat's random stream."""
    config = deep_update(base_config, {"rng_seed": 7})
    run_command(main, tmp_path, config)
    assert_walker_written(config)


@pytest.mark.parametrize(
    "nonbonded_method",
    ["NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "Ewald", "LJPME"],
)
def test_nonbonded_method(base_config, tmp_path, nonbonded_method):
    config = deep_update(
        base_config,
        {"projector_params": {"md_params": {"nonbonded_method": nonbonded_method}}},
    )
    with skip_if_unavailable(f"the OpenMM '{nonbonded_method}' nonbonded method"):
        run_command(main, tmp_path, config)
    assert_walker_written(config)


@pytest.mark.parametrize("constraints", ["HBonds", "AllBonds", "HAngles", None])
def test_constraints(base_config, tmp_path, constraints):
    md_params = {"constraints": constraints}
    if constraints in (None, "HAngles"):
        # Neither unconstrained bonds nor HAngles is stable at 2 fs under the
        # steering bias for this system.
        md_params["timestep_ps"] = 0.0005
    config = deep_update(base_config, {"projector_params": {"md_params": md_params}})
    with skip_if_unavailable(f"the OpenMM '{constraints}' constraint setting"):
        run_command(main, tmp_path, config)
    assert_walker_written(config)


def test_alternative_forcefield(base_config, tmp_path):
    config = deep_update(
        base_config,
        {
            "projector_params": {
                "md_params": {
                    "forcefield": "amber99sb.xml",
                    "water_model": "tip3p.xml",
                }
            }
        },
    )
    with skip_if_unavailable("the amber99sb forcefield"):
        run_command(main, tmp_path, config)
    assert_walker_written(config)


@pytest.mark.parametrize("platform", ["CUDA", "OpenCL"])
def test_gpu_platforms(base_config, tmp_path, platform):
    if not openmm_platform_available(platform):
        pytest.skip(f"the OpenMM {platform} platform is not registered here")
    config = deep_update(
        base_config,
        {
            "projector_params": {
                "platform": platform,
                "platform_properties": Replace({"DeviceIndex": "0"}),
            }
        },
    )
    with skip_if_unavailable(f"the OpenMM {platform} platform"):
        run_command(main, tmp_path, config)
    assert_walker_written(config)
