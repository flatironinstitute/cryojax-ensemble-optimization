"""Smoke tests for the `run_ensemble_optimization` command.

Each test runs the full pipeline once on the 42-atom alanine system with the
smallest step counts the config allows, and asserts the run produced its output.
The question being asked is "does this combination of options run?".
"""

import os

import numpy as np
import pytest

from cryojax_eo.commands._run_ensemble_optimization import main

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
    reason="OpenMM is an optional dependency required for the ensemble optimization "
    "pipeline",
)


@pytest.fixture
def base_config(
    sample_path_to_pdb1,
    sample_path_to_pdb2,
    simulated_particle_stack,
    map_frame_pdb,
    tmp_path,
):
    """A minimal-but-explicit config. Tests override only what they exercise."""
    return {
        "path_to_atomic_models": [sample_path_to_pdb1, sample_path_to_pdb2],
        "path_to_output": str(tmp_path / "ensemble_optimization_output"),
        "atom_selection": "not element H",
        "loads_b_factors": False,
        "n_steps": 1,
        "rng_seed": 0,
        "data_params": {
            "path_to_starfile": simulated_particle_stack["path_to_starfile"],
            "path_to_relion_project": simulated_particle_stack["path_to_relion_project"],
            "loads_envelope": False,
            "path_to_volumetric_mask": None,
            "data_sign": "light-on-dark",
        },
        "projector_params": {
            "n_steps": 10,
            "bias_constant_in_kjpermol": 1000.0,
            "platform": "CPU",
            "platform_properties": {"Threads": "1"},
            "path_to_initial_states": None,
        },
        "likelihood_optimizer_params": {
            "n_steps": 1,
            "step_size": 1.0,
            "n_batches_per_step": 1,
            "batch_size": 2,
            "initial_weights": None,
            "estimates_poses": False,
            "volume_integrator_backend": {
                "enable_pallas": False,
                "spread_mode": "exact",
                "sampling_mode": "average",
            },
        },
        "alignment_params": {
            "path_to_prealigned_atomic_model": map_frame_pdb,
            "path_to_reference_volume": None,
            "downsample_box_size": 32,
            "reference_volume_voxel_size": None,
        },
    }


def assert_ensemble_written(config, n_models):
    path = os.path.join(config["path_to_output"], "final_ensemble.npz")
    assert os.path.exists(path), f"{path} was not written"
    with np.load(path) as output:
        weights = np.asarray(output["weights"])
        walkers = np.asarray(output["walkers"])
    assert weights.shape == (n_models,)
    assert np.isfinite(walkers).all()
    assert np.isfinite(weights).all()
    if n_models > 1:
        # The weights are renormalized by the postprocessing step, which only
        # runs for an ensemble of more than one walker.
        assert weights.sum() == pytest.approx(1.0)


def test_all_options(
    base_config,
    tmp_path,
    sample_path_mrc_file,
    openmm_state_xml,
    atom_indices_txt,
):
    """Every option this command accepts, set explicitly in one run."""
    config = deep_update(
        base_config,
        {
            "atom_selection": atom_indices_txt,
            "loads_b_factors": True,
            "n_steps": 2,
            "rng_seed": 1,
            "data_params": {
                "loads_envelope": True,
                "path_to_volumetric_mask": sample_path_mrc_file,
                "data_sign": "dark-on-light",
            },
            "projector_params": {
                "bias_constant_in_kjpermol": [1000.0, 2000.0],
                "path_to_initial_states": [openmm_state_xml, openmm_state_xml],
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
            "likelihood_optimizer_params": {
                "n_batches_per_step": 2,
                "initial_weights": [1.0, 3.0],
            },
            "alignment_params": {
                "path_to_reference_volume": sample_path_mrc_file,
                "downsample_box_size": 16,
                "reference_volume_voxel_size": 4.0,
            },
        },
    )
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_minimal_options(base_config, tmp_path, sample_path_to_pdb1):
    """A single walker with every optional field left at its default."""
    config = deep_update(
        base_config,
        {
            "path_to_atomic_models": [sample_path_to_pdb1],
            "atom_selection": REMOVE,
            "loads_b_factors": REMOVE,
            "rng_seed": REMOVE,
            "data_params": {
                "loads_envelope": REMOVE,
                "path_to_volumetric_mask": REMOVE,
                "data_sign": REMOVE,
            },
            "projector_params": {
                "platform_properties": REMOVE,
                "path_to_initial_states": REMOVE,
            },
            "likelihood_optimizer_params": {
                "n_batches_per_step": REMOVE,
                "initial_weights": REMOVE,
                "estimates_poses": REMOVE,
                "volume_integrator_backend": REMOVE,
            },
            "alignment_params": {
                "path_to_reference_volume": REMOVE,
                "downsample_box_size": REMOVE,
                "reference_volume_voxel_size": REMOVE,
            },
        },
    )
    run_command(main, tmp_path, config)
    # A single walker skips the postprocessing branch.
    assert_ensemble_written(config, n_models=1)


@pytest.mark.parametrize("sampling_mode", ["average", "point"])
@pytest.mark.parametrize("spread_mode", ["exact", "local"])
@pytest.mark.parametrize("enable_pallas", [False, True])
def test_volume_integrator_backend(
    base_config, tmp_path, spread_mode, sampling_mode, enable_pallas
):
    backend = {
        "enable_pallas": enable_pallas,
        "spread_mode": spread_mode,
        "sampling_mode": sampling_mode,
    }
    if spread_mode == "local":
        backend["spread_width_in_stds"] = 6.0
    config = deep_update(
        base_config,
        {"likelihood_optimizer_params": {"volume_integrator_backend": backend}},
    )
    with skip_if_unavailable("the Pallas/Triton spreading backend"):
        run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_estimates_poses(base_config, tmp_path):
    config = deep_update(
        base_config,
        {
            "likelihood_optimizer_params": {
                "estimates_poses": True,
                "pose_search_params": {
                    "n_rounds": 0,
                    "initial_resolution": 1,
                    "n_candidates": 2,
                    "n_angles_in_parallel": 2,
                    "shift_search_range_in_angstroms": 2.0,
                },
            }
        },
    )
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


@pytest.mark.parametrize("suffix", ["txt", "npy"])
def test_atom_selection_from_file(
    base_config, tmp_path, suffix, atom_indices_txt, atom_indices_npy
):
    indices_file = atom_indices_txt if suffix == "txt" else atom_indices_npy
    config = deep_update(base_config, {"atom_selection": indices_file})
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_atomic_models_from_glob(base_config, tmp_path, pdb_glob_pattern):
    config = deep_update(base_config, {"path_to_atomic_models": pdb_glob_pattern})
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


@pytest.mark.parametrize("data_sign", ["dark-on-light", "light-on-dark"])
def test_data_sign(base_config, tmp_path, data_sign):
    config = deep_update(base_config, {"data_params": {"data_sign": data_sign}})
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_loads_envelope(base_config, tmp_path):
    config = deep_update(base_config, {"data_params": {"loads_envelope": True}})
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_volumetric_mask(base_config, tmp_path, sample_path_mrc_file):
    config = deep_update(
        base_config, {"data_params": {"path_to_volumetric_mask": sample_path_mrc_file}}
    )
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_reference_volume_alignment(base_config, tmp_path, sample_path_mrc_file):
    config = deep_update(
        base_config,
        {
            "alignment_params": {
                "path_to_reference_volume": sample_path_mrc_file,
                "downsample_box_size": 16,
                "reference_volume_voxel_size": 4.0,
            }
        },
    )
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


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
    assert_ensemble_written(config, n_models=2)


@pytest.mark.parametrize("constraints", ["HBonds", "AllBonds", "HAngles", None])
def test_constraints(base_config, tmp_path, constraints, sample_path_to_pdb1):
    md_params = {"constraints": constraints}
    overrides = {"projector_params": {"md_params": md_params}}
    n_models = 2
    if constraints is None:
        # Without constrained bonds a 2 fs step is unstable for this system.
        md_params["timestep_ps"] = 0.0005
    elif constraints == "HAngles":
        # `ala_model_1.pdb` blows up under HAngles in OpenMM on its own, with no
        # pipeline involved, so this case runs on the one model that is stable.
        md_params["timestep_ps"] = 0.0005
        overrides["path_to_atomic_models"] = [sample_path_to_pdb1]
        n_models = 1
    config = deep_update(base_config, overrides)
    with skip_if_unavailable(f"the OpenMM '{constraints}' constraint setting"):
        run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=n_models)


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
    assert_ensemble_written(config, n_models=2)


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
    assert_ensemble_written(config, n_models=2)


def test_initial_states(base_config, tmp_path, openmm_state_xml):
    config = deep_update(
        base_config,
        {"projector_params": {"path_to_initial_states": [openmm_state_xml] * 2}},
    )
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_bias_constant_schedule(base_config, tmp_path):
    config = deep_update(
        base_config,
        {
            "n_steps": 2,
            "projector_params": {"bias_constant_in_kjpermol": [1000.0, 2000.0]},
        },
    )
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_initial_weights_are_normalized(base_config, tmp_path):
    config = deep_update(
        base_config, {"likelihood_optimizer_params": {"initial_weights": [1.0, 3.0]}}
    )
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)


def test_loads_b_factors(base_config, tmp_path):
    config = deep_update(base_config, {"loads_b_factors": True})
    run_command(main, tmp_path, config)
    assert_ensemble_written(config, n_models=2)
