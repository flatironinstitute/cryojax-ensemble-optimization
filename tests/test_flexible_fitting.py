import argparse
import os

import pytest
import yaml

from cryojax_eo.commands._run_flexible_fitting import main as run_main


pytest.importorskip(
    "openmm",
    reason="OpenMM is an optional dependency required for the flexible fitting pipeline",
)


def test_run_flexible_fitting_from_config(
    sample_path_groel_pdb,
    sample_path_mrc_file,
    tmp_path,
):
    output_directory = str(tmp_path / "flexible_fitting_output")

    config = {
        "path_to_atomic_model": sample_path_groel_pdb,
        "path_to_prealigned_atomic_model": sample_path_groel_pdb,
        "path_to_output": output_directory,
        "atom_selection": "name CA",
        "n_steps": 2,
        "reference_volume_params": {
            "path_to_reference_volume": sample_path_mrc_file,
            "flexible_fitting_box_size": 32,
            "rigid_alignment_box_size": 16,
        },
        "projector_params": {
            "n_steps": 10,
            "bias_constant_in_kjpermol": 1000.0,
            "platform": "CPU",
        },
        "walker_optimizer_params": {
            "n_steps": 2,
            "step_size": 1.0,
        },
    }

    config_path = str(tmp_path / "test_flexible_fitting_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_main(argparse.Namespace(config=config_path))

    assert os.path.exists(os.path.join(output_directory, "final_walker.npy"))
