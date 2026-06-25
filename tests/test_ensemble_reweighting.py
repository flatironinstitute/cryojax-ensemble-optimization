import argparse
import os

import yaml

from cryojax_eo.commands._run_ensemble_reweighting import main as run_main


def test_run_ensemble_reweighting_from_config(
    sample_path_to_pdb1,
    sample_path_to_relion_project,
    sample_path_to_starfile,
    tmp_path,
):
    output_directory = str(tmp_path / "reweighting_output")

    config = {
        "path_to_structural_files": [
            sample_path_to_pdb1,
            sample_path_to_pdb1,
            sample_path_to_pdb1,
        ],
        "path_to_output_dir": output_directory,
        "atom_selection": "not element H",
        "max_iter": 100,
        "data_params": {
            "path_to_starfile": sample_path_to_starfile,
            "path_to_relion_project": sample_path_to_relion_project,
            "loads_envelope": False,
            "data_sign": "light-on-dark",
        },
    }

    config_path = str(tmp_path / "test_reweighting_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_main(argparse.Namespace(config=config_path, from_likelihoods=None))

    assert os.path.exists(os.path.join(output_directory, "optimized_weights.yaml"))
    assert os.path.exists(os.path.join(output_directory, "log_likelihood_matrix.npz"))
