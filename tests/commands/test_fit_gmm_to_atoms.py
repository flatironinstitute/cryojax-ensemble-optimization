"""Smoke tests for the `fit_gmm_to_atoms` command."""

import argparse
import os

import numpy as np
import pytest

from cryojax_eo.commands._fit_gmm_to_atoms import main

from .conftest import deep_update, write_config


@pytest.fixture
def base_config():
    """Every `GMMFitConfig` field, at values small enough to fit quickly."""
    return {
        "box_size": 16,
        "voxel_size": 2.0,
        "fit_selection_string": "name CA",
        "init_log_amp": 40.0,
        "init_log_var": 1.0,
        "n_gaussians_per_bead": 1,
        "atol": 1e-3,
        "rtol": 1e-3,
        "max_steps": 50,
    }


def assert_gmm_written(output_file, n_gaussians_per_bead=1):
    assert os.path.exists(output_file), f"{output_file} was not written"
    with np.load(output_file) as fitted:
        positions = fitted["positions"]
        amplitudes = fitted["amplitudes"]
        variances = fitted["variances"]
    assert positions.ndim == 2 and positions.shape[1] == 3
    assert amplitudes.shape == (positions.shape[0], n_gaussians_per_bead)
    assert variances.shape == amplitudes.shape
    assert np.isfinite(positions).all()
    assert (variances > 0).all()


def test_all_options(base_config, tmp_path, sample_path_to_pdb1):
    """Every option this command accepts, set explicitly in one run."""
    output_file = str(tmp_path / "output" / "gmm_model.npz")
    config_path = write_config(tmp_path, base_config)
    main(
        argparse.Namespace(
            config=config_path, input_pdb=sample_path_to_pdb1, output_file=output_file
        )
    )
    assert_gmm_written(output_file)
    # The command copies its config and writes a log next to the output.
    assert os.path.exists(os.path.join(os.path.dirname(output_file), "config.yaml"))


@pytest.mark.parametrize("n_gaussians_per_bead", [1, 2])
def test_n_gaussians_per_bead(
    base_config, tmp_path, sample_path_to_pdb1, n_gaussians_per_bead
):
    output_file = str(tmp_path / "gmm_model.npz")
    config = deep_update(base_config, {"n_gaussians_per_bead": n_gaussians_per_bead})
    main(
        argparse.Namespace(
            config=write_config(tmp_path, config),
            input_pdb=sample_path_to_pdb1,
            output_file=output_file,
        )
    )
    assert_gmm_written(output_file, n_gaussians_per_bead=n_gaussians_per_bead)


def test_creates_missing_output_directory(base_config, tmp_path, sample_path_to_pdb1):
    output_file = str(tmp_path / "does" / "not" / "exist" / "gmm_model.npz")
    main(
        argparse.Namespace(
            config=write_config(tmp_path, base_config),
            input_pdb=sample_path_to_pdb1,
            output_file=output_file,
        )
    )
    assert_gmm_written(output_file)


def test_without_config_file(tmp_path, c2_beads_pdb):
    """With no `--config`, the built-in defaults are used and written out.

    The default `fit_selection_string` is `name "C2"`, which matches nothing in
    either bundled protein model, so this branch needs a coarse-grained input.
    """
    output_file = str(tmp_path / "output" / "gmm_model.npz")
    main(argparse.Namespace(config=None, input_pdb=c2_beads_pdb, output_file=output_file))
    assert_gmm_written(output_file)
    assert os.path.exists(
        os.path.join(os.path.dirname(output_file), "default_config.yaml")
    )


def test_rejects_non_pdb_input(base_config, tmp_path, sample_path_mrc_file):
    output_file = str(tmp_path / "gmm_model.npz")
    with pytest.raises(Exception):
        main(
            argparse.Namespace(
                config=write_config(tmp_path, base_config),
                input_pdb=sample_path_mrc_file,
                output_file=output_file,
            )
        )
