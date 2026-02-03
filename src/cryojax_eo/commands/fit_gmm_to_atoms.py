#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import sys
from pathlib import Path

import jax.numpy as jnp
import yaml

from ..internal._config_validators.gmm_fit_config import GMMFitConfig
from ..internal._config_validators.utils import _validate_files_with_type
from ..utils._gmm_fitting import make_gmm_model_from_atomic_model


def add_args(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config (yaml) file"
    )
    # add arguments for input and output files
    parser.add_argument(
        "-o", "--output_file", type=str, default=None, help="Path to the output file"
    )
    parser.add_argument(
        "-i", "--input_pdb", type=str, default=None, help="Path to the reference file"
    )
    return parser


def mkbasedir(out):
    if not os.path.exists(out):
        try:
            os.makedirs(out)
        except (FileExistsError, PermissionError):
            raise ValueError("Output path does not exist and cannot be created.")
    return


def warnexists(out):
    if os.path.exists(out):
        Warning(f"Warning: {out} already exists. Overwriting.")


def main(args):
    config_file = args.config
    input_pdb = args.input_pdb
    output_file = Path(args.output_file)

    _validate_files_with_type(input_pdb, [".pdb"])
    if config_file is None:
        config = GMMFitConfig()
        config_file = "default_config.yaml"
        config_dict = dict(config.model_dump())
    else:
        with open(config_file) as f:
            config_dict = yaml.safe_load(f)
            config = GMMFitConfig(**config_dict)

    basedir = output_file.parent
    warnexists(basedir)
    mkbasedir(basedir)
    print(f"A copy of the config file and the log will be written to {basedir}")
    sys.stdout.flush()

    # make copy of config to output_path

    logger = logging.getLogger()
    logger_fname = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    logger_fname = os.path.join(basedir, logger_fname + ".log")
    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    config_fname = os.path.basename(config_file)
    with open(os.path.join(basedir, config_fname), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logging.info(
        f"A copy of the used config file has been written "
        f"to {os.path.join(basedir, config_fname)}"
    )

    logging.info("Simulating particle stack...")

    fitted_gmm = make_gmm_model_from_atomic_model(
        pdb_file=input_pdb,
        box_size=config.box_size,
        voxel_size=config.voxel_size,
        fit_selection_string=config.fit_selection_string,
        init_amp=config.init_log_amp,
        init_var=config.init_log_var,
        n_gaussians_per_bead=config.n_gaussians_per_bead,
        atol=config.atol,
        rtol=config.rtol,
        max_steps=config.max_steps,
    )

    jnp.savez(
        output_file,
        positions=fitted_gmm.positions,
        amplitudes=fitted_gmm.amplitudes,
        variances=fitted_gmm.variances,
    )

    logging.info("Simulation complete.")
    logging.info(f"Output written to {basedir}")

    return


def main_cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(GMMFitConfig.model_json_schema(), indent=4),
    )

    main(add_args(parser).parse_args())


if __name__ == "__main__":
    main_cli()
