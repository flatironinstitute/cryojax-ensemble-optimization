#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import sys
import warnings

import cryojax.simulator as cxs
import jax
import yaml
from cryojax.image.operators import CircularCosineMask

from ..data import generate_relion_parameter_dataset, simulate_relion_dataset
from ..internal import DatasetGeneratorConfig
from ..io import load_atomic_models_as_potentials


warnings.filterwarnings("ignore", module="MDAnalysis")


def add_args(parser):
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the config (yaml) file"
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
        Warning("Warning: {} already exists. Overwriting.".format(out))


def simulate_particle_stack_from_config(config: DatasetGeneratorConfig):
    seed = config.rng_seed
    key = jax.random.key(seed)

    key_param, key_stack = jax.random.split(key)
    parameter_dataset = generate_relion_parameter_dataset(key_param, config)

    potentials = load_atomic_models_as_potentials(
        config.atomic_models_params["atomic_models_filenames"],
        select=config.atomic_models_params["atomic_models_select"],
        loads_b_factors=config.atomic_models_params["loads_b_factors"],
    )

    mask = CircularCosineMask(
        coordinate_grid=parameter_dataset[0].instrument_config.coordinate_grid_in_pixels,
        radius=config.mask_radius,
        rolloff_width=config.mask_rolloff_width,
    )
    simulate_relion_dataset(
        key=key_stack,
        parameter_dataset=parameter_dataset,
        potentials=potentials,
        potential_integrator=cxs.GaussianMixtureProjection(),
        ensemble_probabilities=config.atomic_models_params["atomic_models_probabilities"],
        mask=mask,
        noise_snr_range=config.noise_snr,
        overwrite=config.overwrite,
    )
    return


def main(args):
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = DatasetGeneratorConfig(**config_dict)

    project_path = config.path_to_relion_project
    warnexists(project_path)
    mkbasedir(project_path)
    print(
        "A copy of the config file and the log will be written to {}".format(project_path)
    )
    sys.stdout.flush()

    # make copy of config to output_path

    logger = logging.getLogger()
    logger_fname = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    logger_fname = os.path.join(project_path, logger_fname + ".log")
    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    config_fname = os.path.basename(args.config)
    with open(os.path.join(project_path, config_fname), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(project_path, config_fname)
        )
    )

    logging.info("Simulating particle stack...")
    simulate_particle_stack_from_config(config)
    logging.info("Simulation complete.")
    logging.info("Output written to {}".format(project_path))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(DatasetGeneratorConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())
