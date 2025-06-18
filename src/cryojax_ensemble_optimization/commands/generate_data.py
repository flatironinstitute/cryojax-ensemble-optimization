#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import sys

import cryojax.simulator as cxs
import jax
import yaml
from cryojax.image.operators import CircularCosineMask

from ..data import generate_relion_parameter_file, simulate_relion_dataset
from ..internal import DatasetGeneratorConfig
from ..io import load_atomic_models_as_potentials


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
    parameter_file = generate_relion_parameter_file(key_param, config)

    # dumping so serialization happens
    config_dict = dict(config.model_dump())

    potentials = load_atomic_models_as_potentials(
        config_dict["atomic_models_params"]["path_to_atomic_models"],
        selection_string=config_dict["atomic_models_params"]["atom_selection"],
        loads_b_factors=config_dict["atomic_models_params"]["loads_b_factors"],
    )

    mask = CircularCosineMask(
        coordinate_grid=parameter_file[0]["instrument_config"].coordinate_grid_in_pixels,
        radius=config_dict["mask_radius"],
        rolloff_width=config_dict["mask_rolloff_width"],
    )

    simulate_relion_dataset(
        key=key_stack,
        parameter_file=parameter_file,
        path_to_relion_project=config_dict["path_to_relion_project"],
        images_per_file=config_dict["images_per_file"],
        potentials=potentials,
        potential_integrator=cxs.GaussianMixtureProjection(),
        ensemble_probabilities=config_dict["atomic_models_params"][
            "atomic_models_probabilities"
        ],
        mask=mask,
        noise_snr_range=config_dict["noise_snr"],
        overwrite=config_dict["overwrite"],
        batch_size=config_dict["batch_size_for_generation"],
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
