#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
import sys

import yaml

from ..dataset import simulate_relion_dataset
from ..internal import DatasetSimulatorConfig


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


def main(args):
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = DatasetSimulatorConfig(**config_dict)

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
    os.system("cp {} {}".format(args.config, os.path.join(project_path, config_fname)))

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(project_path, config_fname)
        )
    )

    logging.info("Simulating particle stack...")
    simulate_relion_dataset(config)
    logging.info("Simulation complete.")
    logging.info("Output written to {}".format(project_path))

    return


def main_cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(DatasetSimulatorConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())


if __name__ == "__main__":
    main_cli()
