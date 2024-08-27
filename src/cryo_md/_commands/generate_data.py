import warnings
import logging
import os
import argparse
import json

from .._data.utils import load_config, help_config_generator
from .._data.pdb import load_models
from .._simulator.starfile_generator import simulate_stack

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


def main(args):
    config = load_config(args.config)
    warnexists(config["output_path"])
    mkbasedir(config["output_path"])

    # make copy of config to output_path

    logger = logging.getLogger()
    logger_fname = os.path.join(
        config["output_path"], config["experiment_name"] + ".log"
    )
    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    config_fname = os.path.basename(args.config)
    with open(os.path.join(config["output_path"], config_fname), "w") as f:
        json.dump(config, f, indent=4, sort_keys=True)

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(config["output_path"], config_fname)
        )
    )

    models, struct_info = load_models(config)

    logging.info("Simulating particle stack...")
    simulate_stack(
        root_path=config["output_path"],
        starfile_fname=config["starfile_fname"],
        models=models,
        struct_info=struct_info,
        images_per_model=config["particles_per_model"],
        config=config,
        batch_size=config["batch_size"],
        dtype=float,
        seed=config["noise_seed"],
    )
    logging.info("Simulation complete.")
    logging.info("Output written to {}".format(config["output_path"]))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=help_config_generator(),
    )
    main(add_args(parser).parse_args())
