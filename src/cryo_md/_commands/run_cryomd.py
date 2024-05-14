import warnings
import logging
import os
import argparse
import json
import numpy as np
import MDAnalysis as mda
import glob
from natsort import natsorted

from .._data.utils import load_config, help_config_generator
from .._data.pdb import load_models
from .._molecular_dynamics.mdaa_simulator import MDSampler
from .._molecular_dynamics.mdcg_simulator import MDCGSampler
from .._optimization.optimizer import PositionOptimizer
from .._optimization.optimizer import WeightOptimizer
from ..pipeline import Pipeline

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


def generate_pipeline(config):
    """
    Generate the pipeline from the pipeline config
    """
    pipeline_config = config["pipeline"]
    workflow = []

    key_dtype = type(pipeline_config.keys()[0])
    steps_key = np.sort(list(pipeline_config.keys())).astype(key_dtype)

    for key in steps_key:
        pipeline_step = pipeline_config[key]

        if pipeline_step["type"] == "pos_opt":
            workflow.append(
                PositionOptimizer(
                    n_steps=pipeline_step["pos_opt_steps"],
                    step_size=pipeline_step["pos_opt_stepsize"],
                )
            )

        elif pipeline_step["type"] == "weight_opt":
            workflow.append(
                WeightOptimizer(
                    n_steps=pipeline_step["weight_opt_steps"],
                    step_size=pipeline_step["weight_opt_stepsize"],
                )
            )

        elif pipeline_step["type"] == "mdsampler":
            if pipeline_step["mode"] == "all-atom":
                workflow.append(
                    MDSampler(
                        pdb_file=config["models_fname"],
                        restrain_force_constant=pipeline_step[
                            "mdsampler_force_constant"
                        ],
                        n_steps=pipeline_step["n_steps"],
                        n_models=pipeline_step["n_models"],
                        checkpoint_fnames=pipeline_step["checkpoint_fnames"],
                    )
                )

            if pipeline_step["mode"] == "coarse-grained":
                workflow.append(
                    MDCGSampler(
                        models_fname=config["models_fname"],
                        top_file=pipeline_step["top_file"],
                        restrain_force_constant=pipeline_step[
                            "mdsampler_force_constant"
                        ],
                        epsilon_r=pipeline_step["epsilon_r"],
                        n_steps=pipeline_step["n_steps"],
                        n_models=pipeline_step["n_models"],
                        checkpoint_fnames=pipeline_step["checkpoint_fnames"],
                    )
                )

    pipeline = Pipeline(workflow)
    return pipeline


def run_pipeline(pipeline, image_stack, config):
    if "*" in config["models_fname"]:
        models_fname = natsorted(glob.glob(config["models_fname"]))
    else:
        models_fname = [config["models_fname"]]

    _, struct_info = load_models(config)
    init_universes = []
    for i in range(config["n_models"]):
        init_universes.append(mda.Universe(models_fname[i]))

    ref_universe = mda.Universe(config["ref_model_fname"])

    logging.info("Preparing pipeline for run...")
    pipeline.prepare_for_run_(
        config["n_steps"],
        init_universes,
        struct_info,
        config["mode"],
        config["output_path"],
        ref_universe,
    )

    pipeline.run(image_stack)
    return


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
        json.dump(config, f)

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(config["output_path"], config_fname)
        )
    )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=help_config_generator(),
    )
    main(add_args(parser).parse_args())
