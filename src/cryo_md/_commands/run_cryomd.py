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
from .._data.particle_dataloader import load_starfile

warnings.filterwarnings("ignore", module="MDAnalysis")


def add_args(parser):
    parser.add_argument(
        "--config", type=str, help="Path to the config (yaml) file", required=True
    )
    parser.add_argument(
        "--nprocs", type=int, default=1, required=False, help="Number of processors (only if using CPU)"
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

    key_dtype = type(list(pipeline_config.keys())[0])
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
                        models_fname=config["models_fname"],
                        restrain_force_constant=pipeline_step[
                            "mdsampler_force_constant"
                        ],
                        n_steps=pipeline_step["n_steps"],
                        n_models=config["n_models"],
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
                        n_models=config["n_models"],
                        checkpoint_fnames=pipeline_step["checkpoint_fnames"],
                    )
                )

    pipeline = Pipeline(workflow, config)
    return pipeline


def run_pipeline(pipeline, image_stack, config):

    models_fname = config["models_fname"]
    _, struct_info = load_models(config)
    init_universes = []
    for i in range(config["n_models"]):
        init_universes.append(mda.Universe(models_fname[i]))

    ref_universe = mda.Universe(config["ref_model_fname"])

    logging.info("Preparing pipeline for run...")
    pipeline.prepare_for_run_(
        config,
        init_universes,
        struct_info,
        ref_universe,
    )

    pipeline.run(image_stack)
    return


def main(args):

    nprocs = args.nprocs
    config = load_config(args.config)

    for key in config["pipeline"].keys():
        if "mdsampler" == config["pipeline"][key]["type"]:
            if config["pipeline"][key]["platform"] == "CPU":
                if config["pipeline"][key]["platform_properties"]["Threads"] is None:
                    config["pipeline"][key]["platform_properties"]["Threads"] = nprocs
        
    warnexists(config["output_path"])
    mkbasedir(config["output_path"])

    logger_fname = os.path.join(
        config["output_path"], config["experiment_name"] + ".log"
    )
    logger = logging.getLogger()
    logger.handlers.clear()

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
        json.dump(config, f, indent=4)

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(config["output_path"], config_fname)
        )
    )

    starfile_fname = os.path.basename(config["starfile_path"])
    starfile_path = os.path.dirname(config["starfile_path"])
    
    image_stack = load_starfile(starfile_path, starfile_fname, batch_size=config["batch_size"])
    pipeline = generate_pipeline(config)
    run_pipeline(pipeline, image_stack, config)
    
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=help_config_generator(),
    )
    main(add_args(parser).parse_args())
