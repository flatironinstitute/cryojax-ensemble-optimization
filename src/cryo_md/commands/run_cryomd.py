import warnings
import logging
import os
import argparse
import json
import yaml
import numpy as np
import MDAnalysis as mda
import jax

from cryojax.data import RelionDataset

from ..data._atomic_model_loaders import _load_models_for_optimizer
from ..data._config_readers.optimizer_config_reader import OptimizationConfig
from ..molecular_dynamics.mdaa_simulator import MDSimulatorRMSDConstraint

# from .._molecular_dynamics.mdcg_simulator import MDCGSampler
from ..optimization.optimizers import PositionOptimizer
from ..optimization.optimizers import WeightOptimizer
from ..pipeline import Pipeline


warnings.filterwarnings("ignore", module="MDAnalysis")


def add_args(parser):
    parser.add_argument(
        "--config", type=str, help="Path to the config (yaml) file", required=True
    )
    parser.add_argument(
        "--nprocs",
        type=int,
        default=1,
        required=False,
        help="Number of processors (only if using CPU)",
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


def generate_pipeline(config, dataset):
    """
    Generate the pipeline from the pipeline config
    """
    pipeline_config = config["pipeline"]
    workflow = []

    key_dtype = type(list(pipeline_config.keys())[0])
    steps_key = np.sort(list(pipeline_config.keys())).astype(key_dtype)

    key = jax.random.PRNGKey(config["rng_seed"])

    for step_key in steps_key:
        pipeline_step = pipeline_config[step_key]

        if pipeline_step["step_type"] == "pos_opt":
            key, subkey = jax.random.split(key)
            workflow.append(
                PositionOptimizer(
                    rng_key=subkey,
                    n_steps=pipeline_step["pos_opt_steps"],
                    step_size=pipeline_step["pos_opt_stepsize"],
                    batch_size=config["batch_size"],
                    dataset=dataset,
                )
            )

        elif pipeline_step["step_type"] == "weight_opt":
            key, subkey = jax.random.split(key)
            workflow.append(
                WeightOptimizer(
                    rng_key=subkey,
                    n_steps=pipeline_step["weight_opt_steps"],
                    step_size=pipeline_step["weight_opt_stepsize"],
                    batch_size=config["batch_size"],
                    dataset=dataset,
                )
            )

        elif pipeline_step["step_type"] == "mdsampler":
            if pipeline_step["mode"] == "all-atom":
                workflow.append(
                    MDSimulatorRMSDConstraint(
                        models_fname=config["models_fname"],
                        restrain_force_constant=pipeline_step[
                            "mdsampler_force_constant"
                        ],
                        n_steps=pipeline_step["n_steps"],
                        n_models=config["max_n_models"],
                        checkpoint_fnames=pipeline_step["checkpoint_fnames"],
                    )
                )

            if pipeline_step["mode"] == "coarse-grained":
                raise NotImplementedError
                # workflow.append(
                #     MDCGSampler(
                #         models_fname=config["models_fname"],
                #         top_file=pipeline_step["top_file"],
                #         restrain_force_constant=pipeline_step[
                #             "mdsampler_force_constant"
                #         ],
                #         epsilon_r=pipeline_step["epsilon_r"],
                #         n_steps=pipeline_step["n_steps"],
                #         n_models=config["n_models"],
                #         checkpoint_fnames=pipeline_step["checkpoint_fnames"],
                #     )
                # )

    pipeline = Pipeline(workflow, config)
    return pipeline


def run_pipeline(pipeline, config):
    struct_info = _load_models_for_optimizer(config)
    init_universes = []
    for i in range(config["max_n_models"]):
        init_universes.append(mda.Universe(config["models_fname"][i]))

    logging.info(
        f"Initial models loaded from {config['models_fname'][:config['max_n_models']]}"
    )

    ref_universe = mda.Universe(config["ref_model_fname"])

    logging.info("Preparing pipeline for run...")
    pipeline.prepare_for_run_(
        config=config,
        init_universes=init_universes,
        struct_info=struct_info,
        ref_universe=ref_universe,
    )

    pipeline.run()
    return


def main(args):
    nprocs = args.nprocs

    with open(args.config, "r") as f:
        config_json = json.load(f)
        config = dict(OptimizationConfig(**config_json).model_dump())

    for key in config["pipeline"].keys():
        if "mdsampler" == config["pipeline"][key]["step_type"]:
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
        json.dump(config_json, f, indent=4, sort_keys=True)

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(config["output_path"], config_fname)
        )
    )

    dataset = RelionDataset(
        path_to_starfile=config["path_to_starfile"],
        path_to_relion_project=config["path_to_relion_project"],
        get_image_stack=True,
        get_envelope_function=True,
    )

    pipeline = generate_pipeline(config, dataset)
    run_pipeline(pipeline, config)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(OptimizationConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())
