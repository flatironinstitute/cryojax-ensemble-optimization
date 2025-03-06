import warnings
import logging
import os
import argparse
import json
import yaml

from cryojax.data import RelionParticleDataset, RelionParticleMetadata

from ..data._atomic_model_loaders import _load_models_for_optimizer
from ..data._config_readers.optimizer_config_reader import OptimizationConfig
from ..molecular_dynamics.mdaa_simulator import MDSimulatorRMSDConstraint

# from .._molecular_dynamics.mdcg_simulator import MDCGSampler
from ..optimization.optimizers import EnsembleOptimizer
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

    structural_info = _load_models_for_optimizer(config)

    md_sampler = MDSimulatorRMSDConstraint(
        models_fname=config["models_fnames"],
        restrain_force_constant=config["md_sampler_config"]["mdsampler_force_constant"],
        n_steps=config["md_sampler_config"]["n_steps"],
        n_models=config["max_n_models"],
        checkpoint_fnames=config["checkpoints_fnames"],
        platform=config["md_sampler_config"]["platform"],
        properties=config["md_sampler_config"]["platform_properties"],
    )

    ensemble_optimizer = EnsembleOptimizer(
        step_size=config["ensemble_optimizer_config"]["step_size"],
        batch_size=config["ensemble_optimizer_config"]["batch_size"],
        dataset=dataset,
        init_weights=config["ensemble_optimizer_config"]["init_weights"],
        structural_info=structural_info,
        noise_variance=config["ensemble_optimizer_config"]["noise_variance"],
        n_steps=config["ensemble_optimizer_config"]["n_steps"],
    )

    pipeline = Pipeline(
        experiment_name=config["experiment_name"],
        ensemble_optimizer=ensemble_optimizer,
        md_sampler=md_sampler,
        init_models_path=config["models_fnames"],
        ref_model_path=config["ref_model_fname"],
        atom_list_filter=config["atom_list_filter"],
    )
    return pipeline


def main(args):
    nprocs = args.nprocs

    with open(args.config, "r") as f:
        config_json = json.load(f)
        config = dict(OptimizationConfig(**config_json).model_dump())

    if config["md_sampler_config"]["platform"] == "CPU":
        if config["md_sampler_config"]["platform_properties"]["Threads"] is None:
            config["md_sampler_config"]["platform_properties"]["Threads"] = str(nprocs)

    warnexists(config["output_path"])
    mkbasedir(config["output_path"])
    mkbasedir("tmp")

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

    metadata = RelionParticleMetadata(
        path_to_starfile=config["path_to_starfile"],
        path_to_relion_project=config["path_to_relion_project"],
        get_envelope_function=True,
    )
    dataset = RelionParticleDataset(metadata)

    pipeline = generate_pipeline(config, dataset)
    pipeline.run(n_steps=config["n_steps"], output_path=config["output_path"])

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(OptimizationConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())
