import argparse
import datetime
import logging
import os

import jax
import jax.numpy as jnp
import mdtraj
import yaml
from cryojax.data import RelionParticleParameterDataset, RelionParticleStackDataset

from ..data import create_dataloader
from ..ensemble_refinement import (
    EnsembleRefinementOpenMMPipeline,
    IterativeEnsembleOptimizer,
    SteeredMolecularDynamicsSimulator,
)
from ..internal import cryojaxERConfig
from ..io import read_atomic_models


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


def construct_projector_list(config, restrain_atom_list):
    projector_list = []

    for i in range(len(config["atomic_models_filenames"])):
        mkbasedir(os.path.join(config["path_to_output"], f"states_proj_{i}/"))
        projector_list.append(
            SteeredMolecularDynamicsSimulator(
                path_to_initial_pdb=config["atomic_models_filenames"][i],
                bias_constant_in_kj_per_mol_angs=config["md_sampler_config"][
                    "bias_constant_in_units"
                ],
                n_steps=config["md_sampler_config"]["n_steps"],
                restrain_atom_list=restrain_atom_list,
                parameters_for_md={
                    "platform": config["md_sampler_config"]["platform"],
                    "properties": config["md_sampler_config"]["platform_properties"],
                },
                base_state_file_path=os.path.join(
                    config["path_to_output"], f"states_proj_{i}/state_"
                ),
            )
        )
    return projector_list


def construct_likelihood_optimizer(config):
    return IterativeEnsembleOptimizer(
        step_size=config["ensemble_optimizer_config"]["step_size"],
        n_steps=config["ensemble_optimizer_config"]["n_steps"],
    )


def construct_ensemble_refinement_pipeline(config):
    """
    Generate the pipeline from the pipeline config
    """

    restrain_atom_list = mdtraj.load(
        config["atomic_models_filenames"][0]
    ).topology.select(config["atom_list_filter"])

    projector_list = construct_projector_list(config, restrain_atom_list)
    likelihood_optimizer = construct_likelihood_optimizer(config)
    ref_structure = mdtraj.load(config["ref_model_fname"])

    ensemble_refinement_pipeline = EnsembleRefinementOpenMMPipeline(
        prior_projectors=projector_list,
        likelihood_optimizer=likelihood_optimizer,
        n_steps=config["n_steps"],
        ref_structure_for_opt=ref_structure,
        atom_indices_for_opt=restrain_atom_list,
        runs_postprocessing=True,
    )
    return ensemble_refinement_pipeline


def load_initial_walkers_and_scattering_params(config):
    atomic_models_scattering_params = read_atomic_models(
        config["atomic_models_filenames"],
        loads_b_factors=config["loads_b_factors"],
    )

    init_walkers = []
    for atomic_model in atomic_models_scattering_params.values():
        init_walkers.append(atomic_model["atom_positions"])

    init_walkers = jnp.concatenate(init_walkers, axis=0)
    gaussian_amplitudes = atomic_models_scattering_params[0]["gaussian_amplitudes"]
    gaussian_variances = atomic_models_scattering_params[0]["gaussian_variances"]
    return init_walkers, gaussian_amplitudes, gaussian_variances


def run_ensemble_refinement(config):
    key = jax.random.key(config["rng_seed"])
    relion_stack_dataset = RelionParticleStackDataset(
        RelionParticleParameterDataset(
            path_to_starfile=config["path_to_starfile"],
            path_to_relion_project=config["path_to_relion_project"],
            loads_envelope=True,
        )
    )
    dataloader = create_dataloader(
        relion_stack_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        jax_prng_key=key,
    )
    ensemble_refinement_pipeline = construct_ensemble_refinement_pipeline(config)

    init_walkers, gaussian_amplitudes, gaussian_variances = (
        load_initial_walkers_and_scattering_params(config)
    )
    walkers, weights = ensemble_refinement_pipeline(
        initial_walkers=init_walkers,
        initial_weights=config["ensemble_optimizer_config"]["init_weights"],
        dataloader=dataloader,
        args_for_likelihood_optimizer=(
            gaussian_amplitudes,
            gaussian_variances,
            None,
        ),
        output_directory=config["path_to_output"],
    )

    jnp.savez(
        os.path.join(config["path_to_output"], "final_walkers_and_weights.npz"),
        walkers=walkers,
        weights=weights,
    )
    return


def main(args):
    nprocs = args.nprocs

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = dict(cryojaxERConfig(**config_dict).model_dump())

    if config["md_sampler_config"]["platform"] == "CPU":
        if config["md_sampler_config"]["platform_properties"]["Threads"] is None:
            config["md_sampler_config"]["platform_properties"]["Threads"] = str(nprocs)

    warnexists(config["path_to_output"])
    mkbasedir(config["path_to_output"])

    logger = logging.getLogger()
    logger.handlers.clear()

    logger_fname = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    logger_fname = os.path.join(config["path_to_output"], logger_fname + ".log")

    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    config_fname = os.path.basename(args.config)
    with open(os.path.join(config["path_to_output"], config_fname), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(config["path_to_output"], config_fname)
        )
    )

    logging.info("Running ensemble refinement...")
    run_ensemble_refinement(config)
    logging.info("Ensemble refinement complete.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(cryojaxERConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())
