#!/usr/bin/env python3
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
    EnsembleRefinementPipeline,
    EnsembleSteeredMDSimulator,
    IterativeEnsembleOptimizer,
    SteeredMDSimulator,
)
from ..internal import cryojaxERConfig
from ..io import read_atomic_models
from ..utils import get_atom_indices_from_pdb


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


def construct_md_projector(config, restrain_atom_list):
    projector_list = []

    for i in range(len(config["path_to_atomic_models"])):
        mkbasedir(os.path.join(config["path_to_output"], f"states_proj_{i}/"))
        projector_list.append(
            SteeredMDSimulator(
                path_to_initial_pdb=config["path_to_atomic_models"][i],
                bias_constant_in_kj_per_mol_angs=config["projector_params"][
                    "bias_constant_in_units"
                ],
                n_steps=config["projector_params"]["n_steps"],
                restrain_atom_list=restrain_atom_list,
                parameters_for_md={
                    "platform": config["projector_params"]["platform"],
                    "properties": config["projector_params"]["platform_properties"],
                },
                base_state_file_path=os.path.join(
                    config["path_to_output"], f"states_proj_{i}/state_"
                ),
            )
        )
    return EnsembleSteeredMDSimulator(
        projector_list,
    )


def construct_likelihood_optimizer(config, gaussian_amplitudes, gaussian_variances):
    return IterativeEnsembleOptimizer(
        step_size=config["likelihood_optimizer_params"]["step_size"],
        n_steps=config["likelihood_optimizer_params"]["n_steps"],
        gaussian_amplitudes=gaussian_amplitudes,
        gaussian_variances=gaussian_variances,
        image_to_walker_log_likelihood_fn=config["likelihood_optimizer_params"][
            "image_to_walker_log_likelihood_fn"
        ],
    )


def construct_ensemble_refinement_pipeline(
    config, gaussian_amplitudes, gaussian_variances
):
    """
    Generate the pipeline from the pipeline config
    """

    restrain_atom_list = get_atom_indices_from_pdb(
        select=config["atom_selection"],
        pdb_file=config["path_to_reference_model"],
    )

    projector_list = construct_md_projector(config, restrain_atom_list)
    likelihood_optimizer = construct_likelihood_optimizer(
        config, gaussian_amplitudes, gaussian_variances
    )
    ref_structure = mdtraj.load(config["path_to_reference_model"])

    ensemble_refinement_pipeline = EnsembleRefinementPipeline(
        prior_projector=projector_list,
        likelihood_optimizer=likelihood_optimizer,
        n_steps=config["n_steps"],
        ref_structure_for_alignment=ref_structure,
        atom_indices_for_opt=restrain_atom_list,
        runs_postprocessing=True,
    )
    return ensemble_refinement_pipeline


def load_initial_walkers_and_scattering_params(config):
    atomic_models = read_atomic_models(
        config["path_to_atomic_models"],
        loads_b_factors=config["loads_b_factors"],
    )

    restrain_atom_list = get_atom_indices_from_pdb(
        select=config["atom_selection"],
        pdb_file=config["path_to_reference_model"],
    )

    walkers = jnp.array([model["atom_positions"] for model in atomic_models.values()])
    gaussian_variances = jnp.array(
        [model["gaussian_variances"] for model in atomic_models.values()]
    )[:, restrain_atom_list]
    gaussian_amplitudes = jnp.array(
        [model["gaussian_amplitudes"] for model in atomic_models.values()]
    )[:, restrain_atom_list]

    return walkers, gaussian_amplitudes, gaussian_variances


def run_ensemble_refinement(config):
    key = jax.random.key(config["rng_seed"])
    key_dataloader, key_pipeline = jax.random.split(key)
    relion_stack_dataset = RelionParticleStackDataset(
        RelionParticleParameterDataset(
            path_to_starfile=config["path_to_starfile"],
            path_to_relion_project=config["path_to_relion_project"],
            loads_envelope=config["loads_envelope"],
        )
    )
    dataloader = create_dataloader(
        relion_stack_dataset,
        batch_size=config["likelihood_optimizer_params"]["batch_size"],
        shuffle=True,
        jax_prng_key=key_dataloader,
    )
    init_walkers, gaussian_amplitudes, gaussian_variances = (
        load_initial_walkers_and_scattering_params(config)
    )
    ensemble_refinement_pipeline = construct_ensemble_refinement_pipeline(
        config, gaussian_amplitudes, gaussian_variances
    )

    walkers, weights = ensemble_refinement_pipeline.run(
        key=key_pipeline,
        initial_walkers=init_walkers,
        initial_weights=config["likelihood_optimizer_params"]["init_weights"],
        dataloader=dataloader,
        output_directory=config["path_to_output"],
        initial_state_for_projector=config["projector_params"]["path_to_initial_states"],
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

    if config["projector_params"]["platform"] == "CPU":
        if config["projector_params"]["platform_properties"]["Threads"] is None:
            config["projector_params"]["platform_properties"]["Threads"] = str(nprocs)

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
