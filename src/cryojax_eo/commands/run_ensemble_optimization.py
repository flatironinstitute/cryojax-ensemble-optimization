#!/usr/bin/env python3
import argparse
import datetime
import logging
import os

import jax
import jax.numpy as jnp
import mdtraj
import mrcfile
import optax
import yaml
from cryojax.dataset import (
    RelionParticleParameterFile,
    RelionParticleStackDataset,
)
from cryojax.io import read_array_from_mrc
from cryojax.ndimage import downsample_to_shape_with_fourier_cropping

import cryojax_eo as cxeo
from cryojax_eo.internal import EnsOptMDConfig
from cryojax_eo.io import read_atomic_models
from cryojax_eo.simulator import DilatedMask
from cryojax_eo.utils import ModelToVolumeAligner


def add_args(parser):
    parser.add_argument(
        "--config", type=str, help="Path to the config (yaml) file", required=True
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
    return


def run_ensemble_optimization_with_md(ensemble_opt_config: EnsOptMDConfig):
    config = dict(ensemble_opt_config.model_dump())

    # Load the initial walkers and reference structure
    atomic_models = read_atomic_models(
        config["path_to_atomic_models"],
        loads_b_factors=config["loads_b_factors"],
    )

    ref_structure = mdtraj.load(
        config["alignment_params"]["path_to_prealigned_atomic_model"]
    )
    ref_structure = ref_structure.center_coordinates()

    atom_list = ref_structure.topology.select(config["atom_selection"])

    initial_walkers = jnp.array(
        [model["atom_positions"] for model in atomic_models.values()]
    )
    variances = jnp.array([model["variances"] for model in atomic_models.values()])[
        :, atom_list
    ]
    amplitudes = jnp.array([model["amplitudes"] for model in atomic_models.values()])[
        :, atom_list
    ]

    # Load experimental data: images, mask, and consensus volume
    stack_dataset = RelionParticleStackDataset(
        RelionParticleParameterFile(
            path_to_starfile=config["data_params"]["path_to_starfile"],
            loads_envelope=config["data_params"]["loads_envelope"],
        ),
        path_to_relion_project=config["data_params"]["path_to_relion_project"],
    )

    key = jax.random.PRNGKey(config["rng_seed"])
    key_data, key_pipeline = jax.random.split(key)

    dataloader = cxeo.data.create_dataloader(
        stack_dataset,
        batch_size=config["likelihood_optimizer_params"]["batch_size"],
        shuffle=True,
        drop_last=False,
        jax_prng_key=key_data,
    )

    if config["data_params"]["path_to_volumetric_mask"] is not None:
        mask = jnp.asarray(
            mrcfile.open(
                config["data_params"]["path_to_volumetric_mask"],
                mode="r",
            ).data
        ).copy()
        dilated_mask = DilatedMask(mask, stack_dataset[0]["parameters"]["image_config"])  # type: ignore

    else:
        dilated_mask = None

    if config["alignment_params"]["path_to_consensus_volume"] is not None:
        volume_for_alignment, voxel_size = read_array_from_mrc(
            config["alignment_params"]["path_to_consensus_volume"],
            loads_grid_spacing=True,
        )

        if config["alignment_params"]["consensus_volume_voxel_size"] is not None:
            voxel_size = config["alignment_params"]["consensus_volume_voxel_size"]

        box_size_ds = int(config["alignment_params"]["downsample_box_size"])

        voxel_size = voxel_size * volume_for_alignment.shape[0] / box_size_ds
        volume_for_alignment = downsample_to_shape_with_fourier_cropping(
            volume_for_alignment, (box_size_ds, box_size_ds, box_size_ds)
        )

        model_aligner = ModelToVolumeAligner(volume_for_alignment, voxel_size)
    else:
        model_aligner = None

    # Construct prior projector
    projector_list = []

    for i in range(initial_walkers.shape[0]):
        projector_list.append(
            cxeo.ensemble_optimization.SteeredMDSimulator(
                path_to_initial_pdb=config["path_to_atomic_models"][i],
                n_steps=config["projector_params"]["n_steps"],
                restrain_atom_list=atom_list,
                parameters_for_md={
                    "platform": config["projector_params"]["platform"],
                    "properties": config["projector_params"]["platform_properties"],
                },
                base_state_file_path=os.path.join(
                    config["path_to_output"], f"states_proj_{i}/state_"
                ),
            )
        )
    md_projector = cxeo.ensemble_optimization.EnsembleSteeredMDSimulator(projector_list)

    # Construct likelihood optimizer
    data_sign = -1.0 if config["data_params"]["data_sign"] == "dark-on-light" else 1.0
    likelihood_fn = cxeo.ensemble_optimization.LikelihoodOptimalWeightsFn(
        amplitudes,
        variances,
        image_to_walker_log_likelihood_fn="iso_gaussian_var_marg",
        loss_fn_constant_args=data_sign,
        dilated_mask=dilated_mask,
        estimates_pose=config["likelihood_optimizer_params"]["estimates_pose"],
    )

    likelihood_optimizer = (
        cxeo.ensemble_optimization.IterativeEnsembleLikelihoodOptimizer(
            step_size=config["likelihood_optimizer_params"]["step_size"],
            n_steps=config["likelihood_optimizer_params"]["n_steps"],
            n_batches_per_step=config["likelihood_optimizer_params"][
                "n_batches_per_step"
            ],
            likelihood_fn=likelihood_fn,
        )
    )

    # Construct the ensemble optimization pipeline
    ensemble_refinement_pipeline = (
        cxeo.ensemble_optimization.EnsembleOptimizationPipeline(
            prior_projector=md_projector,
            likelihood_optimizer=likelihood_optimizer,
            n_steps=config["n_steps"],
            prealigned_structure=ref_structure,
            atom_indices_for_opt=atom_list,
            model_to_volume_aligner=model_aligner,
            runs_postprocessing=True,
        )
    )

    # Running the optimization

    if isinstance(config["projector_params"]["bias_constant_in_kjpermol"], float):
        bias_constant_scheduler = optax.constant_schedule(
            config["projector_params"]["bias_constant_in_kjpermol"]
        )
    elif (
        isinstance(config["projector_params"]["bias_constant_in_kjpermol"], list)
        and len(config["projector_params"]["bias_constant_in_kjpermol"]) == 2
    ):
        bias_constant_scheduler = optax.linear_schedule(
            init_value=config["projector_params"]["bias_constant_in_kjpermol"][0],
            end_value=config["projector_params"]["bias_constant_in_kjpermol"][1],
            transition_steps=config["n_steps"],
        )
    else:
        raise ValueError(
            "bias_constant_in_kjpermol must be a float or a list of two floats."
        )

    init_walkers = jnp.array(initial_walkers.copy())
    init_weights = jnp.array(config["likelihood_optimizer_params"]["init_weights"])

    walkers, weights = ensemble_refinement_pipeline.run(
        key=key_pipeline,
        initial_walkers=init_walkers,
        initial_weights=init_weights,
        dataloader=dataloader,
        bias_constant_scheduler=bias_constant_scheduler,
        output_directory=config["path_to_output"],
        initial_state_for_projector=config["projector_params"]["path_to_initial_states"],
    )

    jnp.savez(
        os.path.join(config["path_to_output"], "final_walkers.npz"),
        walkers=walkers,
        weights=weights,
    )

    return walkers, weights


def main(args):
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
        config = EnsOptMDConfig(**config_dict)

    warnexists(config.path_to_output)
    mkbasedir(config.path_to_output)

    logger = logging.getLogger()
    logger.handlers.clear()

    logger_fname = datetime.datetime.now().strftime("%Y-%m-%d-%H")
    logger_fname = os.path.join(config.path_to_output, logger_fname + ".log")

    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    config_fname = os.path.basename(args.config)
    with open(os.path.join(config.path_to_output, config_fname), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logging.info(
        "A copy of the used config file has been written to {}".format(
            os.path.join(config.path_to_output, config_fname)
        )
    )

    logging.info("Running ensemble optimization...")
    _, _ = run_ensemble_optimization_with_md(config)
    logging.info("Ensemble optimization complete.")

    return


def main_cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(EnsOptMDConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())


if __name__ == "__main__":
    main_cli()
