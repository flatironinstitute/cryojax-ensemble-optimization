#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import mdtraj
import numpy as np
import optax
import yaml
from cryojax.io import read_array_from_mrc
from cryojax.ndimage import fourier_crop_to_shape
from jaxtyping import Array, Float, Int

from cryojax_eo.ensemble_optimization import (
    SteeredMDSimulator,
    md_params_config_to_openmm_overrides,
)
from cryojax_eo.flexible_fitting import (
    AbstractModelToVolumeLossFn,
    AdamWalkerFlexibleFitting,
    FlexibleFittingPipeline,
    ModelToVolumeCorrelationLossFn,
    ModelToVolumeWeightedMSELossFn,
    SteepestDescWalkerFlexibleFitting,
)
from cryojax_eo.internal import FlexibleFittingConfig
from cryojax_eo.io import read_walkers_from_pdbs
from cryojax_eo.utils import EarlyStopping, ModelToVolumeAligner


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
        Warning(f"Warning: {out} already exists. Overwriting.")
    return


def _make_atom_list(atom_selection, topology) -> np.ndarray:
    suffix = Path(atom_selection).suffix
    if suffix in [".txt", ".npy"]:
        atom_list = np.loadtxt(atom_selection, dtype=int)
    else:
        atom_list = topology.select(atom_selection)
    return np.array(atom_list)


def _construct_model_to_volume_loss_fn(
    amplitudes: Float[Array, "n_atoms n_gaussians_per_atom"],
    variances: Float[Array, "n_atoms n_gaussians_per_atom"],
    voxel_size_ff: Float,
    box_size_ff: Int,
    vol_mask: Float[Array, "dim_z dim_y dim_x"] | None,
    config: dict,
):
    loss_kwargs: dict[str, Any] = dict(
        amplitudes=amplitudes,
        variances=variances,
        voxel_size=voxel_size_ff,
        volume_shape=(box_size_ff, box_size_ff, box_size_ff),
        vol_mask=vol_mask,
        batch_size_for_z_planes=config["walker_optimizer_params"][
            "batch_size_for_z_planes"
        ],
        n_batches_of_atoms=config["walker_optimizer_params"]["n_batches_of_atoms"],
    )
    if config["reference_volume_params"].get("path_to_weights") is not None:
        path_to_weights = config["reference_volume_params"]["path_to_weights"]
        loss_kwargs["weights"] = jnp.asarray(read_array_from_mrc(path_to_weights))
        return ModelToVolumeWeightedMSELossFn(**loss_kwargs)
    else:
        return ModelToVolumeCorrelationLossFn(**loss_kwargs)


def _construct_walker_optimizer(
    config: dict, model_to_vol_loss_fn: AbstractModelToVolumeLossFn
):
    optimizer_kwargs = dict(
        n_steps=config["walker_optimizer_params"]["n_steps"],
        step_size=config["walker_optimizer_params"]["step_size"],
        model_to_vol_loss_fn=model_to_vol_loss_fn,
    )
    if config["walker_optimizer_params"]["type"] == "steepest_desc":
        return SteepestDescWalkerFlexibleFitting(**optimizer_kwargs)
    elif config["walker_optimizer_params"]["type"] == "adam":
        return AdamWalkerFlexibleFitting(**optimizer_kwargs)
    else:
        raise ValueError(
            f"Invalid walker optimizer type: {config['walker_optimizer_params']['type']}"
        )


def run_flexible_fitting(flexible_fitting_config: FlexibleFittingConfig):
    config = dict(flexible_fitting_config.model_dump())

    # Load the initial walkers and reference structure

    logging.debug("Loading atomic models...")
    initial_walker, variances, amplitudes = read_walkers_from_pdbs(
        [config["path_to_atomic_model"]],
        loads_b_factors=config["loads_b_factors"],
    )

    ref_structure = mdtraj.load(str(config["path_to_prealigned_atomic_model"]))
    ref_structure = ref_structure.center_coordinates(mass_weighted=True)

    atom_list = _make_atom_list(config["atom_selection"], ref_structure.topology)
    initial_walker = initial_walker[0]
    variances = variances[atom_list]
    amplitudes = amplitudes[atom_list]

    logging.debug("Atomic model loaded.")

    logging.debug("Loading reference volume...")
    reference_volume, voxel_size = read_array_from_mrc(
        config["reference_volume_params"]["path_to_reference_volume"],
        loads_grid_spacing=True,
    )
    reference_volume = jnp.asarray(reference_volume)

    if config["reference_volume_params"]["reference_volume_voxel_size"] is not None:
        voxel_size = config["reference_volume_params"]["reference_volume_voxel_size"]

    box_size_align = int(config["reference_volume_params"]["rigid_alignment_box_size"])
    voxel_size_ds = voxel_size * reference_volume.shape[0] / box_size_align

    model_aligner = ModelToVolumeAligner(
        fourier_crop_to_shape(
            reference_volume, (box_size_align, box_size_align, box_size_align)
        ),
        voxel_size=voxel_size_ds,
    )

    box_size_ff = int(config["reference_volume_params"]["flexible_fitting_box_size"])
    voxel_size_ff = voxel_size * reference_volume.shape[0] / box_size_ff
    reference_volume = fourier_crop_to_shape(
        reference_volume, (box_size_ff, box_size_ff, box_size_ff)
    )

    if config["reference_volume_params"]["path_to_volumetric_mask"] is not None:
        logging.debug("Loading volumetric mask...")
        vol_mask = read_array_from_mrc(
            config["reference_volume_params"]["path_to_volumetric_mask"],
            loads_grid_spacing=False,
        )
        vol_mask = fourier_crop_to_shape(
            jnp.asarray(vol_mask), (box_size_ff, box_size_ff, box_size_ff)
        )
        logging.debug("Volumetric mask loaded.")

    else:
        vol_mask = None

    logging.debug("Reference volume loaded.")

    # Construct prior projector
    parameters_for_md = md_params_config_to_openmm_overrides(
        config["projector_params"]["md_params"]
    )
    parameters_for_md["platform"] = config["projector_params"]["platform"]
    parameters_for_md["properties"] = config["projector_params"]["platform_properties"]

    prior_projector = SteeredMDSimulator(
        path_to_initial_pdb=config["path_to_atomic_model"],
        n_steps=config["projector_params"]["n_steps"],
        restrain_atom_list=atom_list.tolist(),
        parameters_for_md=parameters_for_md,
        base_state_file_path=os.path.join(config["path_to_output"], "states_proj/state_"),
        # A seed of 0 keeps OpenMM's default behavior of drawing a fresh seed
        # each run (non-reproducible).
        random_seed=config["rng_seed"] if config["rng_seed"] != 0 else None,
    )

    # Construct likelihood optimizer
    model_to_vol_loss_fn = _construct_model_to_volume_loss_fn(
        amplitudes,
        variances,
        voxel_size_ff,
        box_size_ff,
        vol_mask,
        config,
    )

    walker_optimizer = _construct_walker_optimizer(
        config=config,
        model_to_vol_loss_fn=model_to_vol_loss_fn,
    )

    early_stopping = (
        EarlyStopping(
            patience=config["early_stopping"]["patience"],
            rtol=config["early_stopping"]["rtol"],
            atol=config["early_stopping"]["atol"],
        )
        if config.get("early_stopping") is not None
        else None
    )

    # Construct the ensemble optimization pipeline
    flexible_fitting_pipeline = FlexibleFittingPipeline(
        prior_projector=prior_projector,
        walker_optimizer=walker_optimizer,
        n_steps=config["n_steps"],
        prealigned_structure=ref_structure,
        atom_indices_for_opt=atom_list,
        model_to_volume_aligner=model_aligner,
        early_stopping=early_stopping,
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

    final_walker, md_state = flexible_fitting_pipeline.run(
        initial_walker=initial_walker,
        reference_volume=reference_volume,
        bias_constant_scheduler=bias_constant_scheduler,
        output_directory=config["path_to_output"],
        initial_state_for_projector=config["projector_params"]["path_to_initial_state"],
    )

    jnp.save(
        os.path.join(config["path_to_output"], "final_walker.npy"),
        final_walker,
    )

    return final_walker, md_state


def main(args):
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
        config = FlexibleFittingConfig(**config_dict)

    warnexists(config.path_to_output)
    mkbasedir(config.path_to_output)

    logger = logging.getLogger()
    logger.handlers.clear()

    logger_fname = datetime.datetime.now().strftime("%Y-%m-%d")
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
        f"A copy of the used config file has been written "
        f"to {os.path.join(config.path_to_output, config_fname)}"
    )

    logging.info("Running ensemble optimization...")
    run_flexible_fitting(config)
    logging.info("Ensemble optimization complete.")

    return


def main_cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(FlexibleFittingConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())


if __name__ == "__main__":
    main_cli()
