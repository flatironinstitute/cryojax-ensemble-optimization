#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
from pathlib import Path
from typing import Literal

import cryojax.ndimage as cxim
import cryojax.simulator as cxs
import equinox as eqx
import jax.numpy as jnp
import mrcfile
import numpy as np
import yaml
from cryojax.io import read_array_from_mrc
from cryospax import (
    RelionParticleDataset,
    RelionParticleParameterFile,
)
from jaxtyping import Array, Float
from tqdm import tqdm

import cryojax_eo as cxeo
from cryojax_eo.ensemble_optimization import (
    likelihood_iso_gaussian_marg,
    optimize_weights,
)
from cryojax_eo.internal import ReweightingConfig
from cryojax_eo.simulator import DilatedMask


RELION_DATASET_IN_AXES = dict(
    images=eqx.if_array(0),
    parameters=dict(
        image_config=None, transfer_theory=eqx.if_array(0), pose=eqx.if_array(0)
    ),
)


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


@eqx.filter_jit
def _gmm_volume_to_voxel_grid(
    gmm_volume: cxs.GaussianMixtureVolume, image_config: cxs.BasicImageConfig
) -> Float[Array, " z_dim y_dim x_dim"]:
    box_size = image_config.shape[0]
    render_fn = cxs.GaussianMixtureRenderFn(
        (box_size, box_size, box_size), image_config.pixel_size
    )
    return render_fn(gmm_volume)


@eqx.filter_jit
@eqx.filter_vmap(in_axes=(None, RELION_DATASET_IN_AXES, None, None))
def _compute_likelihoods_fn(volume, relion_stack, dilated_mask, image_sign):
    return likelihood_iso_gaussian_marg(
        volume=volume,
        image=relion_stack["images"],
        image_config=relion_stack["parameters"]["image_config"],
        pose=relion_stack["parameters"]["pose"],
        transfer_theory=relion_stack["parameters"]["transfer_theory"],
        dilated_mask=dilated_mask,
        image_sign=image_sign,
        per_particle_args=None,
    )


def compute_likelihoods_for_structural_file(
    path_to_structure: str | Path,
    relion_dataset: RelionParticleDataset,
    selection_string: str,
    dilated_mask: DilatedMask | None,
    data_sign: Literal["dark-on-light", "light-on-dark"],
    n_images_in_parallel: int,
    max_volume_repr_resolution: float | None,
) -> Float[Array, " n_images"]:
    image_sign = -1.0 if data_sign == "dark-on-light" else 1.0
    image_config = relion_dataset.parameter_file[0]["image_config"]
    if Path(path_to_structure).suffix in [".pdb", ".cif"]:
        gmm_volume = cxs.load_tabulated_volume(
            path_to_structure,
            output_type=cxs.GaussianMixtureVolume,
            tabulation="peng",
            include_b_factors=True,
            selection_string=selection_string,
            # pdb_options=dict(center=False),
        )
        voxel_grid = _gmm_volume_to_voxel_grid(gmm_volume, image_config)

    elif Path(path_to_structure).suffix in [".mrc"]:
        voxel_grid = read_array_from_mrc(path_to_structure, loads_grid_spacing=False)

    else:
        raise NotImplementedError(
            f"Structural file format {Path(path_to_structure).suffix} not supported."
        )

    if max_volume_repr_resolution is not None:
        box_size = image_config.shape[0]
        nyquist_freq = 1.0 / (2.0 * image_config.pixel_size)
        cutoff_freq = 1.0 / max_volume_repr_resolution
        frequency_cutoff_fraction = cutoff_freq / nyquist_freq

        lowpass_filter = cxim.LowpassFilter(
            frequency_grid_in_angstroms_or_pixels=cxim.make_frequency_grid(
                (box_size, box_size, box_size), image_config.pixel_size
            ),
            grid_spacing=image_config.pixel_size,
            frequency_cutoff_fraction=frequency_cutoff_fraction,
        )
        voxel_grid = cxim.irfftn(lowpass_filter(cxim.rfftn(voxel_grid)))

    voxel_volume = cxs.FourierVoxelGridVolume.from_real_voxel_grid(voxel_grid)

    likelihoods = []
    dataloader = cxeo.dataset.create_dataloader(
        relion_dataset,
        batch_size=n_images_in_parallel,
        shuffle=False,
    )
    for batch in dataloader:
        batch_likelihoods = _compute_likelihoods_fn(
            voxel_volume, batch["particle_stack"], dilated_mask, image_sign
        )
        likelihoods.append(batch_likelihoods)

    return jnp.concatenate(likelihoods)


def run_ensemble_reweighting(
    ensemble_opt_config: ReweightingConfig,
) -> Float[Array, " n_models"]:
    config = dict(ensemble_opt_config.model_dump())

    logging.debug("Loading experimental data...")
    # Load experimental data: images, mask, and consensus volume
    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            path_to_starfile=config["data_params"]["path_to_starfile"],
            options=dict(
                loads_envelope=config["data_params"]["loads_envelope"],
                broadcasts_image_config=False,
            ),
        ),
        path_to_relion_project=config["data_params"]["path_to_relion_project"],
    )
    logging.debug("Experimental data loaded.")

    if config["data_params"]["path_to_volumetric_mask"] is not None:
        logging.debug("Loading volumetric mask...")
        mask = jnp.asarray(
            mrcfile.open(
                config["data_params"]["path_to_volumetric_mask"],
                mode="r",
            ).data
        ).copy()
        dilated_mask = DilatedMask(mask)  # type: ignore
        logging.debug("Volumetric mask loaded.")

    else:
        dilated_mask = None

    # if config["likelihood_optimizer_params"]["estimates_pose"]:
    #     raise NotImplementedError(
    #         "Pose estimation inside the MD ensemble"
    #         " optimization pipeline is not yet implemented."
    #     )

    # Running the optimization

    likelihood_matrix = np.zeros(
        (len(relion_dataset), len(config["path_to_structural_files"]))
    )
    progress_bar = tqdm(
        range(len(config["path_to_structural_files"])), desc="Computing likelihoods"
    )
    for i in progress_bar:
        file = config["path_to_structural_files"][i]
        logging.info(f"Computing likelihoods for {file}...")
        likelihoods = compute_likelihoods_for_structural_file(
            path_to_structure=file,
            relion_dataset=relion_dataset,
            selection_string=config["atom_selection"],
            dilated_mask=dilated_mask,
            n_images_in_parallel=config["n_images_in_parallel"],
            data_sign=config["data_params"]["data_sign"],
            max_volume_repr_resolution=config["max_volume_repr_resolution"],
        )
        likelihood_matrix[:, i] = np.asarray(likelihoods)

    weights = optimize_weights(
        log_likelihood_matrix=jnp.array(likelihood_matrix),
        max_iter=config["max_iter"],
        tol=config["tol"],
    )
    weight_dict = {}

    logging.info("Final weights:")
    for i, file in enumerate(config["path_to_structural_files"]):
        weight_dict[file] = float(weights[i])
        logging.info(f"  Weight for {file}: {weights[i]:.4f}")
        print(f"  Weight for {file}: {weights[i]:.4f}")

    # save the weights as a yaml file
    with open(
        os.path.join(config["path_to_output_dir"], "optimized_weights.yaml"), "w"
    ) as f:
        yaml.dump(weight_dict, f)

    np.save(
        os.path.join(config["path_to_output_dir"], "log_likelihood_matrix.npy"),
        likelihood_matrix,
    )

    return weights


def main(args):
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
        config = ReweightingConfig(**config_dict)

    warnexists(config.path_to_output_dir)
    mkbasedir(config.path_to_output_dir)

    logger = logging.getLogger()
    logger.handlers.clear()

    logger_fname = datetime.datetime.now().strftime("%Y-%m-%d")
    logger_fname = os.path.join(config.path_to_output_dir, logger_fname + ".log")

    fhandler = logging.FileHandler(filename=logger_fname, mode="a")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)

    config_fname = os.path.basename(args.config)
    with open(os.path.join(config.path_to_output_dir, config_fname), "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    logging.info(
        f"A copy of the used config file has been written "
        f"to {os.path.join(config.path_to_output_dir, config_fname)}"
    )

    logging.info("Running ensemble optimization...")
    run_ensemble_reweighting(config)
    logging.info("Ensemble optimization complete.")

    return


def main_cli():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=yaml.dump(ReweightingConfig.model_json_schema(), indent=4),
    )
    main(add_args(parser).parse_args())


if __name__ == "__main__":
    main_cli()
