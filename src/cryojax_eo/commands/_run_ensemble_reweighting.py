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
from cryojax.jax_util import filter_bmap
from cryospax import (
    RelionParticleDataset,
    RelionParticleParameterFile,
)
from jaxtyping import Array, Float
from tqdm import tqdm

import cryojax_eo as cxeo
from cryojax_eo.ensemble_optimization import (
    HierarchicalSO3GridSearch,
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
    parser.add_argument(
        "--from-likelihoods",
        nargs="?",
        const=True,
        default=None,
        metavar="PATH",
        help=(
            "Skip likelihood computation and re-optimize weights from a pre-computed "
            "log_likelihood_matrix.npz. If PATH is omitted, looks for the file in "
            "the output directory specified by the config. "
            "Useful for tuning max_iter or tol without recomputing likelihoods."
        ),
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
    # return likelihood_iso_gaussian_marg(
    return likelihood_iso_gaussian_marg(
        volume=volume,
        image=relion_stack["images"],
        image_config=relion_stack["parameters"]["image_config"],
        pose=relion_stack["parameters"]["pose"],
        transfer_theory=relion_stack["parameters"]["transfer_theory"],
        integrator=cxs.AutoVolumeProjection(),
        dilated_mask=dilated_mask,
        image_sign=image_sign,
        per_particle_args=None,
    )


@eqx.filter_vmap(in_axes=(None, 0, None, eqx.if_array(0), None))
def _estimate_pose(
    volume: cxs.AbstractVolumeRepresentation,
    image: Float[Array, "y_dim x_dim"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    pose_search: HierarchicalSO3GridSearch,
) -> cxs.QuaternionPose:
    return pose_search(volume, image, image_config, transfer_theory)


@eqx.filter_jit
def estimate_poses(
    volume: cxs.AbstractVolumeRepresentation,
    images: Float[Array, "y_dim x_dim"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    pose_search: HierarchicalSO3GridSearch,
    *,
    n_images_in_parallel: int,
) -> cxs.QuaternionPose:
    return filter_bmap(
        lambda x: _estimate_pose(volume, x[0], image_config, x[1], pose_search),
        xs=(images, transfer_theory),
        batch_size=n_images_in_parallel,
    )


@eqx.filter_vmap
def _convert_quat_to_euler(quat_pose: cxs.QuaternionPose) -> cxs.EulerAnglePose:
    return cxs.EulerAnglePose.from_rotation_and_translation(
        quat_pose.rotation, quat_pose.offset_in_angstroms
    )


def compute_likelihoods_for_structural_file(
    path_to_structure: str | Path,
    relion_dataset: RelionParticleDataset,
    selection_string: str,
    dilated_mask: DilatedMask | None,
    data_sign: Literal["dark-on-light", "light-on-dark"],
    n_images_in_parallel: int,
    max_volume_repr_resolution: float | None,
    estimates_poses: bool,
    path_to_outputdir: str,
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
    pose_search = HierarchicalSO3GridSearch(
        base_grid_res=1, n_rounds=5, n_candidates=40, n_angles_in_parallel=10
    )
    image_config = relion_dataset.parameter_file[0]["image_config"]
    mask = cxim.CircularCosineMask(
        image_config.get_coordinate_grid(physical=False),
        radius=image_config.shape[0] // 2,
        rolloff_width=1.0,
    )

    if estimates_poses:
        max_n_batches = np.ceil(len(relion_dataset) / n_images_in_parallel).astype(int)
        path_to_starfile = os.path.join(
            path_to_outputdir, Path(path_to_structure).stem + "_starfile.star"
        )
        new_parameter_file = RelionParticleParameterFile(
            path_to_starfile=path_to_starfile,
            mode="w",
            exist_ok=True,
            max_optics_groups=max_n_batches + 10,
        )
    for batch in tqdm(dataloader, desc="batches", leave=False):
        if estimates_poses:
            poses = estimate_poses(
                volume=voxel_volume,
                images=batch["particle_stack"]["images"] * mask.get()[None, ...],
                image_config=batch["particle_stack"]["parameters"]["image_config"],
                transfer_theory=batch["particle_stack"]["parameters"]["transfer_theory"],
                pose_search=pose_search,
                n_images_in_parallel=10,
            )
            batch["particle_stack"]["parameters"]["pose"] = _convert_quat_to_euler(poses)
            new_parameter_file.append(batch["particle_stack"]["parameters"])

        batch_likelihoods = _compute_likelihoods_fn(
            voxel_volume,
            batch["particle_stack"],
            dilated_mask,
            image_sign,
        )
        likelihoods.append(batch_likelihoods)

    if estimates_poses:
        new_parameter_file.particle_data["rlnImageName"] = (
            relion_dataset.parameter_file.particle_data["rlnImageName"]
        )
        new_parameter_file.save(overwrite=True)

    return jnp.concatenate(likelihoods)


def run_ensemble_reweighting_from_scratch(
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
            ),
        ),
        path_to_relion_project=config["data_params"]["path_to_relion_project"],
    )
    logging.debug("Experimental data loaded.")

    logging.debug("Computing whitening filter...")
    image_config = relion_dataset.parameter_file[0]["image_config"]
    mask = cxim.CircularCosineMask(
        image_config.get_coordinate_grid(physical=False),
        radius=image_config.shape[0] // 2,
        rolloff_width=1.0,
    )
    logging.debug("Whitening filter computed.")

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
    likelihoods_dict = {}
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
            estimates_poses=config["estimates_poses"],
            path_to_outputdir=config["path_to_output_dir"],
        )
        # make the key in the dictionary the filename without the path and extension
        dict_key = Path(file).stem
        likelihoods_dict[dict_key] = likelihoods
        likelihood_matrix[:, i] = np.asarray(likelihoods)

    weights = optimize_weights(
        log_likelihood_matrix=jnp.array(likelihood_matrix),
        max_iter=config["max_iter"],
        tol=config["tol"],
    )
    weight_dict = {}

    logging.info("Final weights:")
    for key, w in zip(likelihoods_dict.keys(), weights):
        weight_dict[key] = float(w)
        logging.info(f"  {key}: {w:.4f}")
        print(f"  {key}: {w:.4f}")

    # save the weights as a yaml file
    with open(
        os.path.join(config["path_to_output_dir"], "optimized_weights.yaml"), "w"
    ) as f:
        yaml.dump(weight_dict, f)

    np.savez(
        os.path.join(config["path_to_output_dir"], "log_likelihood_matrix.npz"),
        **likelihoods_dict,
    )

    return weights


def run_ensemble_reweighting_from_likelihoods(
    ensemble_opt_config: ReweightingConfig,
    path_to_npz: str | None = None,
) -> Float[Array, " n_models"]:
    config = dict(ensemble_opt_config.model_dump())

    if path_to_npz is None:
        path_to_npz = os.path.join(
            config["path_to_output_dir"], "log_likelihood_matrix.npz"
        )
    if not os.path.exists(path_to_npz):
        raise FileNotFoundError(
            f"No pre-computed likelihoods found at {path_to_npz}. "
            "Run without --from-likelihoods first to compute them."
        )

    logging.info(f"Loading pre-computed likelihoods from {path_to_npz}...")
    data = np.load(path_to_npz)

    keys = [Path(f).stem for f in config["path_to_structural_files"]]
    try:
        likelihood_matrix = np.column_stack([data[k] for k in keys])
    except KeyError as e:
        raise KeyError(
            f"Key {e} not found in {path_to_npz}. "
            "Ensure path_to_structural_files matches the original run."
        ) from e

    weights = optimize_weights(
        log_likelihood_matrix=jnp.array(likelihood_matrix),
        max_iter=config["max_iter"],
        tol=config["tol"],
    )

    weight_dict = {}
    logging.info("Final weights:")
    for key, w in zip(keys, weights):
        weight_dict[key] = float(w)
        logging.info(f"  {key}: {w:.4f}")
        print(f"  {key}: {w:.4f}")

    with open(
        os.path.join(config["path_to_output_dir"], "optimized_weights.yaml"), "w"
    ) as f:
        yaml.dump(weight_dict, f)

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

    if args.from_likelihoods is not None:
        path = None if args.from_likelihoods is True else args.from_likelihoods
        logging.info("Re-computing weights from pre-computed likelihoods...")
        run_ensemble_reweighting_from_likelihoods(config, path_to_npz=path)
        logging.info("Weight optimization complete.")
    else:
        logging.info("Running ensemble reweighting from scratch...")
        run_ensemble_reweighting_from_scratch(config)
        logging.info("Ensemble reweighting complete.")

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
