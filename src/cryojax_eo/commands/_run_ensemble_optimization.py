#!/usr/bin/env python3
import argparse
import datetime
import logging
import os
from pathlib import Path

import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import mdtraj
import mrcfile
import numpy as np
import optax
import yaml
from cryojax.io import read_array_from_mrc
from cryojax.jax_util import NDArrayLike
from cryojax.ndimage import fourier_crop_to_shape
from cryospax import (
    RelionParticleDataset,
    RelionParticleParameterFile,
)

import cryojax_eo as cxeo
from cryojax_eo.ensemble_optimization import (
    EnsembleOptimizationPipeline,
    EnsembleSteeredMDSimulator,
    ImagesToEnsembleLikelihoodFn,
    IterativeEnsembleLikelihoodOptimizer,
    MargGaussianWhiteLogLikelihoodFn,
    SteeredMDSimulator,
    md_params_config_to_openmm_overrides,
)
from cryojax_eo.internal import EnsOptMDConfig
from cryojax_eo.io import read_walkers_from_pdbs
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
        Warning(f"Warning: {out} already exists. Overwriting.")
    return


def _make_atom_list(atom_selection, topology) -> np.ndarray:
    suffix = Path(atom_selection).suffix
    if suffix in [".txt", ".npy"]:
        atom_list = np.loadtxt(atom_selection, dtype=int)
    else:
        atom_list = topology.select(atom_selection)
    return np.array(atom_list)


def _make_volume_integrator(
    gmm_volume: cxs.GaussianMixtureVolume,
    shape: tuple[int, int],
    pixel_size: NDArrayLike,
    ensemble_opt_config: EnsOptMDConfig,
) -> cxs.GaussianMixtureProjection:
    vol_int_options = ensemble_opt_config.likelihood_optimizer_params[
        "volume_integrator_backend"
    ]
    if vol_int_options["spread_mode"] == "local":
        n_spread = cxs.suggest_n_spread(
            gmm_volume,
            pixel_size=pixel_size,
            cutoff_sigma=vol_int_options["spread_width_in_stds"],
        )
        integrator_shape = tuple([s * 2 for s in shape])
    else:
        integrator_shape = shape
        n_spread = None

    return cxs.GaussianMixtureProjection(
        shape=integrator_shape,
        n_spread=n_spread,
        sampling_mode=vol_int_options["sampling_mode"],
        enable_pallas=vol_int_options["enable_pallas"],
    )


def run_ensemble_optimization_with_md(ensemble_opt_config: EnsOptMDConfig):
    data_params = ensemble_opt_config.data_params
    projector_params = ensemble_opt_config.projector_params
    likelihood_optimizer_params = ensemble_opt_config.likelihood_optimizer_params
    alignment_params = ensemble_opt_config.alignment_params

    # Load the initial walkers and reference structure

    logging.debug("Loading atomic models...")
    initial_walkers, variances, amplitudes = read_walkers_from_pdbs(
        ensemble_opt_config.path_to_atomic_models,
        loads_b_factors=ensemble_opt_config.loads_b_factors,
    )

    ref_structure = mdtraj.load(str(alignment_params["path_to_prealigned_atomic_model"]))
    ref_structure = ref_structure.center_coordinates(mass_weighted=True)

    atom_list = _make_atom_list(
        ensemble_opt_config.atom_selection, ref_structure.topology
    )
    variances = variances[atom_list]
    amplitudes = amplitudes[atom_list]

    logging.debug("Atomic models loaded.")

    logging.debug("Loading experimental data...")
    # Load experimental data: images, mask, and consensus volume
    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            path_to_starfile=data_params["path_to_starfile"],
            options=dict(
                loads_envelope=data_params["loads_envelope"],
            ),
        ),
        path_to_relion_project=data_params["path_to_relion_project"],
    )

    key = jax.random.PRNGKey(ensemble_opt_config.rng_seed)
    key_data, key_pipeline = jax.random.split(key)

    dataloader = cxeo.dataset.create_dataloader(
        relion_dataset,
        batch_size=likelihood_optimizer_params["batch_size"],
        shuffle=True,
        drop_last=False,
        jax_prng_key=key_data,
    )
    logging.debug("Experimental data loaded.")

    if data_params["path_to_volumetric_mask"] is not None:
        logging.debug("Loading volumetric mask...")
        mask = jnp.asarray(
            mrcfile.open(
                data_params["path_to_volumetric_mask"],
                mode="r",
            ).data
        ).copy()
        dilated_mask = DilatedMask(mask)  # type: ignore
        logging.debug("Volumetric mask loaded.")

    else:
        mask = None
        dilated_mask = None

    if alignment_params["path_to_reference_volume"] is not None:
        logging.debug("Loading consensus volume for alignment...")
        volume_for_alignment, voxel_size = read_array_from_mrc(
            alignment_params["path_to_reference_volume"],
            loads_grid_spacing=True,
        )

        if alignment_params["reference_volume_voxel_size"] is not None:
            voxel_size = alignment_params["reference_volume_voxel_size"]

        box_size_ds = int(alignment_params["downsample_box_size"])

        voxel_size = voxel_size * volume_for_alignment.shape[0] / box_size_ds
        volume_for_alignment = fourier_crop_to_shape(
            volume_for_alignment, (box_size_ds, box_size_ds, box_size_ds)
        )

        model_aligner = ModelToVolumeAligner(volume_for_alignment, voxel_size)
        logging.debug("Consensus volume loaded.")
    else:
        model_aligner = None

    # Construct prior projector
    projector_list = []

    parameters_for_md = md_params_config_to_openmm_overrides(
        projector_params["md_params"]
    )
    parameters_for_md["platform"] = projector_params["platform"]
    parameters_for_md["properties"] = projector_params["platform_properties"]

    for i in range(initial_walkers.shape[0]):
        projector_list.append(
            SteeredMDSimulator(
                path_to_initial_pdb=ensemble_opt_config.path_to_atomic_models[i],
                n_steps=projector_params["n_steps"],
                restrain_atom_list=atom_list.tolist(),
                parameters_for_md=parameters_for_md,
                base_state_file_path=os.path.join(
                    ensemble_opt_config.path_to_output, f"states_proj_{i}/state_"
                ),
                # Offset per walker so walkers don't share an identical
                # thermostat random stream. A seed of 0 keeps OpenMM's default
                # behavior of drawing a fresh seed each run (non-reproducible).
                # random_seed=(
                #     ensemble_opt_config.rng_seed + i
                # ),
            )
        )
    md_projector = EnsembleSteeredMDSimulator(projector_list)

    # Construct likelihood optimizer
    tmp_image_config = relion_dataset.parameter_file[0]["image_config"]
    volume_integrator = _make_volume_integrator(
        cxs.GaussianMixtureVolume(
            positions=initial_walkers[0, atom_list],
            amplitudes=amplitudes,
            variances=variances,
        ),
        pixel_size=tmp_image_config.pixel_size,
        ensemble_opt_config=ensemble_opt_config,
        shape=tmp_image_config.shape,
    )

    img_to_walker_log_likelihood_fn = MargGaussianWhiteLogLikelihoodFn(
        amplitudes,
        variances,
        data_sign=data_params["data_sign"],
        dilated_mask=dilated_mask,
        integrator=volume_integrator,
    )
    ensemble_likelihood_fn = ImagesToEnsembleLikelihoodFn(
        img_to_walker_log_likelihood_fn, n_walkers_in_parallel=1, n_images_in_parallel=50
    )
    if likelihood_optimizer_params["estimates_pose"]:
        raise NotImplementedError(
            "Pose estimation inside the MD ensemble"
            " optimization pipeline is not yet implemented."
        )

    likelihood_optimizer = IterativeEnsembleLikelihoodOptimizer(
        step_size=likelihood_optimizer_params["step_size"],
        n_steps=likelihood_optimizer_params["n_steps"],
        n_batches_per_step=likelihood_optimizer_params["n_batches_per_step"],
        ensemble_likelihood_fn=ensemble_likelihood_fn,
        pose_search=None,
    )

    runs_postprocessing = True if initial_walkers.shape[0] > 1 else False

    # Construct the ensemble optimization pipeline
    ensemble_refinement_pipeline = EnsembleOptimizationPipeline(
        prior_projector=md_projector,
        likelihood_optimizer=likelihood_optimizer,
        n_steps=ensemble_opt_config.n_steps,
        prealigned_structure=ref_structure,
        atom_indices_for_opt=jnp.asarray(atom_list, dtype=int),
        model_to_volume_aligner=model_aligner,
        runs_postprocessing=runs_postprocessing,
    )

    # Running the optimization

    bias_constant_in_kjpermol = projector_params["bias_constant_in_kjpermol"]
    if isinstance(bias_constant_in_kjpermol, float):
        bias_constant_scheduler = optax.constant_schedule(bias_constant_in_kjpermol)
    elif (
        isinstance(bias_constant_in_kjpermol, list)
        and len(bias_constant_in_kjpermol) == 2
    ):
        bias_constant_scheduler = optax.linear_schedule(
            init_value=bias_constant_in_kjpermol[0],
            end_value=bias_constant_in_kjpermol[1],
            transition_steps=ensemble_opt_config.n_steps,
        )
    else:
        raise ValueError(
            "bias_constant_in_kjpermol must be a float or a list of two floats."
        )

    initial_weights = jnp.array(likelihood_optimizer_params["initial_weights"])

    walkers, weights = ensemble_refinement_pipeline.run(
        key=key_pipeline,
        initial_walkers=initial_walkers,
        initial_weights=initial_weights,
        dataloader=dataloader,
        bias_constant_scheduler=bias_constant_scheduler,
        output_directory=ensemble_opt_config.path_to_output,
        initial_state_for_projector=projector_params["path_to_initial_states"],
    )

    jnp.savez(
        os.path.join(ensemble_opt_config.path_to_output, "final_ensemble.npz"),
        walkers=walkers,
        weights=weights,
    )
    for i in range(walkers.shape[0]):
        # only the filename, remove dir
        pdb_filename = Path(ensemble_opt_config.path_to_atomic_models[i]).name
        logging.info(f"{pdb_filename}: weight = {weights[i]:.4f}")

    return walkers, weights


def main(args):
    with open(args.config) as f:
        config_dict = yaml.safe_load(f)
        config = EnsOptMDConfig(**config_dict)

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
