from typing import Any, Tuple
from functools import partial
import logging

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import equinox as eqx

from cryojax.data import (
    RelionParticleParameters,
    RelionParticleParameterDataset,
    write_simulated_image_stack_from_starfile,
    write_starfile_with_particle_parameters,
)
import cryojax.simulator as cxs
from cryojax.image.operators import FourierGaussian, CircularCosineMask
from cryojax.rotations import SO3

from ._distributions import WhiteGaussianNoise
from ._noise_utils import _compute_noise_variance
from ..data._atomic_model_loaders import _load_models_for_data_generator as load_models


def simulate_relion_dataset(config: dict):
    # Write starfile

    logging.info("Generating Starfile...")
    key = jax.random.PRNGKey(config["rng_seed"])
    key, *subkeys = jax.random.split(key, config["number_of_images"] + 1)
    particle_parameters = _make_particle_parameters(
        jnp.array(subkeys), config
    )

    write_starfile_with_particle_parameters(
        particle_parameters,
        config["path_to_starfile"],
        mrc_batch_size=config["batch_size"],
        overwrite=config["overwrite"],
    )
    logging.info("Starfile generated. Saved to {}".format(config["path_to_starfile"]))

    # load starfile

    logging.info("Loading generated starfile...")
    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=config["path_to_starfile"],
        path_to_relion_project=config["path_to_relion_project"],
        loads_envelope=True,
    )
    logging.info("Starfile loaded.")

    # Load potentials
    logging.info("Generating potentials...")
    potential_integrator = cxs.GaussianMixtureProjection()
    # potential_integrator = cxs.FourierSliceExtraction(interpolation_order=1)
    potentials = []

    atom_positions, struct_info = load_models(config)
    for i in range(len(atom_positions)):
        # tmp_potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
        # tmp_voxel_grid = tmp_potential.as_real_voxel_grid(shape=(config["box_size"], config["box_size"], config["box_size"]), voxel_size=config["pixel_size"])
        # potentials.append(cxs.FourierVoxelGridPotential.from_real_voxel_grid(tmp_voxel_grid, config["pixel_size"], pad_scale=2.0))
        potentials.append(
            cxs.PengAtomicPotential(
                atom_positions[i],
                struct_info["atom_identities"],
                struct_info["b_factors"],
            )
        )
    potentials = tuple(potentials)
    logging.info("Potentials generated.")

    # estimate noise variance
    logging.info("Estimating noise variance...")
    noise_variance, key = _estimate_noise_variance(
        key, parameter_dataset, (potentials, potential_integrator, config)
    )
    logging.info(f"Noise variance estimated. Value: {noise_variance}")

    args = (potentials, potential_integrator, config["weights_models"], noise_variance)
    new_seed = int(jax.random.randint(key, minval=0, maxval=1000000, shape=()))
    logging.info("Generating images with random seed {}".format(new_seed))
    write_simulated_image_stack_from_starfile(
        parameter_dataset,
        _compute_noisy_image,
        args,
        seed=new_seed,
        is_jittable=True,
        batch_size_per_mrc=1000,
        overwrite=config["overwrite"],
    )
    logging.info(
        "Images generated. Saved to {}".format(config["path_to_relion_project"])
    )
    return


def _estimate_noise_variance(
    key: PRNGKeyArray, parameter_dataset: RelionParticleParameterDataset, args: Any
) -> Tuple[Float, PRNGKeyArray]:
    potentials, potential_integrator, config = args

    # define noise mask
    noise_mask = CircularCosineMask(
        coordinate_grid_in_angstroms_or_pixels=parameter_dataset[
            0
        ].instrument_config.coordinate_grid_in_pixels,
        radius_in_angstroms_or_pixels=config["noise_radius_mask"],
        rolloff_width_in_angstroms_or_pixels=1.0,
    )
    n_images_for_est = min(10, len(parameter_dataset))
    key, *subkeys = jax.random.split(jax.random.PRNGKey(0), n_images_for_est + 1)

    noise_var = _compute_noise_variance(
        jnp.array(subkeys),
        parameter_dataset[0:n_images_for_est],
        (potentials, potential_integrator, noise_mask, config),
    ).mean()

    return noise_var, key


@partial(eqx.filter_vmap, in_axes=(0, None))
def _make_particle_parameters(
    key: PRNGKeyArray, config: dict
) -> RelionParticleParameters:
    instrument_config = cxs.InstrumentConfig(
        shape=(config["box_size"], config["box_size"]),
        pixel_size=config["pixel_size"],
        voltage_in_kilovolts=config["voltage_in_kilovolts"],
        pad_scale=config["pad_scale"],
    )
    # Generate random parameters

    # Pose
    # ... instantiate rotations
    key, subkey = jax.random.split(
        key
    )  # split the key to use for the next random number

    rotation = SO3.sample_uniform(subkey)
    key, subkey = jax.random.split(key)  # do this everytime you use a key!!

    # ... now in-plane translation
    offset_x_in_angstroms = jax.random.uniform(
        subkey,
        (1,),
        minval=config["offset_x_in_angstroms"][0],
        maxval=config["offset_x_in_angstroms"][1],
    )
    key, subkey = jax.random.split(key)

    offset_y_in_angstroms = jax.random.uniform(
        subkey,
        (1,),
        minval=config["offset_y_in_angstroms"][0],
        maxval=config["offset_y_in_angstroms"][1],
    )
    key, subkey = jax.random.split(key)

    in_plane_offset_in_angstroms = jnp.concatenate(
        [offset_x_in_angstroms, offset_y_in_angstroms]
    )
    # ... convert 2D in-plane translation to 3D, setting the out-of-plane translation to
    # zero
    offset_in_angstroms = jnp.pad(in_plane_offset_in_angstroms, ((0, 1),))
    # ... build the pose
    pose = cxs.EulerAnglePose.from_rotation_and_translation(
        rotation, offset_in_angstroms
    )

    # CTF Parameters
    # ... defocus
    defocus_in_angstroms = jax.random.uniform(
        subkey,
        (),
        minval=config["defocus_in_angstroms"][0],
        maxval=config["defocus_in_angstroms"][1],
    )
    key, subkey = jax.random.split(key)

    astigmatism_in_angstroms = jax.random.uniform(
        subkey,
        (),
        minval=config["astigmatism_in_angstroms"][0],
        maxval=config["astigmatism_in_angstroms"][1],
    )
    key, subkey = jax.random.split(key)

    astigmatism_angle = jax.random.uniform(
        subkey,
        (),
        minval=config["astigmatism_angle"][0],
        maxval=config["astigmatism_angle"][1],
    )
    key, subkey = jax.random.split(key)

    phase_shift = jax.random.uniform(
        subkey, (), minval=config["phase_shift"][0], maxval=config["phase_shift"][1]
    )
    key, subkey = jax.random.split(key)

    b_factor = jax.random.uniform(
        subkey,
        (),
        minval=config["envelope_bfactor"][0],
        maxval=config["envelope_bfactor"][1],
    )

    # no more random numbers needed

    # now generate your non-random values
    spherical_aberration_in_mm = config["spherical_aberration_in_mm"]
    amplitude_contrast_ratio = config["amplitude_contrast_ratio"]
    ctf_scale_factor = config["ctf_scale_factor"]

    # ... build the CTF
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.ContrastTransferFunction(
            defocus_in_angstroms=defocus_in_angstroms,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
            astigmatism_angle=astigmatism_angle,
            spherical_aberration_in_mm=spherical_aberration_in_mm,
        ),
        amplitude_contrast_ratio=amplitude_contrast_ratio,
        phase_shift=phase_shift,
        envelope=FourierGaussian(b_factor=b_factor, amplitude=ctf_scale_factor),
    )

    relion_particle_parameters = RelionParticleParameters(
        instrument_config=instrument_config,
        pose=pose,
        transfer_theory=transfer_theory,
    )
    return relion_particle_parameters


def _compute_noisy_image(
    key: PRNGKeyArray,
    relion_particle_stack: RelionParticleParameters,
    args: Any,
) -> Float[
    Array,
    "{relion_particle_stack.instrument_config.y_dim} {relion_particle_stack.instrument_config.x_dim}",  # noqa
]:
    potentials, potential_integrator, weights, noise_variance = args

    key, subkey = jax.random.split(key)
    structure_id = jax.random.choice(subkey, weights.shape[0], p=weights)

    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials,
        relion_particle_stack.pose,
        cxs.DiscreteConformationalVariable(structure_id),
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        relion_particle_stack.transfer_theory,
    )

    image_model = cxs.ContrastImageModel(
        relion_particle_stack.instrument_config, scattering_theory
    )

    key, subkey = jax.random.split(key)
    distribution = WhiteGaussianNoise(
        image_model,
        noise_variance=noise_variance,
        normalizes_signal=True,
    )
    return distribution.sample(subkey)
