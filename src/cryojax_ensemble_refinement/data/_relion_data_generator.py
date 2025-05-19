import logging
import os
from functools import partial
from typing import List, Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.data import (
    RelionParticleParameterDataset,
    RelionParticleParameters,
    write_simulated_image_stack_from_starfile,
    write_starfile_with_particle_parameters,
)
from cryojax.image.operators import AbstractMask, FourierGaussian
from cryojax.rotations import SO3
from jaxtyping import Array, Float, PRNGKeyArray

from ..internal._config_validators import DatasetGeneratorConfig
from ..simulator._image_rendering import render_image_with_white_gaussian_noise


def generate_relion_parameter_dataset(
    key: PRNGKeyArray, config: DatasetGeneratorConfig
) -> RelionParticleParameterDataset:
    """

    This functions writes to disk a starfile containing the
    generated particle parameters. The starfile is saved to the path
    specified in the configuration. The function also returns a
    `cryojax.data.RelionParticleParameterDataset` object that can be used to read the
    starfile and access the particle parameters.

    **Arguments:**
        key: JAX random key for generating random numbers.
        config: Configuration object containing the parameters for generating the dataset.
    **Returns:**
        parameter_dataset: A RelionParticleParameterDataset object containing
        the generated particle parameters.
    """

    config_dict = dict(config.model_dump())
    logging.info("Generating Starfile...")
    key, *subkeys = jax.random.split(key, config_dict["number_of_images"] + 1)
    particle_parameters = _make_particle_parameters(jnp.array(subkeys), config_dict)

    write_starfile_with_particle_parameters(
        particle_parameters,
        config_dict["path_to_starfile"],
        mrc_batch_size=config_dict["batch_size"],
        overwrite=config_dict["overwrite"],
    )
    logging.info(
        "Starfile generated. Saved to {}".format(config_dict["path_to_starfile"])
    )

    parameter_dataset = RelionParticleParameterDataset(
        path_to_starfile=config_dict["path_to_starfile"],
        path_to_relion_project=config_dict["path_to_relion_project"],
    )

    return parameter_dataset


def simulate_relion_dataset(
    key: PRNGKeyArray,
    parameter_dataset: RelionParticleParameterDataset,
    potentials: Tuple[cxs.AbstractPotentialRepresentation],
    potential_integrator: cxs.AbstractPotentialIntegrator,
    ensemble_probabilities: Float[Array, " n_potentials"],
    mask: AbstractMask,
    noise_snr_range: List[Float],
    *,
    overwrite: bool = False,
):
    assert len(potentials) == len(ensemble_probabilities), (
        "The number of potentials must be equal to the number of ensemble probabilities."
        f" {len(potentials)} != {len(ensemble_probabilities)}"
    )

    # generate random keys
    key_snr, key_noise, key_potentials = jax.random.split(key, 3)

    # Make sure the ensemble probabilities sum to 1
    ensemble_probabilities = jnp.array(ensemble_probabilities)
    ensemble_probabilities /= jnp.sum(ensemble_probabilities)

    # Generate parameters for each image
    snr_per_image = jax.random.uniform(
        key_snr,
        (len(parameter_dataset),),
        minval=noise_snr_range[0],
        maxval=noise_snr_range[1],
    )

    keys_per_image = jax.random.split(key_noise, len(parameter_dataset))

    ensemble_indices_per_image = jax.random.choice(
        key_potentials,
        a=len(ensemble_probabilities),
        shape=(len(parameter_dataset),),
        p=ensemble_probabilities,
        replace=True,
    )

    # Write metadata to disk
    path_to_relion_project = parameter_dataset.path_to_relion_project
    jnp.savez(
        os.path.join(path_to_relion_project, "metadata.npz"),
        snr_per_image=snr_per_image,
        ensemble_indices_per_image=ensemble_indices_per_image,
    )
    logging.info("Metadata: noise variance and ensemble indices saved to:")
    logging.info(f"  {os.path.join(path_to_relion_project, 'metadata.npz')}")

    # Bundle arguments and write images
    constant_args = (potentials, potential_integrator, mask)
    per_particle_args = (
        keys_per_image,
        ensemble_indices_per_image,
        snr_per_image,
    )
    write_simulated_image_stack_from_starfile(
        param_dataset=parameter_dataset,
        compute_image_fn=render_image_with_white_gaussian_noise,
        constant_args=constant_args,
        per_particle_args=per_particle_args,
        is_jittable=True,
        batch_size_per_mrc=1000,
        overwrite=overwrite,
    )
    logging.info("Images generated. Saved to {}".format(path_to_relion_project))
    logging.info("Simulated dataset generation complete.")
    return


@partial(eqx.filter_vmap, in_axes=(0, None))
def _make_particle_parameters(
    key: PRNGKeyArray, config: dict
) -> RelionParticleParameters:
    """
    WARNING: this function assumes the `config` has been validated
    by `cryojax_ensemble_refinement.internal.GeneratorConfig`.
    Skipping this step could lead to unexpected behavior.
    """
    instrument_config = cxs.InstrumentConfig(
        shape=(config["box_size"], config["box_size"]),
        pixel_size=config["pixel_size"],
        voltage_in_kilovolts=config["voltage_in_kilovolts"],
        pad_scale=config["pad_scale"],
    )
    # Generate random parameters

    # Pose
    # ... instantiate rotations
    key, subkey = jax.random.split(key)  # split the key to use for the next random number

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
    pose = cxs.EulerAnglePose.from_rotation_and_translation(rotation, offset_in_angstroms)

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
        minval=config["astigmatism_angle_in_degrees"][0],
        maxval=config["astigmatism_angle_in_degrees"][1],
    )
    key, subkey = jax.random.split(key)

    phase_shift = jax.random.uniform(
        subkey, (), minval=config["phase_shift"][0], maxval=config["phase_shift"][1]
    )
    key, subkey = jax.random.split(key)

    b_factor = jax.random.uniform(
        subkey,
        (),
        minval=config["envelope_b_factor"][0],
        maxval=config["envelope_b_factor"][1],
    )
    # no more random numbers needed

    # now generate your non-random values
    spherical_aberration_in_mm = config["spherical_aberration_in_mm"]
    amplitude_contrast_ratio = config["amplitude_contrast_ratio"]
    ctf_scale_factor = config["ctf_scale_factor"]

    b_factor = jnp.clip(b_factor, 1e-16, None)
    envelope = FourierGaussian(b_factor=b_factor, amplitude=ctf_scale_factor)

    # ... build the CTF
    transfer_theory = cxs.ContrastTransferTheory(
        ctf=cxs.AberratedAstigmaticCTF(
            defocus_in_angstroms=defocus_in_angstroms,
            astigmatism_in_angstroms=astigmatism_in_angstroms,
            astigmatism_angle=astigmatism_angle,
            spherical_aberration_in_mm=spherical_aberration_in_mm,
        ),
        amplitude_contrast_ratio=amplitude_contrast_ratio,
        phase_shift=phase_shift,
        envelope=envelope,
    )

    relion_particle_parameters = RelionParticleParameters(
        instrument_config=instrument_config,
        pose=pose,
        transfer_theory=transfer_theory,
    )
    return relion_particle_parameters
