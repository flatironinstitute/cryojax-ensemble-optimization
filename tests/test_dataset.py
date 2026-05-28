import os
import pathlib
import shutil

import jax
import jax.numpy as jnp
import pytest
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from cryojax_eo.dataset import (
    create_dataloader,
    make_relion_parameter_file,
    simulate_relion_dataset,
)
from cryojax_eo.internal import DatasetSimulatorConfig


@pytest.fixture
def sample_simulator_config(
    sample_rotations_config,
    sample_path_to_pdb1,
    sample_path_to_pdb2,
):
    path_to_starfile = os.path.join(
        os.path.dirname(__file__), "outputs", "test_starfile.star"
    )
    path_to_relion_project = os.path.join(os.path.dirname(__file__), "outputs")
    return DatasetSimulatorConfig(
        number_of_images=10,
        data_sign="dark-on-light",
        pixel_size=0.8,
        box_size=32,
        pad_scale=1,
        voltage_in_kilovolts=300.0,
        offset_x_in_angstroms=0.0,
        offset_y_in_angstroms=0.0,
        defocus_in_angstroms=[100, 200],
        astigmatism_in_angstroms=[-1.0, 1.0],
        astigmatism_angle_in_degrees=[-1.0, 1.0],
        phase_shift=0.0,
        amplitude_contrast_ratio=0.1,
        spherical_aberration_in_mm=1e-16,
        ctf_scale_factor=1.0,
        envelope_b_factor=0.0,
        noise_snr=0.1,
        mask_radius=32 // 2,
        mask_rolloff_width=1.0,
        rng_seed=0,
        atomic_models_params=dict(
            path_to_atomic_models=[sample_path_to_pdb1, sample_path_to_pdb2],
            atomic_models_probabilities=[0.5, 0.5],
            loads_b_factors=True,
            atom_selection="not element H",
        ),
        path_to_relion_project=pathlib.Path(path_to_relion_project),
        path_to_starfile=pathlib.Path(path_to_starfile),
        images_per_file=10,
        batch_size_for_generation=10,
        overwrite=True,
        rotations=sample_rotations_config,
    )


@pytest.fixture(
    params=[
        {},  # Uniform by default
        {
            "rotation_distribution": "non-uniform",
            "vmf_kappa": 3.0,
            "vmf_mu": [0.0, 0.0, 1.0],
            "vmf_alpha": 0.5,
        },
    ],
    ids=["uniform-rotations", "non-uniform-rotations"],
)
def sample_rotations_config(request):
    return request.param


def test_make_relion_parameter_file(sample_simulator_config):
    parameter_file = make_relion_parameter_file(
        jax.random.key(0), sample_simulator_config
    )

    assert (
        len(parameter_file) == sample_simulator_config.number_of_images
    ), "Number of entries in parameter file different from requested"
    return


def test_simulate_relion_dataset(sample_simulator_config):
    relion_dataset = simulate_relion_dataset(sample_simulator_config)

    assert (
        len(relion_dataset) == sample_simulator_config.number_of_images
    ), "Number of images simulated different from requested"
    shutil.rmtree(relion_dataset.path_to_relion_project)
    return


# parametrize shuffle = True and False, and


@pytest.mark.parametrize("shuffle", [True, False])
def test_create_dataloader(
    shuffle, sample_path_to_starfile, sample_path_to_relion_project
):
    relion_stack_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            sample_path_to_starfile,
        ),
        sample_path_to_relion_project,
    )

    per_particle_args = jnp.arange(0, len(relion_stack_dataset))

    dataloader = create_dataloader(
        relion_stack_dataset,
        batch_size=2,
        jax_prng_key=jax.random.key(0),
        shuffle=shuffle,
        per_particle_args=per_particle_args,
    )

    batch = next(iter(dataloader))

    assert "particle_stack" in batch, "Batch does not contain particle_stack"
    assert "per_particle_args" in batch, "Batch does not contain per_particle_args"
    assert batch["particle_stack"]["images"].shape[0] == 2, "Batch size is incorrect"
    assert (
        batch["per_particle_args"].shape[0] == 2
    ), "Batch size of per_particle_args is incorrect"
    return
