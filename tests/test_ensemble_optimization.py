import os
import shutil

import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
import mdtraj
import pytest
from cryospax import RelionParticleDataset, RelionParticleParameterFile
from optax import constant_schedule

from cryojax_eo import (
    EnsembleOptimizationPipeline,
    EnsembleSteeredMDSimulator,
    ImagesToEnsembleLikelihoodFn,
    IterativeEnsembleLikelihoodOptimizer,
    MargGaussianWhiteLogLikelihoodFn,
    SteeredMDSimulator,
)
from cryojax_eo.dataset import create_dataloader
from cryojax_eo.io import read_walkers_from_pdbs
from cryojax_eo.utils import ModelToVolumeAligner


pytest.importorskip(
    "openmm",
    reason="OpenMM is an optional dependency required for the ensemble optimization "
    "pipeline",
)


@pytest.fixture
def sample_model_aligner(sample_path_to_pdb1):
    gmm_volume = cxs.load_tabulated_volume(
        sample_path_to_pdb1,
        output_type=cxs.GaussianMixtureVolume,
        selection_string="not element H",
        include_b_factors=True,
    )

    voxel_size = 0.2 * 128 / 32
    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), voxel_size)
    real_voxel_grid = render_fn(gmm_volume)
    return ModelToVolumeAligner(real_voxel_grid, voxel_size)


def make_steered_md_simulator(path_to_pdb):
    model = mdtraj.load(path_to_pdb)
    atom_list = model.topology.select("not element H")

    return SteeredMDSimulator(
        path_to_pdb,
        n_steps=10,
        restrain_atom_list=atom_list,
        parameters_for_md={"platform": "CPU", "properties": {"Threads": "4"}},
        base_state_file_path=os.path.join(
            os.path.dirname(__file__), "outputs/md_states", "state_it"
        ),
    )


def test_ensemble_optimization_optimizer(
    sample_path_to_pdb1,
    sample_path_to_pdb2,
    sample_path_to_relion_project,
    sample_path_to_starfile,
    sample_model_aligner,
):
    prealigned_structure = mdtraj.load(sample_path_to_pdb1).center_coordinates()
    atom_list = prealigned_structure.topology.select("not element H")

    walkers, amplitudes, variances = read_walkers_from_pdbs(
        [sample_path_to_pdb1, sample_path_to_pdb2]
    )
    variances = variances[atom_list]
    amplitudes = amplitudes[atom_list]

    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            sample_path_to_starfile,
        ),
        sample_path_to_relion_project,
    )

    dataloader = create_dataloader(
        relion_dataset,
        batch_size=2,
        shuffle=True,
        per_particle_args=None,
        jax_prng_key=jax.random.key(0),
    )

    img_to_walker_likelihood_fn = MargGaussianWhiteLogLikelihoodFn(
        amplitudes=amplitudes,
        variances=variances,
        data_sign="light-on-dark",
        dilated_mask=None,
    )
    ensemble_likelihood_fn = ImagesToEnsembleLikelihoodFn(img_to_walker_likelihood_fn)

    optimizer = IterativeEnsembleLikelihoodOptimizer(
        step_size=1.0,
        n_steps=2,
        n_batches_per_step=2,
        ensemble_likelihood_fn=ensemble_likelihood_fn,
        pose_search=None,
    )

    projector = EnsembleSteeredMDSimulator(
        [
            make_steered_md_simulator(sample_path_to_pdb1),
            make_steered_md_simulator(sample_path_to_pdb2),
        ]
    )

    pipeline = EnsembleOptimizationPipeline(
        projector,
        optimizer,
        n_steps=2,
        prealigned_structure=prealigned_structure,
        atom_indices_for_opt=atom_list,
        model_to_volume_aligner=sample_model_aligner,
        runs_postprocessing=True,
    )

    weights = jnp.array([0.5, 0.5])
    output_directory = os.path.join(
        os.path.dirname(__file__), "outputs/ensemble_optimization_test"
    )
    os.makedirs(output_directory, exist_ok=True)

    pipeline.run(
        jax.random.key(0),
        walkers,
        weights,
        dataloader,
        constant_schedule(1.0e3),
        output_directory=output_directory,
    )
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "outputs/"))

    return


def test_run_ensemble_optimization_from_config(
    sample_path_to_pdb1,
    sample_path_to_pdb2,
    sample_path_to_relion_project,
    sample_path_to_starfile,
    tmp_path,
):
    import argparse

    import yaml

    from cryojax_eo.commands._run_ensemble_optimization import main

    output_directory = str(tmp_path / "script_output")

    config = {
        "path_to_atomic_models": [sample_path_to_pdb1, sample_path_to_pdb2],
        "path_to_output": output_directory,
        "atom_selection": "not element H",
        "loads_b_factors": False,
        "n_steps": 2,
        "rng_seed": 0,
        "data_params": {
            "path_to_starfile": sample_path_to_starfile,
            "path_to_relion_project": sample_path_to_relion_project,
            "loads_envelope": False,
            "data_sign": "light-on-dark",
        },
        "projector_params": {
            "n_steps": 10,
            "bias_constant_in_kjpermol": 1000.0,
            "platform": "CPU",
        },
        "likelihood_optimizer_params": {
            "n_steps": 2,
            "step_size": 1.0,
            "n_batches_per_step": 2,
            "batch_size": 2,
            "volume_integrator_backend": {
                "enable_pallas": False,
                "spread_mode": "exact",
                "sampling_mode": "average",
            },
        },
        "alignment_params": {
            "path_to_prealigned_atomic_model": sample_path_to_pdb1,
        },
    }

    config_path = str(tmp_path / "test_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    main(argparse.Namespace(config=config_path))

    assert os.path.exists(os.path.join(output_directory, "final_ensemble.npz"))
