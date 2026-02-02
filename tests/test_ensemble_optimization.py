import os
import shutil

import jax
import jax.numpy as jnp
import mdtraj
import pytest
from cryojax.dataset import RelionParticleDataset, RelionParticleParameterFile
from cryojax.simulator import GaussianMixtureRenderFn
from optax import constant_schedule

from cryojax_eo.dataset import create_dataloader
from cryojax_eo.ensemble_optimization import (
    EnsembleOptimizationPipeline,
    EnsembleSteeredMDSimulator,
    IterativeEnsembleLikelihoodOptimizer,
    LikelihoodOptimalWeightsFn,
    SteeredMDSimulator,
)
from cryojax_eo.io import load_gmm_volume_parametrization, read_atomic_models
from cryojax_eo.utils import ModelToVolumeAligner


@pytest.fixture
def sample_model_aligner(sample_path_to_pdb1):
    gmm_volume = load_gmm_volume_parametrization(
        [sample_path_to_pdb1],
        selection_string="not element H",
    )[0]

    voxel_size = 0.2 * 128 / 32
    render_fn = GaussianMixtureRenderFn((32, 32, 32), voxel_size)
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

    atomic_models = read_atomic_models(
        [sample_path_to_pdb1, sample_path_to_pdb2], selection_string="all"
    )

    walkers = jnp.array([model["positions"] for model in atomic_models.values()])
    variances = jnp.array([model["variances"] for model in atomic_models.values()])[
        :, atom_list
    ]
    amplitudes = jnp.array([model["amplitudes"] for model in atomic_models.values()])[
        :, atom_list
    ]

    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(sample_path_to_starfile),
        sample_path_to_relion_project,
    )

    dataloader = create_dataloader(
        relion_dataset,
        batch_size=2,
        shuffle=True,
        per_particle_args=None,
        jax_prng_key=jax.random.key(0),
    )

    likelihood_fn = LikelihoodOptimalWeightsFn(
        amplitudes=amplitudes,
        variances=variances,
        image_to_walker_log_likelihood_fn="iso_gaussian_var_marg",
    )

    optimizer = IterativeEnsembleLikelihoodOptimizer(
        step_size=1.0,
        n_steps=2,
        n_batches_per_step=2,
        likelihood_fn=likelihood_fn,
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
