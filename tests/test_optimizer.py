import jax
import jax.numpy as jnp
import mdtraj
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from cryojax_eo import (
    ImagesToEnsembleLikelihoodFn,
    IterativeEnsembleLikelihoodOptimizer,
    MargGaussianWhiteLogLikelihoodFn,
    MultGradWeightOptimizer,
)
from cryojax_eo.dataset import create_dataloader
from cryojax_eo.io import read_walkers_from_pdbs


def test_iterative_optimizer(
    sample_path_to_pdb1,
    sample_path_to_pdb2,
    sample_path_to_relion_project,
    sample_path_to_starfile,
):
    atom_indices = mdtraj.load(sample_path_to_pdb1).topology.select("not element H")
    walkers, amplitudes, variances = read_walkers_from_pdbs(
        [sample_path_to_pdb1, sample_path_to_pdb2]
    )
    walkers = walkers[:, atom_indices]
    variances = variances[atom_indices]
    amplitudes = amplitudes[atom_indices]

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
    )

    ensemble_likelihood_fn = ImagesToEnsembleLikelihoodFn(img_to_walker_likelihood_fn)

    optimizer = IterativeEnsembleLikelihoodOptimizer(
        step_size=1.0,
        n_steps=2,
        ensemble_likelihood_fn=ensemble_likelihood_fn,
        n_batches_per_step=2,
    )

    weights = jnp.array([0.5, 0.5])
    _, new_positions, new_weights = optimizer(walkers, weights, dataloader)

    assert new_positions.shape == walkers.shape, (
        "Optimized positions have incorrect shape"
    )
    assert new_weights.shape == weights.shape, "Optimized weights have incorrect shape"

    return


def test_weight_optimizer(
    sample_path_to_pdb1,
    sample_path_to_pdb2,
    sample_path_to_relion_project,
    sample_path_to_starfile,
):
    walkers, amplitudes, variances = read_walkers_from_pdbs(
        [sample_path_to_pdb1, sample_path_to_pdb2]
    )

    atom_indices = mdtraj.load(sample_path_to_pdb1).topology.select("not element H")
    walkers = walkers[:, atom_indices]
    variances = variances[atom_indices]
    amplitudes = amplitudes[atom_indices]

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
    )

    ensemble_likelihood_fn = ImagesToEnsembleLikelihoodFn(img_to_walker_likelihood_fn)

    optimizer = MultGradWeightOptimizer(
        ensemble_likelihood_fn=ensemble_likelihood_fn,
    )

    weights = jnp.array([0.5, 0.5])
    new_weights = optimizer(walkers, dataloader)

    assert new_weights.shape == weights.shape, "Optimized weights have incorrect shape"

    return
