import jax
import jax.numpy as jnp
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from cryojax_eo.dataset import create_dataloader
from cryojax_eo.ensemble_optimization import (
    ImagesToEnsembleLikelihoodFn,
    IterativeEnsembleLikelihoodOptimizer,
    MargGaussianWhiteLogLikelihoodFn,
    ProjGradDescWeightOptimizer,
)
from cryojax_eo.io import read_atomic_models


def test_iterative_optimizer(
    sample_path_to_pdb1,
    sample_path_to_pdb2,
    sample_path_to_relion_project,
    sample_path_to_starfile,
):
    atomic_models = read_atomic_models(
        [sample_path_to_pdb1, sample_path_to_pdb2], selection_string="not element H"
    )

    positions = jnp.array([model["positions"] for model in atomic_models.values()])
    variances = jnp.array([model["variances"] for model in atomic_models.values()])
    amplitudes = jnp.array([model["amplitudes"] for model in atomic_models.values()])

    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            sample_path_to_starfile, options=dict(broadcasts_image_config=True)
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
        amplitudes=amplitudes[0],
        variances=variances[0],
        image_sign=1.0,
    )

    ensemble_likelihood_fn = ImagesToEnsembleLikelihoodFn(img_to_walker_likelihood_fn)

    optimizer = IterativeEnsembleLikelihoodOptimizer(
        step_size=1.0,
        n_steps=2,
        ensemble_likelihood_fn=ensemble_likelihood_fn,
    )

    weights = jnp.array([0.5, 0.5])
    _, new_positions, new_weights = optimizer(positions, weights, dataloader)

    assert (
        new_positions.shape == positions.shape
    ), "Optimized positions have incorrect shape"
    assert new_weights.shape == weights.shape, "Optimized weights have incorrect shape"

    return


def test_weight_optimizer(
    sample_path_to_pdb1,
    sample_path_to_pdb2,
    sample_path_to_relion_project,
    sample_path_to_starfile,
):
    atomic_models = read_atomic_models(
        [sample_path_to_pdb1, sample_path_to_pdb2], selection_string="not element H"
    )

    positions = jnp.array([model["positions"] for model in atomic_models.values()])
    variances = jnp.array([model["variances"] for model in atomic_models.values()])
    amplitudes = jnp.array([model["amplitudes"] for model in atomic_models.values()])

    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            sample_path_to_starfile, options=dict(broadcasts_image_config=True)
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
        amplitudes=amplitudes[0],
        variances=variances[0],
        image_sign=1.0,
    )

    ensemble_likelihood_fn = ImagesToEnsembleLikelihoodFn(img_to_walker_likelihood_fn)

    optimizer = ProjGradDescWeightOptimizer(
        n_steps=2,
        ensemble_likelihood_fn=ensemble_likelihood_fn,
    )

    weights = jnp.array([0.5, 0.5])
    new_weights = optimizer(positions, weights, dataloader)

    assert new_weights.shape == weights.shape, "Optimized weights have incorrect shape"

    return
