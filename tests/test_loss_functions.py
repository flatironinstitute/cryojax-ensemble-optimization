import jax
import jax.numpy as jnp
import yaml
from cryojax.dataset import RelionParticleParameterFile, RelionParticleStackDataset

from cryojax_ensemble_optimization.ensemble_optimization import (
    compute_likelihood_matrix as compute_likelihood_matrix,
    compute_neg_log_likelihood as compute_neg_log_likelihood,
    compute_neg_log_likelihood_from_weights as compute_neg_log_likelihood_from_weights,
)
from cryojax_ensemble_optimization.ensemble_optimization._likelihood_optimization import (
    loss_functions as lf,
)
from cryojax_ensemble_optimization.internal import DatasetGeneratorConfig


_likelihood_isotropic_gaussian = lf._likelihood_isotropic_gaussian


def test_likelihood_isotropic_gaussian():
    random_image = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
    noise_variance = 1
    assert jnp.isclose(
        _likelihood_isotropic_gaussian(random_image, random_image, noise_variance), 0.0
    )

    constant_image = jnp.ones((10, 10))
    assert jnp.isnan(
        _likelihood_isotropic_gaussian(constant_image, random_image, noise_variance)
    )

    linear_image = jnp.linspace(0, 1, 100).reshape(10, 10)
    random_scale = jax.random.uniform(jax.random.PRNGKey(1), (1,))
    random_bias = jax.random.uniform(jax.random.PRNGKey(2), (1,))
    assert jnp.isclose(
        _likelihood_isotropic_gaussian(
            linear_image, random_scale * linear_image + random_bias, noise_variance
        ),
        0.0,
    )


def test_compute_likelihood_matrix():
    with open("tests/data/particle_stack/config_data_generation.yaml", "r") as f:
        config_json = yaml.safe_load(f)

    key = jax.random.PRNGKey(config_json["rng_seed"])
    config = dict(DatasetGeneratorConfig(**config_json).model_dump())
    relion_stack = RelionParticleStackDataset(
        RelionParticleParameterFile(
            path_to_starfile=config["path_to_starfile"],
            mode="r",
            loads_envelope=False,
        ),
        path_to_relion_project=config["path_to_relion_project"],
        mode="r",
    )

    n_walkers = 2
    n_atoms = jax.random.randint(key, (1,), 10, 20)[0]
    n_gaussians_per_atom = 5
    ensemble_walkers = jax.random.normal(key, (n_walkers, n_atoms, 3))
    gaussian_amplitudes = jax.random.normal(
        key, (n_walkers, n_atoms, n_gaussians_per_atom)
    )
    gaussian_variances = (
        jax.random.normal(key, (n_walkers, n_atoms, n_gaussians_per_atom)) ** 2
    )
    image_to_walker_log_likelihood_fn = _likelihood_isotropic_gaussian
    per_particle_args_noise_variance = 1.0

    n_particles = 5
    likelihood_matrix = compute_likelihood_matrix(
        ensemble_walkers,
        relion_stack[:n_particles],
        gaussian_amplitudes,
        gaussian_variances,
        image_to_walker_log_likelihood_fn,
        per_particle_args_noise_variance,
    )
    assert likelihood_matrix.shape == (n_particles, n_walkers)
