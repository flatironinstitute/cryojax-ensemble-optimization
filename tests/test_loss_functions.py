import cryojax.simulator as cxs
import jax
import numpy as np
import pytest
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from cryojax_eo.ensemble_optimization import (
    GaussianWhiteLogLikelihoodFn,
    MargGaussianWhiteLogLikelihoodFn,  # done
    compute_optimal_scale_and_offset,  # done
    make_image_model_from_gmm,  # done
)
from cryojax_eo.io import read_walkers_from_pdbs
from cryojax_eo.simulator import DilatedMask


@pytest.fixture
def sample_relion_stack():
    return {
        "parameters": {
            "pose": cxs.EulerAnglePose(),
            "transfer_theory": cxs.ContrastTransferTheory(ctf=cxs.AstigmaticCTF()),
            "image_config": cxs.BasicImageConfig(
                shape=(32, 32), pixel_size=0.2, voltage_in_kilovolts=300.0
            ),
        },
        "images": jax.random.normal(jax.random.key(0), shape=(32, 32)),
    }


@pytest.fixture
def simple_volume_mask(sample_path_to_starfile):
    image_config = RelionParticleParameterFile(sample_path_to_starfile)[0]["image_config"]
    volume_shape = (image_config.shape[0], image_config.shape[0], image_config.shape[0])

    voxel_grid = jax.random.randint(jax.random.key(0), volume_shape, minval=0, maxval=2)

    return DilatedMask(voxel_grid, image_config)


def test_make_image_model_from_gmm(sample_path_to_pdb1, sample_relion_stack):
    walkers, amplitudes, variances = read_walkers_from_pdbs([sample_path_to_pdb1])

    make_image_model_from_gmm(
        walkers[0],
        amplitudes,
        variances,
        sample_relion_stack["parameters"]["image_config"],
        sample_relion_stack["parameters"]["pose"],
        sample_relion_stack["parameters"]["transfer_theory"],
    )
    return


def test_compute_scale_and_offset():
    scale = 2.0
    offset = -3.0

    image = jax.random.normal(jax.random.key(0), shape=(32, 32))
    image_transf = scale * image + offset

    computed_scale, computed_offset = compute_optimal_scale_and_offset(
        image, image_transf
    )

    np.testing.assert_allclose(
        computed_scale, scale, err_msg="Computed scale does not match true scale"
    )
    np.testing.assert_allclose(
        computed_offset, offset, err_msg="Computed offset does not match true offset"
    )
    return


@pytest.mark.parametrize("use_dilated_mask", [True, False])
@pytest.mark.parametrize(
    "image_to_walker_likelihood_fn",
    [MargGaussianWhiteLogLikelihoodFn, GaussianWhiteLogLikelihoodFn],
)
def test_likelihood_fn(
    sample_path_to_pdb1,
    sample_path_to_starfile,
    sample_path_to_relion_project,
    image_to_walker_likelihood_fn,
    use_dilated_mask,
    simple_volume_mask,
):
    walkers, amplitudes, variances = read_walkers_from_pdbs([sample_path_to_pdb1])

    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            path_to_starfile=sample_path_to_starfile,
            options=dict(broadcasts_image_config=False),
        ),
        path_to_relion_project=sample_path_to_relion_project,
        mode="r",
    )

    if use_dilated_mask:
        dilated_mask = simple_volume_mask
    else:
        dilated_mask = None

    img_to_walker_likelihood_fn = image_to_walker_likelihood_fn(
        amplitudes=amplitudes,
        variances=variances,
        data_sign="light-on-dark",
        dilated_mask=dilated_mask,
    )
    stack = relion_dataset[0]
    img_to_walker_likelihood_fn(
        walker=walkers[0],
        image=stack["images"],
        image_config=stack["parameters"]["image_config"],
        pose=stack["parameters"]["pose"],
        transfer_theory=stack["parameters"]["transfer_theory"],
        per_particle_args=1.0,
    )
    return


# def test_likelihood_isotropic_gaussian():
#     random_image = jax.random.normal(jax.random.PRNGKey(0), (10, 10))
#     noise_variance = 1
#     assert jnp.isclose(
#         likelihood_isotropic_gaussian(random_image, random_image, noise_variance), 0.0
#     )

#     constant_image = jnp.ones((10, 10))
#     assert jnp.isnan(
#         likelihood_isotropic_gaussian(constant_image, random_image, noise_variance)
#     )

#     linear_image = jnp.linspace(0, 1, 100).reshape(10, 10)
#     random_scale = jax.random.uniform(jax.random.PRNGKey(1), (1,))
#     random_bias = jax.random.uniform(jax.random.PRNGKey(2), (1,))
#     assert jnp.isclose(
#         likelihood_isotropic_gaussian(
#             linear_image, random_scale * linear_image + random_bias, noise_variance
#         ),
#         0.0,
#     )

"""
def test_compute_likelihood_matrix(
    sample_path_to_starfile, sample_path_to_relion_project
):
    key = jax.random.key(0)
    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            path_to_starfile=sample_path_to_starfile,
        ),
        path_to_relion_project=sample_path_to_relion_project,
        mode="r",
    )

    n_walkers = 2
    n_atoms = int(jax.random.randint(key, (1,), 10, 20)[0])
    n_gaussians_per_atom = 5
    ensemble_walkers = jax.random.normal(key, (n_walkers, n_atoms, 3))
    amplitudes = jax.random.normal(key, (n_walkers, n_atoms, n_gaussians_per_atom))
    variances = jax.random.normal(key, (n_walkers, n_atoms, n_gaussians_per_atom)) ** 2

    # test of likelihood_isotropic_gaussian
    image_to_walker_log_likelihood_fn = likelihood_isotropic_gaussian
    per_particle_args_noise_variance = 1.0

    n_particles = 5
    likelihood_matrix = compute_likelihood_matrix(
        ensemble_walkers,
        relion_dataset[:n_particles],
        amplitudes,
        variances,
        image_to_walker_log_likelihood_fn,
        per_particle_args=per_particle_args_noise_variance,
        constant_args=1.0,
    )
    assert likelihood_matrix.shape == (n_particles, n_walkers)

    # test of _likelihood_sliced_wasserstein
    image_to_walker_log_likelihood_fn = likelihood_sliced_wasserstein
    per_particle_args_n_projections = 7

    n_particles = 5
    likelihood_matrix = compute_likelihood_matrix(
        ensemble_walkers,
        relion_dataset[:n_particles],
        amplitudes,
        variances,
        image_to_walker_log_likelihood_fn,
        constant_args=(per_particle_args_n_projections, 2),
        per_particle_args=(),
    )
    assert likelihood_matrix.shape == (n_particles, n_walkers)
"""
