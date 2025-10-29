import cryojax.simulator as cxs
import jax.numpy as jnp
from cryojax.io import read_array_from_mrc

from cryojax_eo.utils import (
    fit_gmm_model_to_voxel_grid,
    Gaussian3D,
    make_gmm_model_from_atomic_model,
)


def test_fit_gmm_to_density(sample_path_gmm_model, sample_path_mrc_file):
    gmm_positions = jnp.load(sample_path_gmm_model)["positions"]
    target_volume, voxel_size = read_array_from_mrc(
        sample_path_mrc_file, loads_grid_spacing=True
    )

    initial_gmm = Gaussian3D(
        positions=gmm_positions,
        amplitude=6.0,
        variance=1.0,
        shape=target_volume.shape,
        voxel_size=voxel_size,
        n_gaussians_per_bead=1,
    )
    fitted_gmm = fit_gmm_model_to_voxel_grid(initial_gmm, target_volume)

    loss_initial = jnp.linalg.norm(target_volume - initial_gmm.to_real_voxel_grid())
    loss_fitted = jnp.linalg.norm(
        target_volume - fitted_gmm.to_real_voxel_grid(target_volume.shape, voxel_size)
    )

    assert (
        loss_fitted < loss_initial
    ), "Fitted GMM should have lower loss than initial GMM"
    return


def test_make_gmm_model_from_atomic_model(sample_path_groel_pdb):
    gmm_model = make_gmm_model_from_atomic_model(
        sample_path_groel_pdb,
        box_size=32,
        voxel_size=4.0,
        fit_selection_string="name CA",
    )

    assert gmm_model.positions.ndim == 2, "GMM positions should be a 2D array"
    assert gmm_model.amplitudes.ndim == 2, "GMM amplitudes should be a 2D array"
    assert gmm_model.variances.ndim == 2, "GMM variances should be a 2D array"
    assert isinstance(
        gmm_model, cxs.GaussianMixtureVolume
    ), "GMM model should be a GaussianMixtureVolume"

    return
