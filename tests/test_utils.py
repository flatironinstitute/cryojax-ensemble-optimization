import cryojax.simulator as cxs
import jax.numpy as jnp
from cryojax.io import read_array_from_mrc, read_atoms_from_pdb

from cryojax_eo.utils import (
    fit_gmm_model_to_voxel_grid,
    Gaussian3D,
    make_gmm_model_from_atomic_model,
    ModelToVolumeAligner,
)


### GMM Fitting ####
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

    render_fn = cxs.GaussianMixtureRenderFn(target_volume.shape, voxel_size)
    fitted_volume = render_fn(fitted_gmm)
    loss_fitted = jnp.linalg.norm(target_volume - fitted_volume)

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


def test_volume_aligner(sample_path_groel_pdb):
    positions, _ = read_atoms_from_pdb(
        sample_path_groel_pdb, selection_string="name CA", center=True
    )

    render_fn = cxs.GaussianMixtureRenderFn((32, 32, 32), 4.0)
    real_voxel_grid = render_fn(cxs.GaussianMixtureVolume(positions, 1.0, 3.0))

    aligner = ModelToVolumeAligner(real_voxel_grid=real_voxel_grid, voxel_size=4.0)

    rotation = cxs.EulerAnglePose(phi_angle=1.0, theta_angle=3.0, psi_angle=3.0)
    rot_mtx = rotation.rotation.as_matrix()

    rotated_positions = positions @ rot_mtx

    _, solution = aligner.align(rotated_positions, 1.0, 3.0)

    cos_theta = (jnp.trace(rot_mtx @ solution.rotation_matrix) - 1) / 2.0

    assert jnp.isclose(cos_theta, 1.0), "Alignment to volume failed"
    return
