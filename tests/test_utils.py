import cryojax.simulator as cxs
import jax.numpy as jnp
from cryojax.io import read_atoms_from_pdb

from cryojax_eo.utils import (
    ModelToVolumeAligner,
)


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
