import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.rotations import SO3
from cryospax import RelionParticleDataset, RelionParticleParameterFile

from cryojax_eo import HierarchicalSO3GridSearch, compute_correlation_at_optimal_offset
from cryojax_eo._image_to_ensemble_utils._utils import _compose_poses_with_alignment
from cryojax_eo.simulator import DilatedMask


def test_global_SO3_hier_search(
    sample_path_to_pdb1,
    sample_path_to_relion_project,
    sample_path_to_starfile,
):
    gmm_volume = cxs.load_tabulated_volume(
        sample_path_to_pdb1,
        output_type=cxs.GaussianMixtureVolume,
        include_b_factors=True,
        selection_string="not element H",
    )

    relion_dataset = RelionParticleDataset(
        RelionParticleParameterFile(
            sample_path_to_starfile,
        ),
        sample_path_to_relion_project,
    )
    pose_search = HierarchicalSO3GridSearch(base_grid_res=1, n_rounds=5, n_candidates=40)
    stack = relion_dataset[0]
    assert "parameters" in stack

    pose = pose_search(
        gmm_volume,
        jnp.asarray(stack["images"]),
        stack["parameters"]["image_config"],
        stack["parameters"]["transfer_theory"],
        cxs.AutoVolumeProjection(),
    )
    eqx.tree_equal(pose, stack["parameters"]["pose"], rtol=1e-2)

    return


def _smooth_groel_volume(path_to_pdb):
    # Wider gaussians, so that sub-pixel shifts are resolved accurately by the
    # pixel-averaged projection and images can be compared tightly
    volume = cxs.load_tabulated_volume(
        path_to_pdb,
        output_type=cxs.GaussianMixtureVolume,
        tabulation="peng",
        include_b_factors=True,
        selection_string="not element H",
    )
    return cxs.GaussianMixtureVolume(
        volume.positions, volume.amplitudes, volume.variances * 400.0
    )


def _simulate(volume, positions, image_config, pose):
    volume = cxs.GaussianMixtureVolume(positions, volume.amplitudes, volume.variances)
    return cxs.make_image_model(volume, image_config, pose).simulate()


def test_compose_poses_with_alignment(sample_path_groel_pdb):
    """The composed pose gives the same image from the original walker as the
    estimated pose gives from the aligned walker."""
    volume = _smooth_groel_volume(sample_path_groel_pdb)
    image_config = cxs.BasicImageConfig(
        shape=(64, 64), pixel_size=4.0, voltage_in_kilovolts=300.0
    )
    key_rotation, key_shift, key_pose = jax.random.split(jax.random.key(1), 3)
    rotation_matrix = SO3.sample_uniform(key_rotation).as_matrix()
    displacement = jax.random.normal(key_shift, (1, 3)) * 8.0
    aligned_positions = volume.positions @ rotation_matrix.T + displacement

    pose_aligned = cxs.QuaternionPose(
        wxyz=SO3.sample_uniform(key_pose).wxyz,
        offset_x_in_angstroms=5.0,
        offset_y_in_angstroms=-7.0,
    )
    # Add the (n_walkers, n_images) batch dimensions that `estimate_poses` returns
    batched_pose = jax.tree.map(lambda x: x[None, None], pose_aligned)
    composed_pose = _compose_poses_with_alignment(
        batched_pose, rotation_matrix[None], displacement[None]
    )
    composed_pose = jax.tree.map(lambda x: x[0, 0], composed_pose)

    target = _simulate(volume, aligned_positions, image_config, pose_aligned)
    image = _simulate(volume, volume.positions, image_config, composed_pose)
    assert jnp.linalg.norm(image - target) / jnp.linalg.norm(target) < 1e-2


def test_masked_correlation_with_unit_mask_matches_unmasked():
    key_a, key_b = jax.random.split(jax.random.key(0))
    image_shifted = jax.random.normal(key_a, (32, 32))
    image_ref = jax.random.normal(key_b, (32, 32))
    image_config = cxs.BasicImageConfig(
        shape=(32, 32), pixel_size=1.0, voltage_in_kilovolts=300.0
    )
    grid = image_config.get_coordinate_grid(physical=True)

    correlation, shift = compute_correlation_at_optimal_offset(
        image_shifted, image_ref, grid
    )
    masked_correlation, masked_shift = compute_correlation_at_optimal_offset(
        image_shifted, image_ref, grid, mask_ref=jnp.ones((32, 32))
    )
    assert jnp.allclose(masked_correlation, correlation, rtol=1e-6)
    assert jnp.allclose(masked_shift, shift)


def test_masked_global_SO3_hier_search(sample_path_to_pdb1):
    """With a mask covering the particle, the masked search finds a pose whose
    image matches the target. Images are compared rather than rotations, which
    are only defined up to the particle's symmetries."""
    volume = cxs.load_tabulated_volume(
        sample_path_to_pdb1,
        output_type=cxs.GaussianMixtureVolume,
        tabulation="peng",
        include_b_factors=True,
        selection_string="not element H",
    )
    image_config = cxs.BasicImageConfig(
        shape=(32, 32), pixel_size=1.0, voltage_in_kilovolts=300.0
    )
    true_pose = cxs.QuaternionPose(
        wxyz=SO3.sample_uniform(jax.random.key(2)).wxyz,
        offset_x_in_angstroms=2.0,
        offset_y_in_angstroms=-3.0,
    )
    target = _simulate(volume, volume.positions, image_config, true_pose)

    radius = jnp.linalg.norm(
        jnp.stack(jnp.meshgrid(*[jnp.arange(32) - 16.0] * 3, indexing="ij"), -1),
        axis=-1,
    )
    spherical_mask = (radius < 14.0).astype(float)

    pose_search = HierarchicalSO3GridSearch(
        base_grid_res=1,
        n_rounds=3,
        n_candidates=10,
        n_angles_in_parallel=50,
        dilated_mask=DilatedMask(spherical_mask),
    )
    estimated_pose = pose_search(
        volume, target, image_config, None, cxs.AutoVolumeProjection()
    )
    image = _simulate(volume, volume.positions, image_config, estimated_pose)
    correlation = jnp.sum(image * target) / (
        jnp.linalg.norm(image) * jnp.linalg.norm(target)
    )
    assert correlation > 0.95
