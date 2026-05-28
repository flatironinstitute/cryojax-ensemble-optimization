import cryojax.simulator as cxs
import jax
import jax.numpy as jnp
from cryojax.io import read_array_from_mrc
from cryojax.ndimage import CircularCosineMask

from cryojax_eo.simulator import DilatedMask, simulate_image_with_white_gaussian_noise


def test_dilated_mask_projection(sample_path_mrc_file):
    voxel_grid, voxel_size = read_array_from_mrc(
        sample_path_mrc_file, loads_grid_spacing=True
    )

    mask = jnp.where(jnp.abs(voxel_grid) > 0.001, 1.0, 0.0)

    image_config = cxs.BasicImageConfig(
        shape=(32, 32), pixel_size=voxel_size, voltage_in_kilovolts=300.0
    )
    dilated_mask = DilatedMask(mask)

    proj_mask = dilated_mask.project(cxs.EulerAnglePose(), image_config)

    assert jnp.abs(proj_mask - mask.sum(0) / mask.sum(0).max()).min() < 1e-5, (
        "Projected mask does not match expected mask"
    )

    return


def test_image_simulation(sample_path_to_pdb1, sample_path_to_pdb2):
    particle_parameters = {
        "pose": cxs.EulerAnglePose(),
        "image_config": cxs.BasicImageConfig(
            shape=(32, 32), pixel_size=0.8, voltage_in_kilovolts=300.0
        ),
        "transfer_theory": cxs.ContrastTransferTheory(
            cxs.AstigmaticCTF(defocus_in_angstroms=200, spherical_aberration_in_mm=1e-16),
            amplitude_contrast_ratio=0.1,
        ),
    }

    volumes = tuple(
        [
            cxs.load_tabulated_volume(
                filename,
                output_type=cxs.GaussianMixtureVolume,
                include_b_factors=True,
                selection_string="not element H",
            )
            for filename in [sample_path_to_pdb1, sample_path_to_pdb2]
        ]
    )

    mask = CircularCosineMask(
        particle_parameters["image_config"].get_coordinate_grid(physical=False),
        radius=32 // 2,
        rolloff_width=1.0,
    )

    simulate_image_with_white_gaussian_noise(
        particle_parameters,
        constant_args=(volumes, mask, 1.0),
        per_particle_args=(jax.random.key(0), 0, 0.1),
    )

    return
