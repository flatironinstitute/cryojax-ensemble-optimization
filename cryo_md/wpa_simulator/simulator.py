# =============================================================================
# This module contains the functions to simulate images from a set of atomic
# coordinates.
#
import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
from typing import Union
import mrcfile
import os
import starfile
import pandas as pd

from cryo_md.wpa_simulator.rotation import gen_euler, calc_rot_matrix
from cryo_md.image.image_stack import load_starfile


def compute_grids(box_size, pixel_size):
    values = np.linspace(-0.5, 0.5, box_size + 1)[:-1]
    proj_grid = values * pixel_size * box_size

    fx2, fy2 = np.meshgrid(-values, values, indexing="xy")
    ctf_grid = np.stack((fx2.ravel(), fy2.ravel()), axis=1) / pixel_size

    return proj_grid, ctf_grid


def compute_ctf_params(volt, cs, amp_contrast, imaging_params):
    volt = volt * 1000.0
    cs = cs * 1e7
    lam = 12.2639 / np.sqrt(volt + 0.97845e-6 * volt**2)

    ctf_params = np.zeros((imaging_params.shape[0], 9))
    ctf_params[:, 0] = imaging_params[:, 5]
    ctf_params[:, 1] = imaging_params[:, 6]
    ctf_params[:, 2] = np.radians(imaging_params[:, 7])
    ctf_params[:, 3] = np.radians(imaging_params[:, 10])
    ctf_params[:, 4] = amp_contrast
    ctf_params[:, 5] = cs
    ctf_params[:, 6] = imaging_params[:, 8]
    ctf_params[:, 7] = imaging_params[:, 9]
    ctf_params[:, 8] = lam

    return ctf_params


def generate_params_(n_images, config, dtype=float):
    """
    Generate a set of parameters for each image in the stack.

    Parameters
    ----------
    n_images : int
        Number of images in the stack.
    config : dict
        Dictionary with at least the following keys:
            - ctf_defocus : float or list of [min, max]
            - ctf_amp : float or list of [min, max]
            - ctf_bfactor : float or list of [min, max]
            - noise_snr : float or list of [min, max]
    dtype : type, optional
        Data type of the parameters, by default float

    Returns
    -------
    params: ArrayLike
        Array of shape (n_images, 11) with the following columns:
            - shifts : 2D shifts (0, 1)
            - euler_angs : Euler angles (2, 3, 4)
            - ctf defocus params : (DefocusU, DefocusV, DefocusAng) = (5, 6, 7)
            - ctf extra params : (Bfactor, Scalefactor, PhaseShift) = (8, 9, 10)
            - noise SNR : (11)

    """
    params = np.zeros((n_images, 12), dtype=dtype)

    params[:, 0:2] = np.zeros((n_images, 2))  # shifts
    params[:, 2:5] = gen_euler(n_images, dtype=dtype)

    for i, key in enumerate(
        [
            "defocus_u",
            "defocus_v",
            "defocus_ang",
            "bfactor",
            "scalefactor",
            "phaseshift",
            "noise_snr",
        ]
    ):
        if isinstance(config[key], float):
            params[:, i + 5] = np.repeat(config[key], n_images)

        elif isinstance(config[key], list) and len(config[key]) == 2:
            params[:, i + 5] = np.random.uniform(
                low=config[key][0], high=config[key][1], size=n_images
            )

        else:
            raise ValueError(
                f"{key} should be a single float value or a list of [min_{key}, max_{key}]"
            )

    params[:, 6] = params[:, 5]

    return params.astype(dtype)


def get_filename(step, n_char=6):
    if step == 0:
        return "0" * n_char
    else:
        n_dec = int(np.log10(step))
        return "0" * (n_char - n_dec) + str(step)


def create_df_for_starfile_(
    starfile_fname, n_images, config, imaging_params, batch_size
):
    starf_new = dict()

    # Generate optics group
    optics_df = pd.DataFrame()
    optics_df["rlnOpticsGroup"] = [1]
    optics_df["rlnVoltage"] = [config["volt"]]
    optics_df["rlnSphericalAberration"] = [config["spherical_aberr"]]
    optics_df["rlnImagePixelSize"] = [config["pixel_size"]]
    optics_df["rlnImageSize"] = [config["box_size"]]
    optics_df["rlnAmplitudeContrast"] = [config["amp_contrast"]]
    starf_new["optics"] = optics_df

    # Generate particles group
    particles_df = pd.DataFrame()
    particles_df["rlnOriginXAngst"] = imaging_params[:, 0]
    particles_df["rlnOriginYAngst"] = imaging_params[:, 1]
    particles_df["rlnAngleRot"] = np.degrees(imaging_params[:, 2])
    particles_df["rlnAngleTilt"] = np.degrees(imaging_params[:, 3])
    particles_df["rlnAnglePsi"] = np.degrees(imaging_params[:, 4])
    particles_df["rlnDefocusU"] = imaging_params[:, 5]
    particles_df["rlnDefocusV"] = imaging_params[:, 6]
    particles_df["rlnDefocusAngle"] = np.degrees(imaging_params[:, 7])
    particles_df["rlnCtfBfactor"] = imaging_params[:, 8]
    particles_df["rlnCtfScalefactor"] = imaging_params[:, 9]
    particles_df["rlnPhaseShift"] = np.degrees(imaging_params[:, 10])

    # fixed values
    particles_df["rlnCtfMaxResolution"] = np.zeros(n_images)
    particles_df["rlnCtfFigureOfMerit"] = np.zeros(n_images)
    particles_df["rlnRandomSubset"] = np.random.randint(1, 2, size=n_images)
    particles_df["rlnClassNumber"] = np.ones(n_images)
    particles_df["rlnOpticsGroup"] = np.ones(n_images)

    n_batches = n_images // batch_size
    n_remainder = n_images % batch_size

    relative_mrcs_path_prefix = starfile_fname.split(".")[0]
    image_names = []

    for step in range(n_batches):
        filename = get_filename(step, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + filename + ".mrcs"
        image_names += [
            get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(batch_size)
        ]

    if n_remainder > 0:
        filename = get_filename(n_batches, n_char=6)
        mrc_relative_path = relative_mrcs_path_prefix + filename + ".mrcs"
        image_names += [
            get_filename(i + 1, n_char=6) + "@" + mrc_relative_path
            for i in range(n_remainder)
        ]

    particles_df["rlnImageName"] = image_names

    starf_new["particles"] = particles_df

    return starf_new

@jax.jit
def simulator_(
    coords: ArrayLike,
    struct_info: ArrayLike,
    grid: ArrayLike,
    grid_ctf: ArrayLike,
    res: float,
    pose_params: ArrayLike,
    ctf_params: ArrayLike,
) -> ArrayLike:
    """
    Simulate a single image.

    Parameters
    ----------
    coords : ArrayLike
        Array of shape (3, n_atoms) with the coordinates of the atoms.
    struct_info : ArrayLike
        Structural information of the models.
    grid : ArrayLike
        Array of shape (box_size, box_size) with the grid coordinates (generated by ImageStack).
    grid_f : ArrayLike
        Array of shape (box_size, box_size) with the Fourier grid coordinates (generated by ImageStack).
    res : float
        Resolution of density map where this image comes from.
    pose_params : ArrayLike
        Array of shape (11) with the following parameters:
            - euler_angles : Euler angles (0, 1, 2)
            - shifts : 2D shifts (3, 4)
            - ctf_defocus : defocus (5)
            - ctf_amp : amplitude contrast (6)
            - ctf_bfactor : B-factor (7)
            - noise_snr : signal-to-noise ratio (9) <- unused here, see add_noise()
            - noise_variance : noise variance (10) <- unused here, see add_noise()

    Returns
    -------
    image : ArrayLike
        Array of shape (box_size, box_size) with the simulated image. Images are normalized so that their noiseless version has an l2-norm of 1.

    Notes
    -----
    For the structural information of the models, the first row should be related to the variance of the Gaussian, e.g., the radius of the aminoacid. The second row should be related to the amplitude of the Gaussian, e.g., the number of electrons in the atom/residue (for coarse grained models)
    """

    # assert pixel_size < 2.0 * res, "Pixel size should be smaller than 2.0 * res due to the Nyquist limit."

    gauss_var = struct_info[0, :] * res**2
    gauss_amp = struct_info[1, :] / jnp.sqrt(gauss_var * 2.0 * jnp.pi)

    # Rotate coordinates
    transf_coords = jnp.matmul(calc_rot_matrix(pose_params[2:]), coords)
    transf_coords = (
        transf_coords - jnp.concatenate([pose_params[0:2], jnp.zeros(1)])[:, None]
    )

    gauss_x = gauss_amp * jnp.exp(
        -0.5 * ((grid[:, None] - transf_coords[0, :]) ** 2 / gauss_var)
    )
    gauss_y = gauss_amp * jnp.exp(
        -0.5 * ((grid[:, None] - transf_coords[1, :]) ** 2 / gauss_var)
    )
    image = -jnp.matmul(gauss_y, gauss_x.T)

    # Apply CTF
    x = grid_ctf[:, 0]
    y = grid_ctf[:, 1]

    ang = jnp.arctan2(y, x)
    s2 = x**2 + y**2
    defocus = 0.5 * (
        ctf_params[0]
        + ctf_params[1]
        + (ctf_params[0] - ctf_params[1]) * jnp.cos(2 * (ang - ctf_params[2]))
    )

    gamma = (
        2
        * jnp.pi
        * (
            -0.5 * defocus * ctf_params[8] * s2
            + 0.25 * ctf_params[5] * ctf_params[8] ** 3 * s2**2
        )
        - ctf_params[3]
    )

    ctf = jnp.sqrt(1 - ctf_params[4] ** 2) * jnp.sin(gamma) - ctf_params[4] * jnp.cos(
        gamma
    )
    ctf *= ctf_params[7] * jnp.exp(-ctf_params[6] / 4.0 * s2)
    ctf = ctf.reshape(*image.shape)

    image_ft = jnp.fft.fftshift(jnp.fft.fft2(image))
    image = jnp.fft.ifft2(jnp.fft.ifftshift(image_ft * ctf)).real
    image /= jnp.linalg.norm(image)

    return image


def add_noise_(image, noise_grid, noise_radius_mask, noise_snr, random_key):
    radii_for_mask = noise_grid[None, :] ** 2 + noise_grid[:, None] ** 2
    mask = radii_for_mask < noise_radius_mask**2

    signal_power = jnp.sqrt(jnp.sum((image * mask) ** 2) / jnp.sum(mask))

    noise_power = signal_power / jnp.sqrt(noise_snr)
    image = image + jax.random.normal(random_key, shape=image.shape) * noise_power

    return image, noise_power


batch_simulator_ = jax.vmap(simulator_, in_axes=(0, None, None, None, None, 0, 0))

batch_add_noise_ = jax.vmap(add_noise_, in_axes=(0, None, None, 0, 0))


def simulate_stack(
    root_path: str,
    starfile_fname: str,
    models: ArrayLike,
    struct_info: ArrayLike,
    images_per_model: ArrayLike,
    config: dict,
    batch_size: Union[int, None] = None,
    dtype: type = float,
    seed: int = 0,
) -> None:
    assert models.shape[0] == len(
        images_per_model
    ), "Number of models and number of images per model do not match"
    assert (
        batch_size is None or batch_size > 0
    ), "Batch size should be a positive integer"

    n_images = np.sum(images_per_model)

    if batch_size is None:
        batch_size = n_images

    assert (
        batch_size <= n_images
    ), "Batch size should be smaller than the total number of images"

    n_batches = n_images // batch_size
    batch_residual = n_images % batch_size

    rep_models = np.repeat(models, images_per_model, axis=0).astype(dtype)

    proj_grid, ctf_grid = compute_grids(config["box_size"], config["pixel_size"])

    imaging_params = generate_params_(n_images, config, dtype=dtype)
    ctf_params = compute_ctf_params(
        config["volt"],
        config["spherical_aberr"],
        config["amp_contrast"],
        imaging_params,
    )
    new_starfile = create_df_for_starfile_(
        starfile_fname, n_images, config, imaging_params, batch_size
    )

    imaging_params = jnp.array(imaging_params)
    ctf_params = jnp.array(ctf_params)

    key = jax.random.PRNGKey(seed)

    noise_variances = np.zeros(n_images)
    noise_grid = jnp.linspace(
        -0.5 * (config["box_size"] - 1),
        0.5 * (config["box_size"] - 1),
        config["box_size"],
    )

    for j in range(n_batches):
        batch_index_o = j * batch_size
        batch_index_f = (j + 1) * batch_size

        key, *subkeys = jax.random.split(key, num=batch_size + 1)
        subkeys = jnp.array(subkeys)

        mrc_relative_path = new_starfile["particles"]["rlnImageName"][
            batch_size * j
        ].split("@")[1]
        mrc_path = os.path.join(root_path, mrc_relative_path)

        batch_images = batch_simulator_(
            rep_models[batch_index_o:batch_index_f],
            struct_info,
            proj_grid,
            ctf_grid,
            config["res"],
            imaging_params[batch_index_o:batch_index_f, 0:5],  # shifts and euler angles
            ctf_params[batch_index_o:batch_index_f],  # ctf params
        )

        batch_images, noise_var = batch_add_noise_(
            batch_images,
            noise_grid,
            config["noise_radius_mask"],
            imaging_params[batch_index_o:batch_index_f, 11],
            subkeys,
        )

        noise_variances[batch_index_o:batch_index_f] = noise_var

        with mrcfile.new_mmap(
            mrc_path,
            shape=(batch_size, config["box_size"], config["box_size"]),
            mrc_mode=2,
            overwrite=True,
        ) as mrc_file:
            for i in range(batch_size):
                mrc_file.data[i] = batch_images[i]

    if batch_residual > 0:
        batch_index_o = n_batches * batch_size
        batch_index_f = n_images

        key, *subkeys = jax.random.split(key, num=batch_residual + 1)
        subkeys = jnp.array(subkeys)

        mrc_relative_path = new_starfile["particles"]["rlnImageName"][
            batch_size * n_batches
        ].split("@")[1]
        mrc_path = os.path.join(root_path, mrc_relative_path)

        batch_images = batch_simulator_(
            rep_models[batch_index_o:batch_index_f],
            struct_info,
            proj_grid,
            ctf_grid,
            config["res"],
            imaging_params[batch_index_o:batch_index_f, 0:5],  # shifts and euler angles
            ctf_params[batch_index_o:batch_index_f],  # ctf params
        )

        batch_images, noise_var = batch_add_noise_(
            batch_images,
            noise_grid,
            config["noise_radius_mask"],
            imaging_params[batch_index_o:batch_index_f, 11],
            subkeys,
        )

        noise_variances[batch_index_o:batch_index_f] = noise_var

        with mrcfile.new_mmap(
            mrc_path,
            shape=(batch_residual, config["box_size"], config["box_size"]),
            mrc_mode=2,
            overwrite=True,
        ) as mrc_file:
            for i in range(batch_residual):
                mrc_file.data[i] = batch_images[i]

    new_starfile["particles"]["rlnNoiseVariance"] = noise_variances
    new_starfile["particles"]["rlnNoiseSNR"] = imaging_params[:, 11]

    starfile.write(new_starfile, os.path.join(root_path, starfile_fname))

    image_stack = load_starfile(root_path, starfile_fname, res=config["res"], batch_size=batch_size)

    return image_stack
