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

from cryo_md.wpa_simulator.rotation import gen_quat, calc_rot_matrix
from cryo_md.image.image_stack import ImageStack


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
            - quat : quaternion (0, 1, 2, 3)
            - shifts : 2D shifts (4, 5)
            - ctf_defocus : defocus (6)
            - ctf_amp : amplitude contrast (7)
            - ctf_bfactor : B-factor (8)
            - noise_snr : signal-to-noise ratio (9)
            - noise_variance : noise variance (10) <- this is filled in later

    """
    params = np.zeros((n_images, 11), dtype=dtype)

    params[:, 0:4] = gen_quat(n_images, dtype=dtype)
    # params[:, 4:6] = shifts  # TODO

    for i, key in enumerate(["ctf_defocus", "ctf_amp", "ctf_bfactor", "noise_snr"]):
        if isinstance(config[key], float):
            params[:, i + 6] = np.repeat(config[key], n_images)

        elif isinstance(config[key], list) and len(config[key]) == 2:
            params[:, i + 6] = np.random.uniform(
                low=config[key][0], high=config[key][1], size=n_images
            )

        else:
            raise ValueError(
                f"{key} should be a single float value or a list of [min_{key}, max_{key}]"
            )

    return params


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def full_simulator_(
    coords: ArrayLike,
    box_size: int,
    pixel_size: float,
    sigma: float,
    noise_radius_mask: int,
    var_imaging_args: ArrayLike,
    random_key: ArrayLike,
) -> ArrayLike:
    """
    Simulate a single image.

    Parameters
    ----------
    coords : ArrayLike
        Array of shape (3, n_atoms) with the coordinates of the atoms.
    box_size : int
        Size of the box in pixels.
    pixel_size : float
        Size of a pixel in Angstroms.
    sigma : float
        Standard deviation of the Gaussian that represents the atomic scattering.
    noise_radius_mask : int
        Radius of the mask that defines the signal for the noise calculation.
    var_imaging_args : ArrayLike
        Array of shape (11) with the following parameters:
            - quat : quaternion (0, 1, 2, 3)
            - shifts : 2D shifts (4, 5)
            - ctf_defocus : defocus (6)
            - ctf_amp : amplitude contrast (7)
            - ctf_bfactor : B-factor (8)
            - noise_snr : signal-to-noise ratio (9)
            - noise_variance : noise variance (10) <- this is filled here!
    random_key : ArrayLike
        Random key for the noise generation (Jax).

    Returns
    -------
    image : ArrayLike
        Array of shape (box_size, box_size) with the simulated image. Images are normalized so that their noiseless version has an l2-norm of 1.
    noise_power : float
        Noise power of the image.
    """
    num_atoms = coords.shape[1]
    norm = 1 / (2 * jnp.pi * sigma**2 * num_atoms)

    # Rotate coordinates
    coords = jnp.matmul(calc_rot_matrix(var_imaging_args[0:4]), coords)

    # Project
    grid_min = -pixel_size * box_size * 0.5
    grid_max = pixel_size * box_size * 0.5
    grid = jnp.arange(grid_min, grid_max, pixel_size)[0:box_size]

    gauss_x = jnp.exp(-0.5 * (((grid[:, None] - coords[0, :]) / sigma) ** 2))
    gauss_y = jnp.exp(-0.5 * (((grid[:, None] - coords[1, :]) / sigma) ** 2))
    image = jnp.matmul(gauss_x, gauss_y.T) * norm

    # # Apply CTF
    freq_pix_1d = jnp.fft.fftfreq(box_size, d=pixel_size)
    freq2_2d = freq_pix_1d[:, None] ** 2 + freq_pix_1d[None, :] ** 2

    elecwavel = 0.019866
    phase = var_imaging_args[6] * jnp.pi * 2.0 * 10000 * elecwavel

    env = jnp.exp(-var_imaging_args[8] * freq2_2d * 0.5)
    ctf = (
        (
            var_imaging_args[8] * jnp.cos(phase * freq2_2d * 0.5)
            - jnp.sqrt(1 - var_imaging_args[8] ** 2) * jnp.sin(phase * freq2_2d * 0.5)
            + 0.0j
        )
        * env
        / var_imaging_args[8]
    )

    # Normalize image
    image = jnp.fft.ifft2(jnp.fft.fft2(image) * ctf).real
    image /= jnp.linalg.norm(image)

    # add noise
    noise_grid = jnp.linspace(-0.5 * (box_size - 1), 0.5 * (box_size - 1), box_size)
    radii_for_mask = noise_grid[None, :] ** 2 + noise_grid[:, None] ** 2
    mask = radii_for_mask < noise_radius_mask**2

    signal_power = jnp.sqrt(jnp.sum((image * mask) ** 2) / jnp.sum(mask))

    noise_power = signal_power / jnp.sqrt(var_imaging_args[9])
    image = image + jax.random.normal(random_key, shape=image.shape) * noise_power

    return image, noise_power


batch_full_simulator_ = jax.vmap(
    full_simulator_, in_axes=(None, None, None, None, None, 0, 0)
)


def simulate_stack(
    models: ArrayLike,
    images_per_model: list,
    config: dict,
    batch_size: Union[int, None] = None,
    dtype: type = float,
    seed: int = 0,
) -> ImageStack:
    """
    Simulate a stack of images.

    Parameters
    ----------
    models : ArrayLike
        Array of shape (n_models, n_atoms, 3) with the coordinates of the atoms.
    images_per_model : list
        List of length n_models with the number of images to simulate for each model.
    config : dict
        Dictionary with the following keys:
            - box_size : int
                Size of the box in pixels.
            - pixel_size : float
                Size of a pixel in Angstroms.
            - sigma : float
                Standard deviation of the Gaussian that represents the atomic scattering.
            - noise_radius_mask : int
                Radius of the mask that defines the signal for the noise calculation.
            - ctf_defocus : float or list of [min, max]
                Defocus value or range of values.
            - ctf_amp : float or list of [min, max]
                Amplitude contrast value or range of values.
            - ctf_bfactor : float or list of [min, max]
                B-factor value or range of values.
            - noise_snr : float or list of [min, max]
                Signal-to-noise ratio value or range of values.
    batch_size : Union[int, None], optional
        Batch size for the image generation, by default None (Unused right now)
    dtype : type, optional
        Data type of the parameters and the images, by default jax.numpy.float32
    seed : int, optional
        Seed for the random number generator, by default 0
    
    Returns
    -------
    images_stack : ImageStack
        ImageStack object with the simulated images and their parameters.
    """
    
    n_images = np.sum(images_per_model)

    images_stack = ImageStack()

    images_stack.init_for_stacking(
        n_images=n_images,
        box_size=config["box_size"],
        pixel_size=config["pixel_size"],
        sigma=config["sigma"],
        noise_radius_mask=config["noise_radius_mask"],
        dtype=jnp.float32,
    )

    key = jax.random.PRNGKey(seed)

    for i in range(models.shape[0]):

        key, *subkeys = jax.random.split(key, num=images_per_model[i] + 1)
        subkeys = jnp.array(subkeys)

        variable_params = generate_params_(n_images=images_per_model[i], config=config, dtype=dtype)

        batch_images, noise_variances = batch_full_simulator_(
            models[i],
            config["box_size"],
            config["pixel_size"],
            config["sigma"],
            config["noise_radius_mask"],
            variable_params,
            subkeys,
        )

        variable_params[:, 10] = noise_variances
        images_stack.stack_batch(batch_images, variable_params)

    return images_stack


batch_over_models_simulator = jax.vmap(
    full_simulator_, in_axes=(0, None, None, None, None, 0, 0)
)


def simulate_stack_traj(
    models,
    config: dict,
    dtype: type = float,
    seed: int = 0,
):
    images_stack = ImageStack()

    images_stack.init_for_stacking(
        n_images=models.shape[0],
        box_size=config["box_size"],
        pixel_size=config["pixel_size"],
        sigma=config["sigma"],
        noise_radius_mask=config["noise_radius_mask"],
        dtype=jnp.float32,
    )

    variable_params = generate_params_(
        n_images=models.shape[0], config=config, dtype=dtype
    )

    key = jax.random.PRNGKey(seed)

    key, *subkeys = jax.random.split(key, num=models.shape[0] + 1)
    subkeys = jnp.array(subkeys)

    batch_images, noise_variances = batch_over_models_simulator(
        models,
        config["box_size"],
        config["pixel_size"],
        config["sigma"],
        config["noise_radius_mask"],
        variable_params,
        subkeys,
    )

    variable_params[:, 10] = noise_variances
    variable_params = jnp.array(variable_params)

    images_stack.stack_batch(batch_images, variable_params)

    return images_stack
