import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
from typing import Union

from cryo_md.wpa_simulator.rotation import gen_quat, calc_rot_matrix
from cryo_md.image.image_stack import ImageStack


def generate_params(n_images, config, dtype=float):
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
    num_atoms = coords.shape[1]
    norm = 1 / (2 * jnp.pi * sigma**2 * num_atoms)

    # Rotate coordinates
    coords = jnp.matmul(calc_rot_matrix(var_imaging_args[0:4]), coords)

    # Project
    grid_min = -pixel_size * (box_size - 1) * 0.5
    grid_max = pixel_size * (box_size - 1) * 0.5 + pixel_size

    grid = jnp.arange(grid_min, grid_max, pixel_size)

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
    coords: ArrayLike,
    n_images: int,
    config: dict,
    batch_size: Union[int, None] = None,
    dtype: type = float,
    seed: int = 0,
) -> ArrayLike:
    if batch_size is None:
        batch_size = n_images

    assert (
        batch_size <= n_images
    ), "The number of batches should be smaller than the number of images"

    images_stack = ImageStack()

    images_stack.init_for_stacking(
        n_images=n_images,
        box_size=config["box_size"],
        pixel_size=config["pixel_size"],
        sigma=config["sigma"],
        noise_radius_mask=config["noise_radius_mask"],
        dtype=jnp.float32,
    )

    n_batches = n_images // batch_size

    variable_params = generate_params(n_images=n_images, config=config, dtype=dtype)

    key = jax.random.PRNGKey(seed)

    for i in range(n_batches):
        start_batch = i * batch_size
        end_batch = (i + 1) * batch_size

        key, *subkeys = jax.random.split(key, num=batch_size + 1)
        subkeys = jnp.array(subkeys)

        batch_images = batch_full_simulator_(
            coords,
            config["box_size"],
            config["pixel_size"],
            config["sigma"],
            config["noise_radius_mask"],
            variable_params[start_batch:end_batch],
            subkeys,
        )

        images_stack.stack_batch(batch_images, variable_params[start_batch:end_batch])

    batch_residual = n_images % batch_size

    if batch_residual > 0:
        key, *subkeys = jax.random.split(key, num=batch_residual + 1)
        subkeys = jnp.array(subkeys)

        batch_images = batch_full_simulator_(
            coords,
            config["box_size"],
            config["pixel_size"],
            config["sigma"],
            config["noise_radius_mask"],
            variable_params[end_batch:],
            subkeys,
        )

        images_stack.stack_batch(batch_images, variable_params[end_batch:])

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

    variable_params = generate_params(
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
