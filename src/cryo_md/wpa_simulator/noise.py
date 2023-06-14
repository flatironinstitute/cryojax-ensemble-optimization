import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from jax.typing import ArrayLike
from typing import Union


@partial(jax.jit, static_argnames=["n_pixels"])
def circular_mask(n_pixels: int, radius: int) -> ArrayLike:
    """
    Creates a circular mask of radius RADIUS_MASK centered in the image

    Args:
        n_pixels (int): Number of pixels along image side.
        radius (int): Radius of the mask.

    Returns:
        mask (ArrayLike): Mask of shape (n_pixels, n_pixels).
    """

    grid = jnp.linspace(-0.5 * (n_pixels - 1), 0.5 * (n_pixels - 1), n_pixels)
    r_2d = grid[None, :] ** 2 + grid[:, None] ** 2
    mask = r_2d < radius**2

    return mask


def add_noise(
    image: ArrayLike,
    box_size: int,
    noise_radius_mask: int,
    snr: float,
    seed: Union[None, int] = None,
) -> ArrayLike:
    """
    Adds noise to image

    Args:
        image (ArrayLike): Image of shape (n_pixels, n_pixels).
        image_params (dict): Dictionary with image parameters.
        seed (int, optional): Seed for random number generator. Defaults to None.

    Returns:
        image_noise (ArrayLike): Image with noise of shape (n_pixels, n_pixels) or (n_channels, n_pixels, n_pixels).
    """

    if seed is not None:
        np.random.seed(seed)

    mask = circular_mask(n_pixels=box_size, radius=noise_radius_mask)
    signal_power = jnp.sqrt(jnp.mean(image[mask] ** 2))

    noise_power = signal_power / jnp.sqrt(snr)
    image_noise = image + jnp.array(np.random.normal(0, noise_power, image.shape))

    return image_noise
