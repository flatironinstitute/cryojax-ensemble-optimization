import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.typing import ArrayLike
from typing import Dict


@jax.jit
def project_density(coord: np.ndarray, sigma: float, grid: np.ndarray) -> ArrayLike:
    """
    Generate a 2D projection from a set of coordinates.

    Args:
        coord (np.ndarray): Coordinates of the atoms in the image
        sigma (float): Standard deviation of the Gaussian function used to model electron density.
        grid: array with the set-up for one row of the grid (square images)

    Returns:
        image (jax.ArrayLike): Image generated from the coordinates
    """

    gauss_x = jnp.exp(-0.5 * (((grid[:, None] - coord[0, :]) / sigma) ** 2))
    gauss_y = jnp.exp(-0.5 * (((grid[:, None] - coord[1, :]) / sigma) ** 2))
    image = jnp.matmul(gauss_x, gauss_y.T)

    return image


@partial(jax.jit, static_argnames=["box_size", "pixel_size"])
def gen_img(
    coord: np.ndarray, box_size: int, pixel_size: float, sigma: float
) -> ArrayLike:
    """
    Generate an image from a set of coordinates.

    Args:
        coord (np.ndarray): Coordinates of the atoms in the image
        box_size (int): Number of pixels along one image size.
        pixel_size (float): Pixel size in Angstroms.
        sigma (float): Standard deviation of the Gaussian function used to model electron density.

    Returns:
        image (torch.Tensor): Image generated from the coordinates
    """

    num_atoms = coord.shape[1]
    norm = 1 / (2 * jnp.pi * sigma**2 * num_atoms)

    grid_min = -pixel_size * (box_size - 1) * 0.5
    grid_max = pixel_size * (box_size - 1) * 0.5 + pixel_size

    grid = jnp.arange(grid_min, grid_max, pixel_size)

    image = project_density(coord, sigma, grid) * norm

    return image
