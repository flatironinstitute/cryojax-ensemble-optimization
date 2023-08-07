import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial

from cryo_md.wpa_simulator.rotation import calc_rot_matrix


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def noiseless_simulator_(
    coords: ArrayLike,
    struct_info: ArrayLike,
    box_size: int,
    pixel_size: float,
    res: float,
    var_imaging_args: ArrayLike,
) -> ArrayLike:
    """
    Simulate a noiseless image. This function is used in the likelihood calculation.

    Parameters
    ----------
    coords : ArrayLike
        Coordinates of the atoms.
    struct_info : ArrayLike
        Structural information.
    box_size : int
        Size of the box.
    pixel_size : float
        Pixel size.
    res : float
        Resolution of density map where this image comes from.
    var_imaging_args : ArrayLike
        Imaging parameters.

    Returns
    -------
    ArrayLike
        Noiseless image.

    Notes
    -----
    For the structural information, the first row should be related to the variance of the Gaussian, e.g., the radius of the aminoacid. The second row should be related to the amplitude of the Gaussian, e.g., the number of electrons in the atom/residue (for coarse grained models)
    """
    box_size = int(box_size)

    gauss_var = struct_info[0, :] * res**2
    gauss_amp = struct_info[1, :] / jnp.sqrt(gauss_var * 2.0 * jnp.pi)

    # Rotate coordinates
    coords = jnp.matmul(calc_rot_matrix(var_imaging_args[0:4]), coords)

    # Project
    grid_min = -pixel_size * box_size * 0.5
    grid_max = pixel_size * box_size * 0.5

    grid = jnp.arange(grid_min, grid_max, pixel_size)[0:box_size]

    gauss_x = gauss_amp * jnp.exp(
        -0.5 * (((grid[:, None] - coords[0, :]) / gauss_var) ** 2)
    )
    gauss_y = gauss_amp * jnp.exp(
        -0.5 * (((grid[:, None] - coords[1, :]) / gauss_var) ** 2)
    )
    image = jnp.matmul(gauss_x, gauss_y.T)

    # # Apply CTF
    freq_pix_1d = jnp.fft.fftfreq(box_size, d=pixel_size)
    freq2_2d = freq_pix_1d[:, None] ** 2 + freq_pix_1d[None, :] ** 2

    elecwavel = 0.019866
    phase = var_imaging_args[6] * jnp.pi * 2.0 * 10000 * elecwavel

    env = jnp.exp(-var_imaging_args[8] * freq2_2d * 0.5)
    ctf = (
        (
            var_imaging_args[7] * jnp.cos(phase * freq2_2d * 0.5)
            - jnp.sqrt(1 - var_imaging_args[7] ** 2) * jnp.sin(phase * freq2_2d * 0.5)
            + 0.0j
        )
        * env
        / var_imaging_args[7]
    )

    image = jnp.fft.ifft2(jnp.fft.fft2(image) * ctf).real

    image /= jnp.linalg.norm(image)

    return image
