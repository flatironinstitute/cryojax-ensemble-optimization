import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial

from cryo_md.wpa_simulator.rotation import calc_rot_matrix


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def noiseless_simulator_(
    coords: ArrayLike,
    box_size: int,
    pixel_size: float,
    sigma: float,
    var_imaging_args: ArrayLike,
) -> ArrayLike:
    box_size = int(box_size)

    num_atoms = coords.shape[1]
    norm = 1 / (2 * jnp.pi * sigma**2 * num_atoms)

    # Rotate coordinates
    # coords = jnp.matmul(calc_rot_matrix(var_imaging_args[0:4]), coords)

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

    return image
