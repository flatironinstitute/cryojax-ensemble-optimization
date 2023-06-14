import numpy as np
import jax.numpy as jnp
import jax
from functools import partial
from jax.typing import ArrayLike


@partial(jax.jit, static_argnames=["box_size"])
def calc_ctf(
    box_size: int, pixel_size: float, amp: float, phase: float, b_factor: float
):
    freq_pix_1d = jnp.fft.fftfreq(box_size, d=pixel_size)

    freq2_2d = freq_pix_1d[:, None] ** 2 + freq_pix_1d[None, :] ** 2

    env = jnp.exp(-b_factor * freq2_2d * 0.5)
    ctf = (
        amp * jnp.cos(phase * freq2_2d * 0.5)
        - jnp.sqrt(1 - amp**2) * jnp.sin(phase * freq2_2d * 0.5)
        + 0.0j
    )

    return ctf * env / amp


def apply_ctf(
    image: ArrayLike,
    box_size: int,
    pixel_size: float,
    ctf_amp: float,
    ctf_bfactor: float,
    ctf_defocus: float,
):
    elecwavel = 0.019866
    phase = ctf_defocus * jnp.pi * 2.0 * 10000 * elecwavel

    ctf = calc_ctf(
        box_size=box_size,
        pixel_size=pixel_size,
        amp=ctf_amp,
        phase=phase,
        b_factor=ctf_bfactor,
    )

    conv_image_ctf = jnp.fft.fft2(image) * ctf
    image_ctf = jnp.fft.ifft2(conv_image_ctf).real

    return image_ctf
