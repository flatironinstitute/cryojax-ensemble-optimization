import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Union

from cryo_md.wpa_simulator.projection import gen_img
from cryo_md.wpa_simulator.ctf import apply_ctf
from cryo_md.wpa_simulator.noise import add_noise
from cryo_md.wpa_simulator.rotation import gen_quat, rotate_struct


def simulate_image(
    coords: np.ndarray,
    config: dict,
    quat: Union[np.ndarray, None] = None,
) -> ArrayLike:

    if config["rotation"]:
        if quat is None:
            quat = gen_quat()

        coords = rotate_struct(coords=coords.copy(), quat=quat)

    image = gen_img(
        coords,
        box_size=config["box_size"],
        pixel_size=config["pixel_size"],
        sigma=config["sigma"],
    )

    if config["ctf"]:
        image = apply_ctf(
            image=image,
            box_size=config["box_size"],
            pixel_size=config["pixel_size"],
            ctf_amp=config["ctf_amp"],
            ctf_bfactor=config["ctf_bfactor"],
            ctf_defocus=config["ctf_defocus"],
        )

    if config["noise"]:
        image = add_noise(
            image=image,
            box_size=config["box_size"],
            noise_radius_mask=config["noise_radius_mask"],
            snr=config["noise_snr"],
            seed=config["noise_seed"]
        )

    return image
