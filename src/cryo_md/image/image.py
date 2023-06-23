import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Union
import json
import matplotlib.pyplot as plt

from cryo_md.wpa_simulator.ctf import apply_ctf
from cryo_md.wpa_simulator.noise import add_noise


class Image:
    def __init__(self):
        pass

    def init_from_json(self, filename: str):
        with open(filename, "r") as metadata_file:
            self.metadata = json.load(metadata_file)

        self._data = jnp.load(self.metadata["filename"])
        self.dtype = self._data.dtype
        self.shape = self._data.shape

        return

    def init_from_data(
        self,
        data: ArrayLike,
        pixel_size: float,
        sigma: float,
        orientation: Union[ArrayLike, None] = None,
        dtype=None,
    ) -> None:
        """
        Initialize image using existing data from a projection

        Parameters:
        data: array_like, shape = (N, N)
            Array containing the image data, that is, the value of the density at every pixel.

        pixel_size: float
            Size of each pixel in Amstrong.

        sigma: float
            Width used to generate Gaussians

        orientation (optional): array_like, shape = (4,)
            quaternions defining the orientation of the image with convention (x, y, z, w)

        """

        self.metadata = {
            "box_size": data.shape[0],
            "pixel_size": pixel_size,
            "sigma": sigma,
            "ctf": False,
            "noise": False,
        }

        if orientation is None:
            self.metadata["rotation"] = False

        else:
            self.metadata["rotation"] = True
            self.metadata["orientation"] = orientation

        if dtype is None:
            self.dtype = data.dtype

        self._data = jnp.asarray(data, dtype=self.dtype)
        self.shape = self._data.shape

        return

    def apply_ctf(self, amplitude: float, bfactor: float, defocus: float) -> None:
        self._data = apply_ctf(
            image=self._data,
            box_size=self.metadata["box_size"],
            pixel_size=self.metadata["pixel_size"],
            ctf_amp=amplitude,
            ctf_bfactor=bfactor,
            ctf_defocus=defocus,
        )

        self.metadata["ctf"] = True
        self.metadata["ctf_amp"] = amplitude
        self.metadata["ctf_bfactor"] = bfactor
        self.metadata["ctf_defocus"] = defocus

        return

    def add_noise(
        self, noise_radius_mask: int, snr: float, seed: Union[int, None] = None
    ) -> None:
        self._data = add_noise(
            image=self._data,
            box_size=self.metadata["box_size"],
            noise_radius_mask=noise_radius_mask,
            snr=snr,
            seed=seed,
        )

        self.metadata["noise"] = True
        self.metadata["noise_radius_mask"] = noise_radius_mask
        self.metadata["noise_snr"] = snr
        self.metadata["noise_seed"] = seed

        return

    def show(self, figsize=(4, 4)):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.imshow(self._data, cmap="gray")

    def save_image(self, fname_prefix: str):
        self.metadata["filename"] = f"{fname_prefix}.npy"
        jnp.save(f"{fname_prefix}.npy", self._data)

        with open(f"{fname_prefix}.json", "w") as metadata_file:
            json.dump(self.metadata, metadata_file)

        return
