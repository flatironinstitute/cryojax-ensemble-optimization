import numpy as np
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Union


class ImageStack:
    def __init__(self):
        pass

    def init_for_stacking(
        self,
        n_images: int,
        box_size: int,
        pixel_size: float,
        sigma: float,
        noise_radius_mask: int,
        dtype,
    ):
        """
        Use this function when you are going to generate images and stack them sequentally
        """

        self.images = np.empty((n_images, box_size, box_size), dtype=dtype)
        self.variable_params = np.empty((n_images, 10), dtype=dtype)

        self.stacked_images_ = 0

        self.constant_params = np.empty((4,), dtype=dtype)
        self.constant_params[0] = box_size
        self.constant_params[1] = pixel_size
        self.constant_params[2] = sigma
        self.constant_params[3] = noise_radius_mask

    def stack_batch(self, batch_images, batch_params):
        if batch_images.ndim == 2:
            batch_images = batch_images[None, :, :]

        if batch_params.ndim == 1:
            batch_params = batch_params[None, :]

        self.images[
            self.stacked_images_ : self.stacked_images_ + batch_images.shape[0]
        ] = batch_images
        self.variable_params[
            self.stacked_images_ : self.stacked_images_ + batch_images.shape[0]
        ] = batch_params

        self.stacked_images_ += batch_images.shape[0]
