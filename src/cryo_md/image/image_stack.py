"""
Class for storing images and their parameters.
"""
import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing import Optional


class ImageStack:
    def __init__(self, fname: Optional[str] = None):
        """
        Class for storing images and their parameters.

        Parameters
        ----------
        fname : str, optional
            Path to file containing images and parameters, by default None. If None, then you can use init_for_stacking() to initialize the class for stacking images. If str, then the class will be initialized from the file.

        Attributes
        ----------
        images : ArrayLike
            Array of images
        variable_params : ArrayLike
            Array of variable parameters, contrains 11 parameters:
                0:4 - quaternions that define the rotation matrix
                4:6 - in-plane translation
                6 - CTF defocus
                7 - CTF amplitude
                8 - CTF bfactor
                9 - noise SNR
                10 - noise variance
        constant_params : ArrayLike
            Array of constant parameters


        """
        if fname is not None:
            self.load_(fname)

        pass

    def init_for_stacking(
        self,
        n_images: int,
        box_size: int,
        pixel_size: float,
        res: float,
        noise_radius_mask: int,
        dtype: type = np.float32,
    ):
        """
        Use this function when you are going to generate images and stack them sequentally, i.e. you know the number of images beforehand.

        Parameters
        ----------
        n_images : int
            Number of images to stack
        box_size : int
            Size of the box
        pixel_size : float
            Pixel size
        res : float
            Standard deviation of the Gaussian that is used to project the atoms onto the image
        noise_radius_mask : int
            Radius of the mask that defines the signal for the noise calculation
        dtype : type, optional
            Data type of the images and parameters, by default np.float32

        Returns
        -------
        None
        """

        self.n_images = n_images
        self.images = np.empty((n_images, box_size, box_size), dtype=dtype)
        self.variable_params = np.empty((n_images, 11), dtype=dtype)

        self.stacked_images_ = 0

        self.constant_params = np.empty((4,), dtype=dtype)
        self.constant_params[0] = box_size
        self.constant_params[1] = pixel_size
        self.constant_params[2] = res
        self.constant_params[3] = noise_radius_mask

    def stack_batch(self, batch_images: ArrayLike, batch_params: ArrayLike):
        """
        Stack a batch of images and their parameters.

        Parameters
        ----------
        batch_images : ArrayLike
            Array of images
        batch_params : ArrayLike
            Array of parameters

        Returns
        -------
        None
        """
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

    def load_(self, fname: str):
        """
        Load images and parameters from a file.

        Parameters
        ----------
        fname : str
            Path to file containing images and parameters

        Returns
        -------
        None
        """
        numpy_file = np.load_(fname)
        self.images = numpy_file["images"]
        self.variable_params = numpy_file["variable_params"]
        self.constant_params = numpy_file["constant_params"]

    def save(self, fname: str):
        """
        Save images and parameters to a file.

        Parameters
        ----------
        fname : str
            Path to file to save images and parameters

        Returns
        -------
        None
        """
        np.savez(
            fname,
            images=self.images,
            variable_params=self.variable_params,
            constant_params=self.constant_params,
        )
