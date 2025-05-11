import glob
import os
from typing import List, Optional, Union

import jax.numpy as jnp
import numpy as np
from natsort import natsorted
from pydantic import BaseModel, Field, field_serializer, model_validator


class GeneratorConfig(BaseModel, extra="forbid"):
    """
    Parameters for the data generation pipeline.

    The experiment name is simply for naming purposes.

    If an item can be either a list or a single value, the list will be used as the range for random data generation. For example, if `offset_x_in_angstroms` is defined as `[0, 10]`, the offset in the x direction will be randomly generated between 0 and 10 for each image. If a single value is provided, the same value will be used for all images.

    """  # noqa

    # Experiment setup
    experiment_name: str = Field(
        description="Name of the experiment. Used to name the output files."
    )
    number_of_images: int = Field(description="Number of images to generate.")
    weights_models: Union[float, List[float]] = Field(
        description="Probabilstic weights for each model. Will be normalized to sum to 1."
    )

    # Instrument
    pixel_size: float = Field(description="Pixel size in Angstroms.")
    box_size: int = Field(description="Size of the simulation box in pixels.")
    pad_scale: int = Field(1, description="Factor to scale the box size for padding.")
    voltage_in_kilovolts: float = Field(300.0, description="Voltage in kilovolts.")

    # Pose
    offset_x_in_angstroms: Union[float, List[float]] = Field(
        0.0, description="Offset in x direction in Angstroms."
    )
    offset_y_in_angstroms: Union[float, List[float]] = Field(
        0.0, description="Offset in y direction in Angstroms."
    )

    # Transfer Theory
    defocus_in_angstroms: Union[float, List[float]] = Field(
        0.0, description="Defocus in Angstroms."
    )
    astigmatism_in_angstroms: Union[float, List[float]] = Field(
        0.0, description="Astigmatism in Angstroms."
    )
    astigmatism_angle: Union[float, List[float]] = Field(
        0.0, description="Astigmatism angle in degrees."
    )
    phase_shift: Union[float, List[float]] = Field(
        0.0, description="Phase shift in radians."
    )
    amplitude_contrast_ratio: float = Field(1.0, description="Amplitude contrast ratio.")
    spherical_aberration_in_mm: float = Field(
        2.7, description="Microscope spherical aberration in mm."
    )
    ctf_scale_factor: float = Field(1.0, description="CTF scale factor.")
    envelope_bfactor: Union[float, List[float]] = Field(
        0.0, description="Envelope B-factor in Angstroms^2."
    )

    # Random stuff
    noise_snr: Union[float, List[float]] = Field(description="Signal to noise ratio.")
    noise_radius_mask: Optional[float] = Field(
        None,
        description="Radius of the mask for noise generation. This is used to compute the variance of the signal, and then define the noise variance through the SNR",
    )
    rng_seed: int = Field(0, description="Seed for random number generation.")

    # I/O
    path_to_models: str = Field(
        description="Path to the directory containing the atomic models for image generation."
    )
    models_fnames: Union[str, List[str]] = Field(
        description="Filename of the atomic model(s) to use for image generation. If a pattern is provided, all files matching the pattern will be used. The atomic models should be in path_to_models."
    )
    path_to_relion_project: str = Field(
        description="Path to the RELION project directory."
    )
    path_to_starfile: str = Field(description="Path to the RELION star file.")
    batch_size: int = Field(description="Batch size for data generation.")
    overwrite: bool = Field(False, description="Overwrite existing files if True.")

    @model_validator(mode="after")
    def validate_config_generator_req_values(self):
        # Experiment setup
        if np.all(np.array(self.weights_models) <= 0):
            raise ValueError("Particles per model must be greater than 0")

        if self.number_of_images <= 0:
            raise ValueError("Number of images must be greater than 0")

        # Instrument
        if self.box_size <= 0:
            raise ValueError("Box size must be greater than 0")

        if self.pixel_size <= 0:
            raise ValueError("Pixel size must be greater than 0")

        if self.voltage_in_kilovolts <= 0:
            raise ValueError("Voltage must be greater than 0")

        # Transfer Theory
        if np.all(np.array(self.defocus_in_angstroms) < 0):
            raise ValueError("Defocus u must be greater or equal to 0")

        if np.all(np.array(self.astigmatism_in_angstroms) < 0):
            raise ValueError("Defocus v must be greater or equal to 0")

        if np.all(np.array(self.noise_snr) <= 0):
            raise ValueError("Noise snr must be greater than 0")

        if self.amplitude_contrast_ratio < 0 or self.amplitude_contrast_ratio > 1:
            raise ValueError("Amplitude contrast must be between 0 and 1")

        if self.spherical_aberration_in_mm <= 0:
            raise ValueError("Spherical aberration must be greater than 0")

        # Noise
        if self.noise_radius_mask is not None:
            if self.noise_radius_mask <= 0:
                raise ValueError("Noise raidus mask must be greater than 0")

            if self.noise_radius_mask > self.box_size:
                raise ValueError(
                    "Noise raidus mask must be less than half of the box size"
                )

        # I/O
        if self.batch_size <= 0:
            raise ValueError("Batch size must be greater than 0")

        if not os.path.exists(self.path_to_models):
            raise FileNotFoundError(
                f"Working directory {self.path_to_models} does not exist."
            )

        if isinstance(self.models_fnames, str):
            if "*" in self.models_fnames:
                models_fnames = os.path.join(self.path_to_models, self.models_fnames)
                models_fnames = natsorted(glob.glob(models_fnames))
                if len(models_fnames) == 0:
                    raise FileNotFoundError(
                        f"No files found with pattern {self.models_fnames}"
                    )
            else:
                models_fnames = [self.models_fnames]

        elif isinstance(self.models_fnames, list):
            models_fnames = self.models_fnames

        else:
            raise ValueError("models_fnames must be a string or a list of strings.")

        for i in range(len(models_fnames)):
            models_fnames[i] = os.path.join(self.path_to_models, models_fnames[i])
            if not os.path.exists(models_fnames[i]):
                raise FileNotFoundError(f"Model {models_fnames[i]} does not exist.")

        return self

    @field_serializer("weights_models")
    def serialize_weights_models(self, v):
        if isinstance(v, int):
            v = [v]
        v = jnp.array(v)
        return v / jnp.sum(v)

    @field_serializer("models_fnames")
    def serialize_models_fname(self, v):
        if isinstance(v, str):
            if "*" in v:
                v = os.path.join(self.path_to_models, v)
                v = natsorted(glob.glob(v))
            else:
                v = [os.path.join(self.path_to_models, v)]
        elif isinstance(v, list):
            v = [os.path.join(self.path_to_models, i) for i in v]
        return v

    @field_serializer("offset_x_in_angstroms")
    def serialize_offset_x_in_angstroms(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("offset_y_in_angstroms")
    def serialize_offset_y_in_angstroms(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("defocus_in_angstroms")
    def serialize_defocus_in_angstroms(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("astigmatism_in_angstroms")
    def serialize_astigmatism_in_angstroms(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("astigmatism_angle")
    def serialize_astigmatism_angle(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("phase_shift")
    def serialize_phase_shift(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("envelope_bfactor")
    def serialize_envelope_bfactor(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("noise_snr")
    def serialize_noise_snr(self, v):
        if isinstance(v, float):
            v = jnp.array([v, v])
        else:
            v = jnp.array(v)
        return v

    @field_serializer("noise_radius_mask")
    def serialize_noise_radius_mask(self, v):
        if v is None:
            v = self.box_size // 3
        return v
