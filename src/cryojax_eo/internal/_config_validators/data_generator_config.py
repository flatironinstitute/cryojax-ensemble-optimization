import warnings
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_serializer,
    field_validator,
)

from .utils import _validate_files_with_type


class DatasetSimulatorConfigAtomicModels(BaseModel, extra="forbid"):
    """
    Parameter for loading the atomic models parameters used
    in the data generation pipeline.
    """

    path_to_atomic_models: str | list[FilePath] = Field(
        description="Path to the atomic models directory. "
        + "If a pattern is provided, all files matching the pattern will be used."
    )

    atomic_models_probabilities: PositiveFloat | list[PositiveFloat] = Field(
        description="Probabilstic weights for each model. Will be normalized to sum to 1."
    )

    loads_b_factors: bool = Field(
        default=False,
        description="Whether to load the B-factors from the PDB file. "
        + "Only used if the atomic model is in PDB format. "
        + "Otherwise it will be ignored.",
    )

    atom_selection: str = Field(
        default="all",
        description="Selection string for the atoms to use. "
        + "Only used if the atomic model is in PDB format. "
        + "Otherwise it will be ignored.",
    )

    @field_serializer("atomic_models_probabilities")
    def serialize_atomic_model_probabilities(self, v):
        if isinstance(v, int):
            v = [v]
        v = jnp.array(v)
        return v / jnp.sum(v)

    @field_serializer("path_to_atomic_models")
    def serialize_path_to_atomic_models(self, v):
        return _validate_files_with_type(v, file_types=[".pdb", ".npz"])


class DatasetSimulatorConfigRotations(BaseModel, extra="forbid"):
    rotation_distribution: Literal["uniform", "non-uniform"] = Field(
        default="uniform",
        description="Distribution of particle rotations. "
        + "'uniform' for uniform rotations; "
        + "'non-uniform' for mixture of uniform and von Mises-Fisher rotations.",
    )
    vmf_kappa: PositiveFloat | list[PositiveFloat] = Field(
        default=1.0,
        description="Kappa parameter for the von Mises-Fisher distribution (1/variance)"
        + "Only used if `rotation_distribution` is 'non-uniform'.",
    )
    vmf_mu: list[float] = Field(
        default=[0.0, 0.0, 1.0],
        description="Mean direction for the von Mises-Fisher distribution. "
        + "Only used if `rotation_distribution` is 'non-uniform'.",
    )
    vmf_alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Mixture weight between vMF and uniform rotations in [0, 1] - (1 means only vMF, 0 means only uniform)"
        + "Only used if `rotation_distribution` is 'non-uniform'.",
    )

    @field_validator("vmf_mu")
    @classmethod
    def validate_vmf_mu(cls, v):
        arr = jnp.asarray(v, dtype=float)
        if arr.shape != (3,):
            raise ValueError("`rotations.vmf_mu` must be a 3-vector.")
        norm = jnp.linalg.norm(arr)
        if norm <= 0:
            raise ValueError("`rotations.vmf_mu` must be non-zero.")
        return arr

    @field_serializer("vmf_mu")
    def serialize_vmf_mu(self, v):
        arr = jnp.asarray(v, dtype=float)
        return arr / jnp.linalg.norm(arr)


class DatasetSimulatorConfig(BaseModel, extra="forbid"):
    """
    Parameters for the data generation pipeline.

    If an item can be either a list or a single value, the list will be used as the range for random data generation. For example, if `offset_x_in_angstroms` is defined as `[0, 10]`, the offset in the x direction will be randomly generated between 0 and 10 for each image. If a single value is provided, the same value will be used for all images.

    """  # noqa

    # Experiment setup
    number_of_images: PositiveInt = Field(description="Number of images to generate.")

    data_sign: Literal["dark-on-light", "light-on-dark"] = Field(
        default="dark-on-light",
        description="Sign convention for the data. "
        + "'dark-on-light' means that the particles "
        + "are dark on a light background (default). "
        + "'light-on-dark' means that the particles "
        + "are light on a dark background.",
    )
    # Instrument
    pixel_size: PositiveFloat = Field(description="Pixel size in Angstroms.")
    box_size: PositiveInt = Field(description="Size of the simulation box in pixels.")
    pad_scale: PositiveInt = Field(
        1, description="Factor to scale the box size for padding."
    )
    voltage_in_kilovolts: PositiveFloat = Field(
        300.0, description="Voltage in kilovolts."
    )

    # Pose
    offset_x_in_angstroms: float | list[float] = Field(
        0.0, description="Offset in x direction in Angstroms."
    )
    offset_y_in_angstroms: float | list[float] = Field(
        0.0, description="Offset in y direction in Angstroms."
    )

    rotations: dict = Field(
        default_factory=lambda _ : dict(DatasetSimulatorConfigRotations().model_dump()),
        description="Rotation sampling configuration. Parsed by "
        + "`DatasetSimulatorConfigRotations`.",
    )

    # Transfer Theory
    defocus_in_angstroms: PositiveFloat | list[PositiveFloat] = Field(
        0.0, description="Defocus in Angstroms."
    )
    astigmatism_in_angstroms: float | list[float] = Field(
        0.0, description="Astigmatism in Angstroms."
    )
    astigmatism_angle_in_degrees: float | list[float] = Field(
        0.0, description="Astigmatism angle in degrees."
    )
    phase_shift: float | list[float] = Field(0.0, description="Phase shift in radians.")
    amplitude_contrast_ratio: PositiveFloat = Field(
        0.1, description="Amplitude contrast ratio."
    )
    spherical_aberration_in_mm: PositiveFloat = Field(
        2.7, description="Microscope spherical aberration in mm."
    )
    ctf_scale_factor: PositiveFloat = Field(1.0, description="CTF scale factor.")
    envelope_b_factor: float | list[float] = Field(
        0.0, description="Envelope B-factor in Angstroms^2."
    )

    # Noise and randomness
    noise_snr: PositiveFloat | list[PositiveFloat] = Field(
        description="Signal to noise ratio."
    )
    mask_radius: PositiveFloat | None = Field(
        default=None,
        description="Radius for a circular cryojax Mask."
        + " This is used to compute the variance of the signal, "
        + "and then define the noise variance through the SNR. "
        + "If None, will be set to box_size // 3.",
    )
    mask_rolloff_width: PositiveFloat = Field(
        default=0.0, description="Width of the rolloff for the mask. "
    )

    rng_seed: int = Field(0, description="Seed for random number generation.")

    # Atomic modelss
    atomic_models_params: dict = Field(
        description="Parameters for the atomic models. This is a dictionary "
        + "formatted by the `DatasetSimulatorConfigAtomicModels` class."
    )

    # I/O
    path_to_relion_project: Path = Field(
        description="Path to the RELION project directory."
    )
    path_to_starfile: Path = Field(description="Path to the RELION star file.")
    images_per_file: PositiveInt = Field(description="Images per .mrcs.")
    batch_size_for_generation: PositiveInt = Field(
        default=1,
        description="Batch size for the data generation. "
        + "This is used to generate the data in batches.",
    )
    overwrite: bool = Field(False, description="Overwrite existing files if True.")

    @field_validator("atomic_models_params")
    @classmethod
    def validate_atomic_models_params(cls, v):
        return dict(DatasetSimulatorConfigAtomicModels(**v).model_dump())

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

    @field_serializer("rotations")
    @classmethod
    def serialize_rotations(cls, v):
        return dict(DatasetSimulatorConfigRotations(**v).model_dump())

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

    @field_serializer("astigmatism_angle_in_degrees")
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

    @field_serializer("envelope_b_factor")
    def serialize_envelope_b_factor(self, v):
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

    @field_serializer("mask_radius")
    def serialize_noise_radius_mask(self, v):
        if v is None:
            v = self.box_size // 3

        elif v > self.box_size:
            warnings.warn(
                "Noise radius mask is greater than box size. Setting to box size."
            )
            v = self.box_size

        return v
