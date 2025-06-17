import glob
import os
import warnings
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union
from typing_extensions import Annotated, Literal

import jax.numpy as jnp
import mdtraj
from natsort import natsorted
from pydantic import (
    AfterValidator,
    BaseModel,
    DirectoryPath,
    Field,
    field_serializer,
    field_validator,
    FilePath,
    model_validator,
    PositiveFloat,
    PositiveInt,
)


def _validate_file_with_type(filename: str, file_type: str) -> str:
    """
    Check if the file is a PDB file.
    """
    assert Path(filename).exists(), f"File {filename} does not exist."
    assert Path(filename).is_file(), f"Path {filename} is not a file."

    if not Path(filename).suffix == f".{file_type}":
        raise ValueError(f"File {filename} is not a {file_type} file.")
    return filename


# Might be useful, and I don't want to figure it out again
# TODO: remove if not needed
def _contains_file_type(
    directory_path: DirectoryPath, file_type: str | List[str]
) -> DirectoryPath:
    if isinstance(file_type, str):
        file_type = [file_type]

    failing_types = []
    for ftype in file_type:
        files_in_directory = glob.glob(os.path.join(directory_path, f"*.{ftype}"))
        if len(files_in_directory) == 0:
            failing_types.append(ftype)

    if len(failing_types) > 0:
        raise ValueError(
            f"Directory {directory_path} does not contain any files "
            + f"of type(s): {', '.join(failing_types)}"
        )

    return directory_path


def _validate_files_with_type(
    path_to_files: Union[str, List[FilePath]], file_type: str
) -> List[str]:
    if isinstance(path_to_files, str):
        if "*" in path_to_files:
            output = [Path(f) for f in natsorted(glob.glob(path_to_files))]
        elif Path(path_to_files).is_file():
            output = [Path(path_to_files)]
        else:
            raise ValueError(
                f"Path {path_to_files} is not a file or does not use * wild card."
            )
    elif isinstance(path_to_files, list):
        output = [Path(f) for f in path_to_files]

    for f in output:
        assert f.exists(), f"{f} does not exist."
        assert f.is_file(), f"{f} is not a file."
        assert f.suffix == f".{file_type}", (
            f"{f} is not a {file_type} file. "
            + f"Valid file types are: {', '.join([f'.{file_type}'])}"
        )
    return [str(f) for f in output]


class DatasetGeneratorConfigAtomicModels(BaseModel, extra="forbid"):
    """
    Parameter for loading the atomic models parameters used
    in the data generation pipeline.
    """

    path_to_atomic_models: Union[str, List[FilePath]] = Field(
        description="Path to the atomic models directory. "
        + "If a pattern is provided, all files matching the pattern will be used."
    )

    atomic_models_probabilities: Union[PositiveFloat, List[PositiveFloat]] = Field(
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
        return _validate_files_with_type(v, file_type="pdb")


class DatasetGeneratorConfig(BaseModel, extra="forbid"):
    """
    Parameters for the data generation pipeline.

    If an item can be either a list or a single value, the list will be used as the range for random data generation. For example, if `offset_x_in_angstroms` is defined as `[0, 10]`, the offset in the x direction will be randomly generated between 0 and 10 for each image. If a single value is provided, the same value will be used for all images.

    """  # noqa

    # Experiment setup
    number_of_images: PositiveInt = Field(description="Number of images to generate.")

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
    offset_x_in_angstroms: Union[float, List[float]] = Field(
        0.0, description="Offset in x direction in Angstroms."
    )
    offset_y_in_angstroms: Union[float, List[float]] = Field(
        0.0, description="Offset in y direction in Angstroms."
    )

    # Transfer Theory
    defocus_in_angstroms: Union[PositiveFloat, List[PositiveFloat]] = Field(
        0.0, description="Defocus in Angstroms."
    )
    astigmatism_in_angstroms: Union[float, List[float]] = Field(
        0.0, description="Astigmatism in Angstroms."
    )
    astigmatism_angle_in_degrees: Union[float, List[float]] = Field(
        0.0, description="Astigmatism angle in degrees."
    )
    phase_shift: Union[float, List[float]] = Field(
        0.0, description="Phase shift in radians."
    )
    amplitude_contrast_ratio: PositiveFloat = Field(
        0.1, description="Amplitude contrast ratio."
    )
    spherical_aberration_in_mm: PositiveFloat = Field(
        2.7, description="Microscope spherical aberration in mm."
    )
    ctf_scale_factor: PositiveFloat = Field(1.0, description="CTF scale factor.")
    envelope_b_factor: Union[float, List[float]] = Field(
        0.0, description="Envelope B-factor in Angstroms^2."
    )

    # Noise and randomness
    noise_snr: Union[PositiveFloat, List[PositiveFloat]] = Field(
        description="Signal to noise ratio."
    )
    mask_radius: Optional[PositiveFloat] = Field(
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
        + "formatted by the `DatasetGeneratorConfigAtomicModels` class."
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
        return dict(DatasetGeneratorConfigAtomicModels(**v).model_dump())

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


class cryojaxERConfigOptimizationConfig(BaseModel, extra="forbid"):
    n_steps: PositiveInt = Field(
        default=1, description="Number of steps for the optimization process."
    )
    step_size: PositiveFloat = Field(
        description="Step size in Angstroms for the optimization process."
    )
    batch_size: PositiveInt = Field(
        description="Batch size for SGD",
    )
    init_weights: Optional[List[float]] = Field(
        default=None,
        description="Initial weights for the models. "
        "If None, will be set to uniform distribution.",
    )
    noise_variance: Optional[PositiveFloat] = Field(
        default=None,
        description="Variance of the noise to be added to the gradients.",
    )

    image_to_walker_log_likelihood_fn: Literal[
        "iso_gaussian", "iso_gaussian_var_marg"
    ] = Field(
        default="iso_gaussian_var_marg",
        description="Type of likelihood function to use. "
        + "Must be 'iso_gaussian' or 'iso_gaussian_var_marg'.",
    )

    @field_serializer("init_weights")
    def serialize_init_weights(self, v):
        if v is not None:
            v = jnp.array(v)
            v = v / jnp.sum(v)

        return v


class cryojaxERConfigMDConfig(BaseModel, extra="forbid"):
    projector_mode: Literal["openmm"] = Field(
        default="openmm", description="Type of projection method. Default is openmm."
    )
    n_steps: PositiveInt = Field(
        description="Number of steps for the MD sampler. Must be greater than 0."
    )
    bias_constant_in_units: PositiveFloat = Field(
        description="Force constant for the MD sampler. Must be greater than 0."
    )
    platform: Literal["CPU", "CUDA", "OpenCL"] = Field(
        default="CPU",
        description="Platform to use for the MD sampler. "
        + "Must be 'CPU', 'CUDA', or 'OpenCL'.",
    )
    platform_properties: Dict[str, str | None] = Field(
        default={"Threads": None}, description="Platform properties for OpenMM."
    )

    path_to_initial_states: Optional[str | List[FilePath]] = Field(
        default=None,
        description="Path to the initial states. "
        + "If None, will be set to the path to the atomic models.",
    )

    @field_validator("platform_properties")
    @classmethod
    def validate_platform_properties(cls, v):
        if "Threads" in v:
            if v["Threads"] is not None:
                assert int(v["Threads"]) > 0, "Number of threads must be greater than 0"
        return v

    @field_validator("path_to_initial_states")
    @classmethod
    def validate_path_to_initial_states(cls, v):
        if v is None:
            return v
        else:
            return _validate_files_with_type(v, file_type="xml")


class cryojaxERConfig(BaseModel, extra="forbid"):
    # I/O

    path_to_atomic_models: Union[str, List[FilePath]] = Field(
        description="Path to the atomic models directory. "
        + "If a pattern is provided, all files matching the pattern will be used."
    )
    path_to_reference_model: Annotated[
        str, AfterValidator(partial(_validate_file_with_type, file_type="pdb"))
    ] = Field(
        description="Path to the reference model. "
        + "This model should be aligned to the cryo-EM particles, "
        + " and will be used for alignment."
    )
    path_to_starfile: FilePath = Field(
        description="Path to the starfile containing the particle information."
    )
    path_to_relion_project: DirectoryPath = Field(
        description="Path to the relion project directory."
    )
    loads_envelope: bool = Field(
        description="Whether to load the envelope from the starfile. "
    )

    path_to_output: Path = Field(
        description="Path to the output directory. "
        + "If it does not exist, it will be created.",
    )

    # Pipeline
    projector_params: Dict = Field(
        description="Parameters for the physics-based ensemble projector. "
        + "This is a dictionary formatted by the `cryojaxERConfigMDConfig` class."
    )
    likelihood_optimizer_params: Dict = Field(
        description="Parameters for the ensemble optimizer. "
        + "This is a dictionary formatted by "
        + "the `cryojaxERConfigOptimizationConfig` class."
    )

    # Optimization
    n_steps: PositiveInt = Field(
        description="Number of steps of cryoJAX ensemble refinement to run."
    )

    # Miscellaneous
    atom_selection: str = Field(
        default="all",
        description="Selection string for the atoms to use. "
        + "Only used if the atomic model is in PDB format. "
        + "Otherwise it will be ignored.",
    )

    loads_b_factors: bool = Field(
        default=False,
        description="Whether to load the thermal b-factors from the PDB file. "
        + "Only used if the atomic model is in PDB format. "
        + "Otherwise it will be ignored."
        + "Also known as Debye-Waller factors.",
    )
    rng_seed: int = Field(default=0, description="Random seed.")

    @model_validator(mode="after")
    def validate_config(self):
        if self.atom_selection is not None:
            try:
                mdtraj.load(self.path_to_reference_model).topology.select(
                    self.atom_selection
                )
            except Exception as e:
                raise ValueError(
                    f"Invalid atom list filter {self.atom_selection}. Error: {e}"
                )

        if self.projector_params["path_to_initial_states"] is not None:
            n_initial_states = len(self.projector_params["path_to_initial_states"])
            n_atomic_models = len(self.path_to_atomic_models)
            assert n_atomic_models == n_initial_states, (
                f"Number of initial states {n_initial_states} "
                + f"does not match number of atomic models {n_atomic_models}."
            )
        return self

    @field_validator("path_to_atomic_models")
    @classmethod
    def validate_path_to_atomic_models(cls, v):
        return _validate_files_with_type(v, file_type="pdb")

    @field_validator("path_to_reference_model")
    @classmethod
    def validate_path_to_reference_model(cls, v):
        return _validate_file_with_type(v, file_type="pdb")

    @field_validator("path_to_starfile")
    @classmethod
    def validate_path_to_starfile(cls, v):
        return _validate_file_with_type(v, file_type="star")

    @field_validator("likelihood_optimizer_params")
    @classmethod
    def validate_ensemble_opt_config(cls, values):
        return dict(cryojaxERConfigOptimizationConfig(**values).model_dump())

    @field_validator("projector_params")
    @classmethod
    def validate_md_sampler_config(cls, values):
        return dict(cryojaxERConfigMDConfig(**values).model_dump())

    # @field_validator("path_to_output")
    # @classmethod
    # def serialize_output_path(cls, v):
    #     if not os.path.exists(v):
    #         new_path = os.path.join(v, "Job001")

    #     else:
    #         # list all subdirectories
    #         subdirs = [f for f in os.listdir(v) if os.path.isdir(os.path.join(v, f))]

    #         # get the last job number
    #         job_numbers = []
    #         for subdir in subdirs:
    #             if subdir.startswith("Job"):
    #                 job_numbers.append(int(subdir[3:]))
    #         job_numbers = sorted(job_numbers)
    #         if len(job_numbers) == 0:
    #             new_path = os.path.join(v, "Job001")
    #         else:
    #             new_path = os.path.join(v, f"Job{job_numbers[-1] + 1:03d}")

    #     return new_path
