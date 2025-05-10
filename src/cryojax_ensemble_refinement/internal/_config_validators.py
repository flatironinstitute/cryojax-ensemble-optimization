import warnings
import numpy as np
import os
from natsort import natsorted
from functools import partial
import glob
from typing import List, Union, Optional, Dict
import mdtraj
from typing_extensions import Annotated, Literal
from pydantic import (
    BaseModel,
    model_validator,
    field_serializer,
    Field,
    PositiveInt,
    PositiveFloat,
    DirectoryPath,
    FilePath,
    field_validator,
    AfterValidator,
)

from pathlib import Path

import jax.numpy as jnp


def _is_file_type(filename: str, file_type: str) -> str:
    """
    Check if the file is a PDB file.
    """
    if not filename.endswith(f".{file_type}"):
        raise ValueError(f"File {filename} is not a {file_type} file.")
    return filename


def _contains_file_type(
    directory_path: DirectoryPath, file_type: str | List[str]
) -> DirectoryPath:
    if isinstance(file_type, str):
        file_type = [file_type]

    failing_types = []
    for ftype in file_type:
        files_in_directory = glob.glob(os.joint(directory_path, f"*.{ftype}"))
        if len(files_in_directory) == 0:
            failing_types.append(ftype)

    if len(failing_types) > 0:
        raise ValueError(
            f"Directory {directory_path} does not contain any files "
            + f"of type(s): {', '.join(failing_types)}"
        )

    return directory_path


def _validate_file_names_in_dir(
    file_names: Union[str, List[str]], base_directory: DirectoryPath
) -> List[FilePath]:
    if isinstance(file_names, str):
        if "*" in file_names:
            file_names = os.path.join(base_directory, file_names)
            file_names = natsorted(glob.glob(file_names))
        else:
            file_names = [os.path.join(base_directory, file_names)]
    elif isinstance(file_names, list):
        file_names = [os.path.join(base_directory, fname) for fname in file_names]

    file_names = [Path(fname) for fname in file_names]
    assert all([fname.exists() for fname in file_names]), (
        f"Some files do not exist in the directory {base_directory}. "
        + f"Files: {', '.join([str(fname) for fname in file_names if not fname.exists()])}"
    )
    return file_names


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
    number_of_images: PositiveInt = Field(description="Number of images to generate.")

    weights_models: Union[PositiveFloat, List[PositiveFloat]] = Field(
        description="Probabilstic weights for each model. Will be normalized to sum to 1."
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
    astigmatism_angle: Union[float, List[float]] = Field(
        0.0, description="Astigmatism angle in degrees."
    )
    phase_shift: Union[float, List[float]] = Field(
        0.0, description="Phase shift in radians."
    )
    amplitude_contrast_ratio: PositiveFloat = Field(
        1.0, description="Amplitude contrast ratio."
    )
    spherical_aberration_in_mm: PositiveFloat = Field(
        2.7, description="Microscope spherical aberration in mm."
    )
    ctf_scale_factor: PositiveFloat = Field(1.0, description="CTF scale factor.")
    envelope_bfactor: Union[float, List[float]] = Field(
        0.0, description="Envelope B-factor in Angstroms^2."
    )

    # Random stuff
    noise_snr: Union[PositiveFloat, List[PositiveFloat]] = Field(
        description="Signal to noise ratio."
    )
    noise_radius_mask: Optional[PositiveFloat] = Field(
        None,
        description="Radius of the mask for noise generation."
        + " This is used to compute the variance of the signal, "
        + "and then define the noise variance through the SNR",
    )
    rng_seed: int = Field(0, description="Seed for random number generation.")

    # I/O
    path_to_models: DirectoryPath = Field(
        description="Path to the directory containing the atomic models for image generation."  # noqa
    )
    models_fnames: Union[str, List[str]] = Field(
        description="Filename of the atomic model(s) to use for image generation."
        + "If a pattern is provided, all files matching the pattern will be used."
        + " The atomic models should be in path_to_models."
    )
    path_to_relion_project: Path = Field(
        description="Path to the RELION project directory."
    )
    path_to_starfile: Path = Field(description="Path to the RELION star file.")
    batch_size: PositiveInt = Field(description="Batch size for data generation.")
    overwrite: bool = Field(False, description="Overwrite existing files if True.")

    @model_validator(mode="after")
    def validate_config_generator_req_values(self):
        # Noise
        if self.noise_radius_mask is not None:
            if self.noise_radius_mask > self.box_size:
                warnings.warn(
                    "Noise radius mask is greater than box size. Setting to box size."
                )
                self.noise_radius_mask = self.box_size

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
        return _validate_file_names_in_dir(v, self.path_to_models)

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
        description="Initial weights for the models. If None, will be set to uniform distribution.",
    )
    noise_variance: PositiveFloat = Field(
        description="Variance of the noise to be added to the gradients.",
    )

    @field_serializer("init_weights")
    def serialize_init_weights(self, v):
        if v is not None:
            v = jnp.array(v)
            v = v / jnp.sum(v)

        return v


class cryojaxERConfigMDConfig(BaseModel, extra="forbid"):
    mode: Literal["all-atom"] = Field(
        default="all-atom", description="Mode of the MD sampler."
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
    platform_properties: Dict = Field(
        default={"Threads": None}, description="Platform properties for OpenMM."
    )

    @field_validator("platform_properties")
    @classmethod
    def validate_platform_properties(cls, v):
        if "Threads" in v:
            if v["Threads"] is not None:
                assert v["Threads"] > 0, "Number of threads must be greater than 0"
        return v


class cryojaxERConfig(BaseModel, extra="forbid"):
    experiment_name: str = Field(
        description="Name of the experiment. Used to create the output directory."
    )
    # I/O
    path_to_models_and_chkpoints: Annotated[
        DirectoryPath,
        AfterValidator(partial(_contains_file_type, file_type=[".pdb", ".chk"])),
    ] = Field(description="Path to the directory containing the models and checkpoints.")

    models_fnames: Union[str, List[str]] = Field(
        description="List of files containing the atomic models. "
        + "path_to_models_and_chkpoints. "
        + "If a string, it must contain a glob pattern. "
        + "Files must in .pdb format."
    )
    checkpoints_fnames: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="List of files containing the openmm checkpoints in "
        + "path_to_models_and_chkpoints. "
        + "If a string, it must contain a glob pattern. "
        + "Files must in .chk format.",
    )
    ref_model_fname: Annotated[
        str, AfterValidator(partial(_is_file_type, file_type="pdb"))
    ] = Field(
        description="File containing the reference model in path_to_models_and_chkpoints."
    )
    path_to_starfile: FilePath = Field(
        description="Path to the starfile containing the particle information."
    )
    path_to_relion_project: DirectoryPath = Field(
        description="Path to the relion project directory."
    )
    output_path: Path = Field(
        description="Path to the output directory. "
        + "If it does not exist, it will be created.",
    )

    # Pipeline
    md_sampler_config: dict
    ensemble_optimizer_config: dict

    # Optimization
    n_steps: PositiveInt = Field(
        description="Number of steps of cryoJAX ensemble refinement to run."
    )

    # Image and MD stuff
    atom_list_filter: Optional[str] = None
    rng_seed: int = Field(default=0, description="Random seed.")

    @model_validator(mode="after")
    def validate_config(self):
        self.models_fnames = _validate_file_names_in_dir(
            self.models_fnames, self.path_to_models_and_chkpoints
        )

        if self.checkpoints_fnames is not None:
            self.checkpoints_fnames = _validate_file_names_in_dir(
                self.checkpoints_fnames, self.path_to_models_and_chkpoints
            )

        ref_model_path = Path(
            os.path.join(self.path_to_models_and_chkpoints, self.ref_model_fname)
        )
        assert ref_model_path.exists(), (
            f"Reference model {ref_model_path} does not exist."
        )

        if self.atom_list_filter is not None:
            try:
                mdtraj.load(self.ref_model_fname).topology.select(self.atom_list_filter)
            except ValueError:
                raise ValueError(f"Invalid atom list filter {self.atom_list_filter}")

        return self

    @field_validator("ensemble_optimizer_config")
    @classmethod
    def validate_ensemble_opt_config(cls, values):
        return dict(cryojaxERConfigOptimizationConfig(**values).model_dump())

    @field_validator("md_sampler_config")
    @classmethod
    def validate_md_sampler_config(cls, values):
        return dict(cryojaxERConfigMDConfig(**values).model_dump())

    @field_validator("output_path")
    @classmethod
    def serialize_output_path(cls, v):
        if not os.path.exists(v):
            new_path = os.path.join(v, "Job001")

        else:
            # list all subdirectories
            subdirs = [f for f in os.listdir(v) if os.path.isdir(os.path.join(v, f))]

            # get the last job number
            job_numbers = []
            for subdir in subdirs:
                if subdir.startswith("Job"):
                    job_numbers.append(int(subdir[3:]))
            job_numbers = sorted(job_numbers)
            if len(job_numbers) == 0:
                new_path = os.path.join(v, "Job001")
            else:
                new_path = os.path.join(v, f"Job{job_numbers[-1] + 1:03d}")

        return new_path
