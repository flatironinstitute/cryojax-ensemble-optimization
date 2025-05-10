import glob
import os
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


class _EnsembleOptimizerValidator(BaseModel, extra="forbid"):
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


class _PipelineMDSamplerAllAtomValidator(BaseModel, extra="forbid"):
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


class _PipelineMDSamplerCGValidator(BaseModel, extra="forbid"):
    step_type: str = "mdsampler"
    mode: str = "cg"
    n_steps: int
    mdsampler_force_constant: float
    checkpoint_fnames: Optional[Union[str, List[str]]] = None
    platform: str = "CPU"
    platform_properties: dict = {"Threads": None}
    top_file: str
    epsilon_r: float = 15.0

    @field_validator("n_steps")
    @classmethod
    def validate_n_steps(cls, v):
        if v <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return v

    @field_validator("mdsampler_force_constant")
    @classmethod
    def validate_mdsampler_force_constant(cls, v):
        if v <= 0:
            raise ValueError("Force constant must be greater than 0")
        return v

    # @field_validator("checkpoint_fnames")
    # @classmethod
    # def validate_checkpoint_fnames(cls, v):
    #     if v is not None:
    #         if "*" in v:
    #             checkpoint_fnames = natsorted(glob.glob(v))
    #             if len(checkpoint_fnames) == 0:
    #                 raise FileNotFoundError(f"No files found with pattern {v}")
    #         else:
    #             checkpoint_fnames = [v]
    #             if not os.path.exists(checkpoint_fnames[0]):
    #                 raise FileNotFoundError(
    #                     f"Checkpoint {checkpoint_fnames[0]} does not exist."
    #                 )

    #         v = checkpoint_fnames
    #     return v

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, v):
        if v not in ["CPU", "CUDA"]:
            raise ValueError("Invalid platform, must be 'CPU' or 'CUDA'")
        return v

    @field_validator("platform_properties")
    @classmethod
    def validate_platform_properties(cls, v):
        if v["Threads"] is not None:
            if v["Threads"] <= 0:
                raise ValueError("Number of threads must be greater than 0")
        return v

    @field_validator("epsilon_r")
    @classmethod
    def validate_epsilon_r(cls, v):
        if v <= 0:
            raise ValueError("Epsilon_r must be greater than 0")
        return v

    @field_validator("top_file")
    @classmethod
    def validate_top_file(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Topology file {v} does not exist.")
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
        return dict(_EnsembleOptimizerValidator(**values).model_dump())

    @field_validator("md_sampler_config")
    @classmethod
    def validate_md_sampler_config(cls, values):
        return dict(_PipelineMDSamplerAllAtomValidator(**values).model_dump())

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

