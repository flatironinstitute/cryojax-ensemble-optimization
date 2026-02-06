from functools import partial
from pathlib import Path
from typing import Annotated, Literal

import jax.numpy as jnp
import mdtraj
from pydantic import (
    AfterValidator,
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_serializer,
    field_validator,
    model_validator,
)

from .utils import _validate_file_with_type, _validate_files_with_type


class EnsOptMDConfigOptimizationConfig(BaseModel, extra="forbid"):
    n_steps: PositiveInt = Field(
        default=1, description="Number of steps for the optimization process."
    )
    step_size: PositiveFloat = Field(
        description="Step size in Angstroms for the optimization process."
    )
    batch_size: PositiveInt = Field(
        description="Batch size for SGD",
    )
    initial_weights: list[float] | None = Field(
        default=None,
        description="Initial weights for the models. "
        "If None, will be set to uniform distribution.",
    )
    estimates_pose: bool = Field(
        default=False,
        description="Whether to estimate the pose of the particles during optimization. "
        + "If True, the pose will be estimated using the current weights of the ensemble."
        + " If False, the pose will be estimated using uniform weights.",
    )

    @field_serializer("initial_weights")
    def serialize_initial_weights(self, v):
        if v is not None:
            v = jnp.array(v)
            v = v / jnp.sum(v)

        return v

    @field_validator("estimates_pose")
    @classmethod
    def validate_estimates_pose(cls, v):
        if v:
            raise Warning(
                "estimates_pose is set to True. This feature is still experimental, "
                + "and may slow down the optimization process."
            )
        return v


class EnsOptMDConfigProjector(BaseModel, extra="forbid"):
    n_steps: PositiveInt = Field(
        description="Number of steps for the MD sampler. Must be greater than 0."
    )
    bias_constant_in_kjpermol: PositiveFloat | list[PositiveFloat] = Field(
        description="Biasing constant for the projection step. "
        + "Can be a single value or a list of two values for linear scheduling."
    )
    platform: Literal["CPU", "CUDA", "OpenCL"] = Field(
        default="CPU",
        description="Platform to use for the MD sampler. "
        + "Must be 'CPU', 'CUDA', or 'OpenCL'.",
    )
    platform_properties: dict[str, str] | None = Field(
        default=None,
        description="Platform properties for OpenMM. "
        + "For CPU the default is {'Threads': '1'}"
        + " and for CUDA the default is {'DeviceIndex': '0'}.",
    )

    path_to_initial_states: str | list[FilePath] | None = Field(
        default=None,
        description="Path to the initial states. "
        + "If None, will be set to the path to the atomic models.",
    )

    @model_validator(mode="after")
    def validate_config(self):
        if self.platform == "CPU":
            if self.platform_properties is None:
                self.platform_properties = {"Threads": "1"}
        elif self.platform == "CUDA":
            if self.platform_properties is None:
                self.platform_properties = {"DeviceIndex": "0"}
        else:
            if self.platform_properties is None:
                raise ValueError(
                    "platform_properties must be provided for OpenCL platform"
                )
        return self

    @field_validator("path_to_initial_states")
    @classmethod
    def validate_path_to_initial_states(cls, v):
        if v is None:
            return v
        else:
            return _validate_files_with_type(v, file_types=[".xml"])


class EnsOptAlignConfig(BaseModel, extra="forbid"):
    path_to_prealigned_atomic_model: Annotated[
        str, AfterValidator(partial(_validate_file_with_type, file_type=".pdb"))
    ] = Field(
        description="Path to the reference model. "
        + "This model should be aligned to the cryo-EM particles, "
        + " and will be used for alignment of the walkers during optimization."
    )

    path_to_reference_volume: FilePath | None = Field(
        default=None,
        description="Path to the consensus volume. "
        + "Used for rigid body alignment of walkers. "
        + "If None, no alignment will be performed.",
    )

    downsample_box_size: PositiveInt = Field(
        default=32,
        description="Box size to downsample the volume for alignment. "
        + "The box size must be a positive integer.",
    )

    reference_volume_voxel_size: PositiveFloat | None = Field(
        default=None,
        description="Overrides the voxel size stored in the MRC header "
        + "of the consensus volume."
        + " If None, the voxel size from the MRC header will be used.",
    )

    @field_validator("path_to_prealigned_atomic_model")
    @classmethod
    def validate_path_to_prealigned_atomic_model(cls, v):
        return _validate_file_with_type(v, file_type=".pdb")

    @field_validator("path_to_reference_volume")
    @classmethod
    def validate_path_to_consensus_volume(cls, v):
        if v is not None:
            return _validate_file_with_type(v, file_type=".mrc")


class EnsOptDataConfig(BaseModel, extra="forbid"):
    path_to_starfile: FilePath = Field(
        description="Path to the starfile containing the particle information."
    )
    path_to_relion_project: DirectoryPath = Field(
        description="Path to the relion project directory."
    )
    loads_envelope: bool = Field(
        default=False, description="Whether to load the envelope from the starfile. "
    )
    path_to_volumetric_mask: FilePath | None = Field(
        default=None,
        description="Path to a volumetric mask. "
        + "For example: the dillated mask obtained from a homogeneous refinment job, "
        + "or a mask to focus on a specific region. "
        + "If None, no mask will be applied.",
    )
    data_sign: Literal["dark-on-light", "light-on-dark"] = Field(
        default="dark-on-light",
        description="Sign convention for the data. "
        + "'dark-on-light' means that the particles "
        + "are dark on a light background (default). "
        + "'light-on-dark' means that the particles "
        + "are light on a dark background.",
    )

    @field_validator("path_to_starfile")
    @classmethod
    def validate_path_to_starfile(cls, v):
        return _validate_file_with_type(v, file_type=".star")

    @field_validator("path_to_volumetric_mask")
    @classmethod
    def validate_path_to_volumetric_mask(cls, v):
        if v is not None:
            return _validate_file_with_type(v, file_type=".mrc")


class EnsOptMDConfig(BaseModel, extra="forbid"):
    # I/O
    path_to_atomic_models: str | list[FilePath] = Field(
        description="Path to the atomic models directory. "
        + "If a pattern is provided, all files matching the pattern will be used."
    )

    path_to_output: Path = Field(
        description="Path to the output directory. "
        + "If it does not exist, it will be created.",
    )
    atom_selection: str | FilePath = Field(
        default="all",
        description="Selection string for atom selection, "
        + "or a txt/npy file containing atom indices",
    )

    loads_b_factors: bool = Field(
        default=False,
        description="Whether to load the thermal b-factors from the PDB file. "
        + "Only used if the atomic model is in PDB format. "
        + "Otherwise it will be ignored."
        + "Also known as Debye-Waller factors.",
    )

    # Data
    data_params: dict = Field(
        description="Parameters for the experimental data. "
        + "This is a dictionary formatted by the `EnsOptDataConfig` class."
    )

    # Pipeline
    projector_params: dict = Field(
        description="Parameters for the physics-based ensemble projector. "
        + "This is a dictionary formatted by the `EnsOptMDConfigProjector` class."
    )
    likelihood_optimizer_params: dict = Field(
        description="Parameters for the ensemble optimizer. "
        + "This is a dictionary formatted by "
        + "the `EnsOptMDConfigOptimizationConfig` class."
    )

    alignment_params: dict = Field(
        description="Parameters for the alignment of the walkers. "
    )
    # Optimization
    n_steps: PositiveInt = Field(
        description="Number of steps of cryoJAX ensemble refinement to run."
    )

    # Miscellaneous
    rng_seed: int = Field(default=0, description="Random seed.")

    @model_validator(mode="after")
    def validate_config(self):
        n_atomic_models = len(self.path_to_atomic_models)
        if self.projector_params["path_to_initial_states"] is not None:
            n_initial_states = len(self.projector_params["path_to_initial_states"])
            assert n_atomic_models == n_initial_states, (
                f"Number of initial states {n_initial_states} "
                + f"does not match number of atomic models {n_atomic_models}."
            )

        if self.likelihood_optimizer_params["initial_weights"] is not None:
            n_initial_weights = len(self.likelihood_optimizer_params["initial_weights"])
            if n_atomic_models != n_initial_weights:
                raise Warning(
                    f"Number of initial weights {n_initial_weights} "
                    + f"does not match number of atomic models {n_atomic_models}."
                    + " Setting initial weights to uniform distribution."
                )
            self.likelihood_optimizer_params["initial_weights"] = jnp.asarray(
                [1.0 / n_atomic_models for _ in range(n_atomic_models)]
            )
        return self

    @field_validator("path_to_atomic_models")
    @classmethod
    def validate_path_to_atomic_models(cls, v):
        return _validate_files_with_type(v, file_types=[".pdb"])

    @field_validator("likelihood_optimizer_params")
    @classmethod
    def validate_ensemble_opt_config(cls, values):
        return dict(EnsOptMDConfigOptimizationConfig(**values).model_dump())

    @field_validator("projector_params")
    @classmethod
    def validate_md_sampler_config(cls, values):
        return dict(EnsOptMDConfigProjector(**values).model_dump())

    @field_validator("alignment_params")
    @classmethod
    def validate_aligner_config(cls, values):
        return dict(EnsOptAlignConfig(**values).model_dump())

    @field_validator("data_params")
    @classmethod
    def validate_data_config(cls, values):
        return dict(EnsOptDataConfig(**values).model_dump())

    @field_validator("atom_selection")
    @classmethod
    def validate_atom_selection(cls, values):
        suffix = Path(values).suffix
        if suffix in [".txt", ".npy"]:
            assert Path(values).exists(), f"Indices File: {values} does not exist."

        elif suffix not in [".txt", ".npy", ""]:
            raise ValueError("Invalid file type for atom selection.")

        else:
            try:
                mdtraj.Topology().select(values)
            except Exception as e:
                raise ValueError(f"Invalid atom selection string: {values}. Error: {e}")
        return values


### Keeping just in case we want to re-enable auto-incrementing output paths ###

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
