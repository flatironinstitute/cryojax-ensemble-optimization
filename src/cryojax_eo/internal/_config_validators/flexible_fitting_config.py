from pathlib import Path
from typing import Literal

import mdtraj
from pydantic import (
    BaseModel,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from .ensemble_opt_config import MDParamsConfig as MDParamsConfig
from .utils import _validate_file_with_type, _validate_files_with_type


class FFOptimizationConfig(BaseModel, extra="forbid"):
    n_steps: PositiveInt = Field(
        default=1, description="Number of steps for the optimization process."
    )
    step_size: PositiveFloat = Field(
        description="Step size in Angstroms for the optimization process."
    )

    batch_size_for_z_planes: PositiveInt = Field(
        default=1,
        description="The number of z-planes to evaluate in parallel with"
        " `jax.vmap`. By default, `1`.",
    )
    n_batches_of_atoms: PositiveInt = Field(
        default=1,
        description="The number of iterations used to evaluate the volume, "
        "where the iteration is taken over groups of atoms. "
        "This is useful if `batch_size = 1` and GPU memory is exhausted. "
        "By default, `1`.",
    )


class FFProjectorConfig(BaseModel, extra="forbid"):
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

    path_to_initial_state: FilePath | None = Field(
        default=None,
        description="Path to the initial openMM state. "
        + "If None, the projector will be initialized with a random state. ",
    )
    md_params: MDParamsConfig = Field(
        default_factory=MDParamsConfig,
        description="Overrides for OpenMM MD simulation parameters. "
        + "Any omitted field falls back to the built-in default. "
        + "Note: 'platform' and 'platform_properties' are set via the fields above, "
        + "not here.",
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

    @field_validator("path_to_initial_state")
    @classmethod
    def validate_path_to_initial_state(cls, v):
        if v is None:
            return v
        else:
            return _validate_files_with_type([v], file_types=[".xml"])[0]


class RefVolFFConfig(BaseModel, extra="forbid"):
    path_to_reference_volume: FilePath | None = Field(
        default=None,
        description="Path to the consensus volume. "
        + "Used for rigid body alignment of walkers. "
        + "If None, no alignment will be performed.",
    )

    flexible_fitting_box_size: PositiveInt = Field(
        default=128,
        description="Box size to crop the volume for flexible fitting. "
        + "The box size must be a positive integer.",
    )

    rigid_alignment_box_size: PositiveInt = Field(
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

    @field_validator("path_to_reference_volume")
    @classmethod
    def validate_path_to_reference_volume(cls, v):
        if v is not None:
            return _validate_file_with_type(v, file_type=".mrc")


class FlexibleFittingConfig(BaseModel, extra="forbid"):
    # I/O
    path_to_atomic_model: FilePath = Field(
        description="Path to the atomic model used for initialization."
        " Either in .pdb or .cif format."
    )
    path_to_prealigned_atomic_model: FilePath = Field(
        description="Path to the reference model. "
        + "This model should be aligned to the cryo-EM particles, "
        + " and will be used for alignment of the walkers during optimization."
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

    # Reference Volume Params
    reference_volume_params: dict = Field(
        description="Parameters for the reference volume. "
    )

    # Pipeline
    projector_params: dict = Field(
        description="Parameters for the physics-based ensemble projector. "
        + "This is a dictionary formatted by the `FFConfigProjector` class."
    )
    walker_optimizer_params: dict = Field(
        description="Parameters for the ensemble optimizer. "
        + "This is a dictionary formatted by "
        + "the `FFOptimizationConfig` class."
    )
    # Optimization
    n_steps: PositiveInt = Field(
        description="Number of steps of cryoJAX ensemble refinement to run."
    )

    @field_validator("path_to_atomic_model")
    @classmethod
    def validate_path_to_atomic_model(cls, v):
        return _validate_files_with_type([v], file_types=[".pdb", ".cif"])[0]

    @field_validator("walker_optimizer_params")
    @classmethod
    def validate_ensemble_opt_config(cls, values):
        return dict(FFOptimizationConfig(**values).model_dump())

    @field_validator("projector_params")
    @classmethod
    def validate_md_sampler_config(cls, values):
        return dict(FFProjectorConfig(**values).model_dump())

    @field_validator("reference_volume_params")
    @classmethod
    def validate_ref_vol_config(cls, values):
        return dict(RefVolFFConfig(**values).model_dump())

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
