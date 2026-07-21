import warnings
from pathlib import Path
from typing import Annotated, Literal

import mdtraj
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from .utils import _validate_file_with_type, _validate_files_with_type


class VolumeIntegratorBackendConfig(BaseModel, extra="forbid"):
    enable_pallas: bool = Field(
        default=False,
        description="Whether to use the Pallas/Triton GPU kernel, instead of pure "
        + "JAX, when spread_mode='local'. Ignored when spread_mode='exact'. Most "
        + "advantageous for the backward pass.",
    )
    spread_mode: Literal["exact", "local"] = Field(
        default="local",
        description="How each gaussian is projected onto the grid. 'exact' "
        + "evaluates dense gaussian integrals over the whole grid. 'local' instead "
        + "spreads each gaussian onto only its nearby grid points, with the "
        + "truncation width set by `spread_width_in_stds`, trading accuracy for "
        + "speed since gaussians are short-ranged relative to typical grid sizes.",
    )
    spread_width_in_stds: PositiveFloat = Field(
        default=6.0,
        description="Truncation width for 'local' spread mode, in standard "
        + "deviations of the gaussian. Ignored when spread_mode='exact'.",
    )
    sampling_mode: Literal["average", "point"] = Field(
        default="average",
        description="How the projected volume is sampled at each pixel. 'average' "
        + "uses gaussian integrals (error functions) to compute the average value "
        + "over the pixel. 'point' evaluates the gaussian at the pixel center.",
    )

    @model_validator(mode="after")
    def validate_config(self):
        if self.spread_mode == "exact" and self.spread_width_in_stds is not None:
            warnings.warn(
                "spread_width_in_stds is ignored when spread_mode='exact'.",
                stacklevel=2,
            )
        return self


class EnsOptMDConfigOptimizationConfig(BaseModel, extra="forbid"):
    n_steps: PositiveInt = Field(
        default=1, description="Number of steps for the optimization process."
    )
    step_size: PositiveFloat = Field(
        description="Step size in Angstroms for the optimization process."
    )
    n_batches_per_step: PositiveInt = Field(
        default=1,
        description="Number of batches to use for each optimization step. "
        + "Using more batches will provide a better estimate of the gradients, "
        + "but will also increase the computational cost. "
        + "If set to 1, the optimization will be performed using the entire dataset.",
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
    volume_integrator_backend: VolumeIntegratorBackendConfig = Field(
        default_factory=VolumeIntegratorBackendConfig,
        description="Backend options for the volume integrator used during "
        + "optimization.",
    )

    @field_validator("initial_weights")
    @classmethod
    def validate_initial_weights(cls, v):
        if v is not None:
            total = sum(v)
            v = [w / total for w in v]
        return v

    @field_validator("estimates_pose")
    @classmethod
    def validate_estimates_pose(cls, v):
        if v:
            warnings.warn(
                "estimates_pose is set to True. This feature is still experimental, "
                + "and may slow down the optimization process.",
                stacklevel=2,
            )
        return v


class MDParamsConfig(BaseModel, extra="forbid"):
    """YAML-serializable overrides for OpenMM MD simulation parameters.

    Omitted fields fall back to the defaults in ``_get_default_md_params()``.
    Use ``md_params_config_to_openmm_overrides`` to convert this to an openmm dict.
    """

    forcefield: str = Field(
        default="amber14-all.xml",
        description="OpenMM forcefield XML file name.",
    )
    water_model: str = Field(
        default="amber14/tip3p.xml",
        description="OpenMM water model XML file name.",
    )
    nonbonded_method: Literal[
        "PME",
        "CutoffNonPeriodic",
        "NoCutoff",
        "CutoffPeriodic",
        "Ewald",
        "LJPME",
    ] = Field(
        default="CutoffNonPeriodic",
        description=(
            "Nonbonded method alias. "
            "'PME', 'Ewald', 'CutoffPeriodic', and 'LJPME' require a periodic box. "
            "'CutoffNonPeriodic' is the default for implicit/vacuum simulations. "
            "'NoCutoff' disables any cutoff (slow; small systems only)."
        ),
    )
    nonbonded_cutoff_nm: PositiveFloat = Field(
        default=1.0,
        description="Cutoff distance for nonbonded interactions, in nanometers.",
    )
    constraints: Literal["HBonds", "AllBonds", "HAngles"] | None = Field(
        default="HBonds",
        description=(
            "Bond/angle constraints. 'HBonds' constrains bonds involving hydrogen "
            "(recommended with a 2 fs timestep). 'AllBonds' constrains all bonds. "
            "'HAngles' additionally constrains H–X–H angles, allowing ~4 fs timesteps. "
            "Set to null to disable constraints entirely."
        ),
    )
    temperature_K: PositiveFloat = Field(
        default=300.0,
        description="Langevin integrator temperature, in Kelvin.",
    )
    friction_per_ps: PositiveFloat = Field(
        default=1.0,
        description="Langevin integrator friction coefficient, in 1/picosecond.",
    )
    timestep_ps: PositiveFloat = Field(
        default=0.002,
        description="Integration timestep, in picoseconds.",
    )


class EnsOptMDConfigProjector(BaseModel, extra="forbid"):
    n_steps: PositiveInt = Field(
        description="Number of steps for the MD sampler. Must be greater than 0."
    )
    bias_constant_in_kjpermol: (
        PositiveFloat | Annotated[list[PositiveFloat], Field(min_length=2, max_length=2)]
    ) = Field(
        description="Biasing constant for the projection step. "
        + "Can be a single value, or a list of exactly two values [start, end] "
        + "for linear scheduling."
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
    md_params: MDParamsConfig = Field(
        default_factory=MDParamsConfig,
        description="Overrides for OpenMM MD simulation parameters. "
        + "Any omitted field falls back to the built-in default. "
        + "Note: 'platform' and 'platform_properties' are set via the fields above, "
        + "not here.",
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
    path_to_prealigned_atomic_model: str = Field(
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
    path_to_atomic_models: list[str] = Field(
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

        initial_states = self.projector_params["path_to_initial_states"]
        if initial_states is not None and len(initial_states) != n_atomic_models:
            raise ValueError(
                f"Number of initial states {len(initial_states)} "
                f"does not match number of atomic models {n_atomic_models}."
            )

        initial_weights = self.likelihood_optimizer_params["initial_weights"]
        if initial_weights is not None and len(initial_weights) != n_atomic_models:
            warnings.warn(
                f"Number of initial weights {len(initial_weights)} "
                f"does not match number of atomic models {n_atomic_models}. "
                "Falling back to a uniform distribution.",
                stacklevel=2,
            )
            initial_weights = None

        if initial_weights is None:
            initial_weights = [1.0 / n_atomic_models] * n_atomic_models

        self.likelihood_optimizer_params["initial_weights"] = initial_weights
        return self

    @field_validator("path_to_atomic_models", mode="before")
    @classmethod
    def validate_path_to_atomic_models(cls, v):
        return _validate_files_with_type(v, file_types=[".pdb"])

    @field_validator("likelihood_optimizer_params")
    @classmethod
    def validate_likelihood_optimizer_params(cls, values):
        return EnsOptMDConfigOptimizationConfig(**values).model_dump()

    @field_validator("projector_params")
    @classmethod
    def validate_projector_params(cls, values):
        return EnsOptMDConfigProjector(**values).model_dump()

    @field_validator("alignment_params")
    @classmethod
    def validate_alignment_params(cls, values):
        return EnsOptAlignConfig(**values).model_dump()

    @field_validator("data_params")
    @classmethod
    def validate_data_params(cls, values):
        return EnsOptDataConfig(**values).model_dump()

    @field_validator("atom_selection")
    @classmethod
    def validate_atom_selection(cls, v):
        suffix = Path(v).suffix
        if suffix in [".txt", ".npy"]:
            if not Path(v).exists():
                raise ValueError(f"Indices file {v} does not exist.")
        elif suffix != "":
            raise ValueError("Invalid file type for atom selection.")
        else:
            try:
                mdtraj.Topology().select(v)
            except Exception as e:
                raise ValueError(f"Invalid atom selection string: {v}. Error: {e}")
        return v


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
