from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union
from typing_extensions import Annotated, Literal

import jax.numpy as jnp
import mdtraj
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


class EnsOptMDConfigMDConfig(BaseModel, extra="forbid"):
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
            return _validate_files_with_type(v, file_types=[".xml"])


class EnsOptMDConfig(BaseModel, extra="forbid"):
    # I/O

    path_to_atomic_models: Union[str, List[FilePath]] = Field(
        description="Path to the atomic models directory. "
        + "If a pattern is provided, all files matching the pattern will be used."
    )
    path_to_reference_model: Annotated[
        str, AfterValidator(partial(_validate_file_with_type, file_type=".pdb"))
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
        + "This is a dictionary formatted by the `EnsOptMDConfigMDConfig` class."
    )
    likelihood_optimizer_params: Dict = Field(
        description="Parameters for the ensemble optimizer. "
        + "This is a dictionary formatted by "
        + "the `EnsOptMDConfigOptimizationConfig` class."
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
        return _validate_files_with_type(v, file_types=[".pdb"])

    @field_validator("path_to_reference_model")
    @classmethod
    def validate_path_to_reference_model(cls, v):
        return _validate_file_with_type(v, file_type=".pdb")

    @field_validator("path_to_starfile")
    @classmethod
    def validate_path_to_starfile(cls, v):
        return _validate_file_with_type(v, file_type=".star")

    @field_validator("likelihood_optimizer_params")
    @classmethod
    def validate_ensemble_opt_config(cls, values):
        return dict(EnsOptMDConfigOptimizationConfig(**values).model_dump())

    @field_validator("projector_params")
    @classmethod
    def validate_md_sampler_config(cls, values):
        return dict(EnsOptMDConfigMDConfig(**values).model_dump())

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
