from pathlib import Path
from typing import Literal

import mdtraj
from pydantic import (
    BaseModel,
    DirectoryPath,
    Field,
    FilePath,
    PositiveFloat,
    PositiveInt,
    field_validator,
)

from .utils import _validate_file_with_type, _validate_files_with_type


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


class ReweightingConfig(BaseModel, extra="forbid"):
    # I/O
    path_to_structural_files: list[FilePath] = Field(
        description="Path to the structural files directory. "
        + "This should be a list of .pdb, .cif, or .mrc files."
    )

    path_to_output_dir: Path = Field(
        description="Path to the output directory. "
        + "If it does not exist, it will be created.",
    )
    atom_selection: str = Field(
        default="all",
        description="Selection string for atom selection."
        " Only used for .cif or .pdb files.",
    )

    loads_b_factors: bool = Field(
        default=False,
        description="Whether to load the thermal b-factors from the PDB file. "
        + "Only used if the atomic model is in PDB format. "
        + "Otherwise it will be ignored."
        + "Also known as Debye-Waller factors."
        + "Only used for .cif or .pdb files.",
    )

    # Data
    data_params: dict = Field(
        description="Parameters for the experimental data. "
        + "This is a dictionary formatted by the `EnsOptDataConfig` class."
    )

    # Optimization
    max_iter: PositiveInt = Field(
        description="Maximum number of proj. gradient descent steps."
    )
    tol: PositiveFloat = Field(
        default=1e-4,
        description="Tolerance for the stopping criteria of the optimization.",
    )
    n_images_in_parallel: PositiveInt = Field(
        default=1,
        description="Number of images to use in parallel for computing the likelihoods. "
        + "Using more images will require more memory, "
        + "but will also improve the computational performance. ",
    )

    # initial_weights: list[float] | None = Field(
    #     default=None,
    #     description="Initial weights for the models. "
    #     "If None, will be set to uniform distribution.",
    # )

    max_volume_repr_resolution: PositiveFloat | None = Field(
        default=None,
        description="Maximum resolution to use for the volume representation. "
        "If None, no filtering will be applied.",
    )
    estimates_poses: bool = Field(
        default=False,
        description="Whether to estimate the poses of the particles for each model. "
        "If True, the estimated poses will be saved in a new starfile"
        "in the output directory. "
        "This will significantly increase the computational cost of the optimization, "
        "but it can also improve the performance of the optimization, "
        " especially for low-resolution data.",
    )

    # @model_validator(mode="after")
    # def validate_config(self):
    #     n_structures = len(self.path_to_structural_files)

    #     if self.initial_weights is not None:
    #         n_initial_weights = len(self.initial_weights)
    #         if n_structures != n_initial_weights:
    #             raise Warning(
    #                 f"Number of initial weights {n_initial_weights} "
    #                 + f"does not match number of atomic models {n_structures}."
    #                 + " Setting initial weights to uniform distribution."
    #             )
    #         self.initial_weights = [1.0 / n_structures for _ in range(n_structures)]

    #     else:
    #         self.initial_weights = [1.0 / n_structures for _ in range(n_structures)]
    #     return self

    @field_validator("path_to_structural_files")
    @classmethod
    def validate_path_to_structural_files(cls, v):
        return _validate_files_with_type(v, file_types=[".pdb", ".cif", ".mrc"])

    @field_validator("data_params")
    @classmethod
    def validate_data_config(cls, values):
        return dict(EnsOptDataConfig(**values).model_dump())

    @field_validator("atom_selection")
    @classmethod
    def validate_atom_selection(cls, values):
        try:
            mdtraj.Topology().select(values)
        except Exception as e:
            raise ValueError(f"Invalid atom selection string: {values}. Error: {e}")
        return values
