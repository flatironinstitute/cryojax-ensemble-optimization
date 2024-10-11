import MDAnalysis as mda
import os
from natsort import natsorted
import glob
from pydantic import (
    BaseModel,
    field_validator,
    model_validator,
    field_serializer,
    model_serializer,
)
import logging
from typing import List, Optional, Union


class _PipelineWeightOpValidator(BaseModel, extra="forbid"):
    step_type: str = "weight_opt"
    weight_opt_steps: int
    weight_opt_stepsize: float

    @field_validator("weight_opt_steps")
    @classmethod
    def validate_weight_opt_steps(cls, value):
        if value < 0:
            raise ValueError("Number of steps must be greater than 0")
        return value

    @field_validator("weight_opt_stepsize")
    @classmethod
    def validate_weight_opt_stepsize(cls, value):
        if value <= 0:
            raise ValueError("Stepsize must be greater than 0")
        return value


class _PipelinePosOpValidator(BaseModel, extra="forbid"):
    step_type: str = "pos_opt"
    pos_opt_steps: int
    pos_opt_stepsize: float

    @field_validator("pos_opt_steps")
    @classmethod
    def validate_pos_opt_steps(cls, value):
        if value <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return value

    @field_validator("pos_opt_stepsize")
    @classmethod
    def validate_pos_opt_stepsize(cls, value):
        if value <= 0:
            raise ValueError("Stepsize must be greater than 0")
        return value


class _PipelineMDSamplerAllAtomValidator(BaseModel, extra="forbid"):
    step_type: str = "mdsampler"
    mode: str = "all-atom"
    n_steps: int
    mdsampler_force_constant: float
    checkpoint_fnames: Optional[Union[str, List[str]]] = None
    platform: str = "CPU"
    platform_properties: dict = {"Threads": None}

    @field_validator("n_steps")
    @classmethod
    def validate_n_steps(cls, value):
        if value <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return value

    @field_validator("mdsampler_force_constant")
    @classmethod
    def validate_mdsampler_force_constant(cls, value):
        if value <= 0:
            raise ValueError("Force constant must be greater than 0")
        return value

    @field_validator("platform")
    @classmethod
    def validate_platform(cls, value):
        if value not in ["CPU", "CUDA"]:
            raise ValueError("Invalid platform, must be 'CPU' or 'CUDA'")
        return value

    @field_validator("platform_properties")
    @classmethod
    def validate_platform_properties(cls, v):
        if v["Threads"] is not None:
            if v["Threads"] <= 0:
                raise ValueError("Number of threads must be greater than 0")
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


class OptimizationConfig(BaseModel, extra="forbid"):
    experiment_name: str
    mode: str

    # I/O
    path_to_models_and_chkpoints: str
    models_fname: Union[str, List[str]]
    ref_model_fname: str
    path_to_starfile: str
    path_to_relion_project: str
    output_path: str
    max_n_models: Optional[int] = None

    # Pipeline
    pipeline: dict

    # Optimization
    n_steps: int
    batch_size: int

    # Image and MD stuff
    resolution: float
    atom_list_filter: Optional[str] = None
    noise_variance: float
    rng_seed: int = 0

    @field_validator("noise_variance", mode="before")
    @classmethod
    def validate_noise_variance(cls, v):
        if v <= 0:
            raise ValueError("Noise variance must be greater than 0")
        return v

    @field_validator("mode", mode="before")
    @classmethod
    def validate_mode(cls, value):
        if value not in ["all-atom", "resid", "cg"]:
            raise ValueError("Invalid mode, must be 'all-atom', 'resid' or 'cg'")

        if value in ["resid", "cg"]:
            raise NotImplementedError(f"Mode {value} not implemented yet")
        return value

    @field_validator("path_to_models_and_chkpoints", mode="before")
    @classmethod
    def validate_path_to_models_and_chkpoints(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Working Directory {v} does not exist.")
        return v

    @field_validator("path_to_starfile")
    @classmethod
    def validate_starfile_path(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Starfile {v} does not exist.")
        return v

    @field_validator("path_to_relion_project")
    @classmethod
    def validate_path_to_relion_project(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Relion project {v} does not exist.")
        return v

    @field_validator("n_steps")
    @classmethod
    def validate_n_steps(cls, v):
        if v <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return v

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, v):
        if v <= 0:
            raise ValueError("Resolution must be greater than 0")
        return v

    @model_validator(mode="after")
    def validate_config(self):
        if isinstance(self.models_fname, str):
            if "*" in self.models_fname:
                models_fname = natsorted(
                    glob.glob(
                        self.models_fname, root_dir=self.path_to_models_and_chkpoints
                    )
                )
                if len(models_fname) == 0:
                    raise FileNotFoundError(
                        f"No files found with pattern {self.models_fname}"
                    )
            else:
                models_fname = [self.models_fname]

        elif isinstance(self.models_fname, list):
            assert len(self.models_fname) > 0, "No models Given."
            models_fname = self.models_fname

        else:
            raise ValueError("models_fname must be a string or a list of strings.")

        for i in range(len(models_fname)):
            models_fname[i] = os.path.join(
                self.path_to_models_and_chkpoints, models_fname[i]
            )
            assert os.path.exists(
                models_fname[i]
            ), f"Model {models_fname[i]} does not exist."

        if self.max_n_models is not None:
            if len(models_fname) < self.max_n_models:
                raise ValueError(
                    f"Number of models {len(models_fname)} is less than max_n_models {self.max_n_models}"
                )

        ref_model_path = os.path.join(
            self.path_to_models_and_chkpoints, self.ref_model_fname
        )
        assert os.path.exists(
            ref_model_path
        ), f"Reference model {ref_model_path} does not exist."

        if self.atom_list_filter is not None:
            u = mda.Universe(ref_model_path)
            try:
                u.select_atoms(self.atom_list_filter)
            except ValueError:
                raise ValueError(f"Invalid atom list filter {self.atom_list_filter}")

        return self

    @field_validator("pipeline")
    def validate_pipeline(cls, v):
        for key in v.keys():
            if v[key]["step_type"] == "mdsampler":
                if v[key]["mode"] == "all-atom":
                    _PipelineMDSamplerAllAtomValidator(**v[key])

                elif v[key]["mode"] == "cg":
                    # v[key] = _PipelineMDSamplerCGValidator(**v[key])
                    raise NotImplementedError("CG mode not implemented yet")

                else:
                    raise ValueError(f"Invalid mode {v[key]['mode']}")

            elif v[key]["step_type"] == "weight_opt":
                _PipelineWeightOpValidator(**v[key])

            elif v[key]["step_type"] == "pos_opt":
                _PipelinePosOpValidator(**v[key])

            else:
                raise ValueError(f"Invalid pipeline element {v[key]}")

        return v

    @field_serializer("atom_list_filter")
    def serialize_atom_list_filter(self, v):
        if v is None:
            if self.mode == "all-atom":
                v = "protein and not name H*"
            elif self.mode == "cg":
                v = "protein"
        return v

    @model_serializer
    def serialize_model(self):
        """
        Need to serialize this elements as a model since they need each other
        """

        # Serialize model files
        if isinstance(self.models_fname, str):
            if "*" in self.models_fname:
                models_fname = natsorted(
                    glob.glob(
                        self.models_fname, root_dir=self.path_to_models_and_chkpoints
                    )
                )
                if len(models_fname) == 0:
                    raise FileNotFoundError(
                        f"No files found with pattern {self.models_fname}"
                    )
            else:
                models_fname = [self.models_fname]

        else:
            models_fname = self.models_fname

        if self.max_n_models is not None:
            models_fname = models_fname[: self.max_n_models]

        else:
            self.max_n_models = len(models_fname)

        self.models_fname = []
        for i in range(len(models_fname)):
            self.models_fname.append(
                os.path.join(self.path_to_models_and_chkpoints, models_fname[i])
            )

        self.ref_model_fname = os.path.join(
            self.path_to_models_and_chkpoints, self.ref_model_fname
        )

        # serialize checkpoint names in samplers
        for key in self.pipeline.keys():
            if self.pipeline[key]["step_type"] == "mdsampler":
                if (
                    "checkpoint_fnames" in self.pipeline[key].keys()
                    and self.pipeline[key]["checkpoint_fnames"] is not None
                ):
                    if isinstance(self.pipeline[key]["checkpoint_fnames"], str):
                        if "*" in self.pipeline[key]["checkpoint_fnames"]:
                            checkpoint_fnames = natsorted(
                                glob.glob(
                                    self.pipeline[key]["checkpoint_fnames"],
                                    root_dir=self.path_to_models_and_chkpoints,
                                )
                            )
                            if len(checkpoint_fnames) == 0:
                                raise FileNotFoundError(
                                    f"No files found with pattern {checkpoint_fnames}"
                                )
                        else:
                            checkpoint_fnames = [
                                self.pipeline[key]["checkpoint_fnames"]
                            ]

                    else:
                        checkpoint_fnames = self.pipeline[key]["checkpoint_fnames"]

                    checkpoint_fnames = checkpoint_fnames[: self.max_n_models]
                    assert (
                        len(checkpoint_fnames) == len(self.models_fname)
                    ), f"Number of checkpoint files {len(checkpoint_fnames)} does not match number of models {len(self.models_fname)}"

                    self.pipeline[key]["checkpoint_fnames"] = []
                    for i in range(len(checkpoint_fnames)):
                        self.pipeline[key]["checkpoint_fnames"].append(
                            os.path.join(
                                self.path_to_models_and_chkpoints, checkpoint_fnames[i]
                            )
                        )

                    for i in range(len(self.pipeline[key]["checkpoint_fnames"])):
                        assert os.path.exists(
                            self.pipeline[key]["checkpoint_fnames"][i]
                        ), f"Checkpoint {self.pipeline[key]['checkpoint_fnames'][i]} does not exist."

                    logging.info(
                        f"Checkpoint files for pipeline step {key} were set to {self.pipeline[key]['checkpoint_fnames']}"
                    )
                    logging.info("Models and checkpoints will be linked as")
                    for i in range(len(self.pipeline[key]["checkpoint_fnames"])):
                        logging.info(
                            f"Model {self.models_fname[i]} with checkpoint {self.pipeline[key]['checkpoint_fnames'][i]}"
                        )

                else:
                    continue

        # Serialize pipeline
        for key in self.pipeline.keys():
            if self.pipeline[key]["step_type"] == "mdsampler":
                if self.pipeline[key]["mode"] == "all-atom":
                    self.pipeline[key] = dict(
                        _PipelineMDSamplerAllAtomValidator(
                            **self.pipeline[key]
                        ).model_dump()
                    )
                elif self.pipeline[key]["mode"] == "cg":
                    self.pipeline[key] = dict(
                        _PipelineMDSamplerCGValidator(**self.pipeline[key]).model_dump()
                    )

            elif self.pipeline[key]["step_type"] == "weight_opt":
                self.pipeline[key] = dict(
                    _PipelineWeightOpValidator(**self.pipeline[key]).model_dump()
                )

            elif self.pipeline[key]["step_type"] == "pos_opt":
                self.pipeline[key] = dict(
                    _PipelinePosOpValidator(**self.pipeline[key]).model_dump()
                )
        return self
