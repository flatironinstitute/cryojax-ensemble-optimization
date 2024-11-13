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
from typing import List, Optional, Union
import jax.numpy as jnp


class _EnsembleOptimizerValidator(BaseModel, extra="forbid"):
    n_steps: int = 1
    step_size: float
    batch_size: int
    init_weights: Optional[List[float]] = None
    noise_variance: float

    @field_validator("n_steps")
    @classmethod
    def validate_pos_opt_steps(cls, value):
        if value <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return value

    @field_validator("step_size")
    @classmethod
    def validate_pos_opt_stepsize(cls, value):
        if value <= 0:
            raise ValueError("Stepsize must be greater than 0")
        return value

    @field_validator("noise_variance", mode="before")
    @classmethod
    def validate_noise_variance(cls, v):
        if v <= 0:
            raise ValueError("Noise variance must be greater than 0")
        return v

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v):
        if v <= 0:
            raise ValueError("Batch size must be greater than 0")
        return v

    @field_serializer("init_weights")
    def serialize_init_weights(self, v):
        if v is not None:
            v = jnp.array(v)
            v = v / jnp.sum(v)

        return v


class _PipelineMDSamplerAllAtomValidator(BaseModel, extra="forbid"):
    mode: str = "all-atom"
    n_steps: int
    mdsampler_force_constant: float
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
    models_fnames: Union[str, List[str]]
    checkpoints_fnames: Optional[Union[str, List[str]]] = None
    ref_model_fname: str
    path_to_starfile: str
    path_to_relion_project: str
    output_path: str
    max_n_models: Optional[int] = None

    # Pipeline
    md_sampler_config: dict
    ensemble_optimizer_config: dict

    # Optimization
    n_steps: int

    # Image and MD stuff
    atom_list_filter: Optional[str] = None
    rng_seed: int = 0

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

    @model_validator(mode="after")
    def validate_config(self):
        if isinstance(self.models_fnames, str):
            if "*" in self.models_fnames:
                models_fnames = natsorted(
                    glob.glob(
                        self.models_fnames, root_dir=self.path_to_models_and_chkpoints
                    )
                )
                if len(models_fnames) == 0:
                    raise FileNotFoundError(
                        f"No files found with pattern {self.models_fnames}"
                    )
            else:
                models_fnames = [self.models_fnames]

        elif isinstance(self.models_fnames, list):
            assert len(self.models_fnames) > 0, "No models Given."
            models_fnames = self.models_fnames

        else:
            raise ValueError("models_fnames must be a string or a list of strings.")

        for i in range(len(models_fnames)):
            models_fnames[i] = os.path.join(
                self.path_to_models_and_chkpoints, models_fnames[i]
            )
            assert os.path.exists(
                models_fnames[i]
            ), f"Model {models_fnames[i]} does not exist."

        if self.checkpoints_fnames is not None:
            if isinstance(self.checkpoints_fnames, str):
                if "*" in self.checkpoints_fnames:
                    checkpoints_fnames = natsorted(
                        glob.glob(
                            self.checkpoints_fnames,
                            root_dir=self.path_to_models_and_chkpoints,
                        )
                    )
                    if len(checkpoints_fnames) == 0:
                        raise FileNotFoundError(
                            f"No files found with pattern {self.checkpoints_fnames}"
                        )
                else:
                    checkpoints_fnames = [self.checkpoints_fnames]

            elif isinstance(self.checkpoints_fnames, list):
                assert len(self.checkpoints_fnames) > 0, "No checkpoints Given."
                checkpoints_fnames = self.checkpoints_fnames

            else:
                raise ValueError(
                    "checkpoints_fnames must be a string or a list of strings."
                )

            for i in range(len(checkpoints_fnames)):
                checkpoints_fnames[i] = os.path.join(
                    self.path_to_models_and_chkpoints, checkpoints_fnames[i]
                )
                assert os.path.exists(
                    checkpoints_fnames[i]
                ), f"Checkpoint {checkpoints_fnames[i]} does not exist."

        else:
            checkpoints_fnames = None

        if self.max_n_models is not None:
            if len(models_fnames) < self.max_n_models:
                raise ValueError(
                    f"Number of models {len(models_fnames)} is less than max_n_models {self.max_n_models}"
                )

            if checkpoints_fnames is not None:
                assert len(checkpoints_fnames[: self.max_n_models]) == len(
                    models_fnames[: self.max_n_models]
                ), "Number of checkpoints does not match number of models"

        elif self.max_n_models is None and checkpoints_fnames is not None:
            assert len(checkpoints_fnames) == len(
                models_fnames
            ), "Number of checkpoints does not match number of models"

        else:
            pass

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

    @field_validator("ensemble_optimizer_config")
    @classmethod
    def validate_ensemble_opt_config(cls, v):
        _EnsembleOptimizerValidator(**v)
        return v

    @field_validator("md_sampler_config")
    @classmethod
    def validate_md_sampler_config(cls, v):
        _PipelineMDSamplerAllAtomValidator(**v)
        return v

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
                new_path = os.path.join(v, f"Job{job_numbers[-1]+1:03d}")

        v = new_path
        return v

    @model_serializer
    def serialize_model(self):
        """
        Need to serialize this elements as a model since they need each other
        """

        if self.atom_list_filter is None:
            if self.mode == "all-atom":
                self.atom_list_filter = "protein and not name H*"
            elif self.mode == "cg":
                self.atom_list_filter = "protein"

        # Serialize model files
        if isinstance(self.models_fnames, str):
            if "*" in self.models_fnames:
                models_fnames = natsorted(
                    glob.glob(
                        self.models_fnames, root_dir=self.path_to_models_and_chkpoints
                    )
                )
                if len(models_fnames) == 0:
                    raise FileNotFoundError(
                        f"No files found with pattern {self.models_fnames}"
                    )
            else:
                models_fnames = [self.models_fnames]

        else:
            models_fnames = self.models_fnames

        if self.checkpoints_fnames is not None:
            if isinstance(self.checkpoints_fnames, str):
                if "*" in self.checkpoints_fnames:
                    checkpoints_fnames = natsorted(
                        glob.glob(
                            self.checkpoints_fnames,
                            root_dir=self.path_to_models_and_chkpoints,
                        )
                    )
                    if len(checkpoints_fnames) == 0:
                        raise FileNotFoundError(
                            f"No files found with pattern {self.checkpoints_fnames}"
                        )
                else:
                    checkpoints_fnames = [self.checkpoints_fnames]

            else:
                checkpoints_fnames = self.checkpoints_fnames

        else:
            checkpoints_fnames = None

        if self.max_n_models is not None:
            models_fnames = models_fnames[: self.max_n_models]
            checkpoints_fnames = (
                checkpoints_fnames[: self.max_n_models]
                if checkpoints_fnames is not None
                else None
            )

        else:
            self.max_n_models = len(models_fnames)

        self.models_fnames = []
        for i in range(len(models_fnames)):
            self.models_fnames.append(
                os.path.join(self.path_to_models_and_chkpoints, models_fnames[i])
            )

        if checkpoints_fnames is not None:
            self.checkpoints_fnames = []
            for i in range(len(checkpoints_fnames)):
                self.checkpoints_fnames.append(
                    os.path.join(
                        self.path_to_models_and_chkpoints, checkpoints_fnames[i]
                    )
                )
        else:
            self.checkpoints_fnames = None

        self.ref_model_fname = os.path.join(
            self.path_to_models_and_chkpoints, self.ref_model_fname
        )

        # serialize other configs
        self.ensemble_optimizer_config = dict(
            _EnsembleOptimizerValidator(**self.ensemble_optimizer_config).model_dump()
        )

        if self.ensemble_optimizer_config["init_weights"] is None:
            weights = jnp.ones(self.max_n_models) / self.max_n_models
            self.ensemble_optimizer_config["init_weights"] = weights

        self.md_sampler_config = dict(
            _PipelineMDSamplerAllAtomValidator(**self.md_sampler_config).model_dump()
        )

        return self
