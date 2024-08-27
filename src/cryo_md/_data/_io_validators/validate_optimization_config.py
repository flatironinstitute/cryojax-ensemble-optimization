import MDAnalysis as mda
import os
from natsort import natsorted
import glob
from pydantic import BaseModel, validator, root_validator
import logging
from typing import List, Optional, Union


class PipelineWeightOpValidator(BaseModel):
    type: str = "weight_opt"
    weight_opt_steps: int
    weight_opt_stepsize: float

    @validator("weight_opt_steps")
    def validate_weight_opt_steps(cls, v):
        if v <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return v

    @validator("weight_opt_stepsize")
    def validate_weight_opt_stepsize(cls, v):
        if v <= 0:
            raise ValueError("Stepsize must be greater than 0")
        return v


class PipelinePosOpValidator(BaseModel):
    type: str = "pos_opt"
    pos_opt_steps: int
    pos_opt_stepsize: float

    @validator("pos_opt_steps")
    def validate_pos_opt_steps(cls, v):
        if v <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return v

    @validator("pos_opt_stepsize")
    def validate_pos_opt_stepsize(cls, v):
        if v <= 0:
            raise ValueError("Stepsize must be greater than 0")
        return v


class PipelineMDSamplerAllAtomValidator(BaseModel):
    mode: str = "all-atom"
    n_steps: int
    mdsampler_force_constant: float
    checkpoint_fnames: Optional[Union[str, List[str]]] = None
    platform: str = "CPU"
    platform_properties: dict = {"Threads": None}

    @validator("n_steps")
    def validate_n_steps(cls, v):
        if v <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return v

    @validator("mdsampler_force_constant")
    def validate_mdsampler_force_constant(cls, v):
        if v <= 0:
            raise ValueError("Force constant must be greater than 0")
        return v

    @validator("checkpoint_fnames")
    def validate_checkpoint_fnames(cls, v):
        if v is not None:
            if "*" in v:
                checkpoint_fnames = natsorted(glob.glob(v))
                if len(checkpoint_fnames) == 0:
                    raise FileNotFoundError(f"No files found with pattern {v}")
            else:
                checkpoint_fnames = [v]
                if not os.path.exists(checkpoint_fnames[0]):
                    raise FileNotFoundError(
                        f"Checkpoint {checkpoint_fnames[0]} does not exist."
                    )

            v = checkpoint_fnames
        return v

    @validator("platform")
    def validate_platform(cls, v):
        if v not in ["CPU", "CUDA"]:
            raise ValueError("Invalid platform, must be 'CPU' or 'CUDA'")
        return v

    @validator("platform_properties")
    def validate_platform_properties(cls, v):
        if v["Threads"] is not None:
            if v["Threads"] <= 0:
                raise ValueError("Number of threads must be greater than 0")
        return v


class PipelineMDSamplerCGValidator(BaseModel):
    mode: str = "cg"
    n_steps: int
    mdsampler_force_constant: float
    checkpoint_fnames: Optional[Union[str, List[str]]] = None
    platform: str = "CPU"
    platform_properties: dict = {"Threads": None}
    top_file: str
    epsilon_r: float = 15.0

    @validator("n_steps")
    def validate_n_steps(cls, v):
        if v <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return v

    @validator("mdsampler_force_constant")
    def validate_mdsampler_force_constant(cls, v):
        if v <= 0:
            raise ValueError("Force constant must be greater than 0")
        return v

    @validator("checkpoint_fnames")
    def validate_checkpoint_fnames(cls, v):
        if v is not None:
            if "*" in v:
                checkpoint_fnames = natsorted(glob.glob(v))
                if len(checkpoint_fnames) == 0:
                    raise FileNotFoundError(f"No files found with pattern {v}")
            else:
                checkpoint_fnames = [v]
                if not os.path.exists(checkpoint_fnames[0]):
                    raise FileNotFoundError(
                        f"Checkpoint {checkpoint_fnames[0]} does not exist."
                    )

            v = checkpoint_fnames
        return v

    @validator("platform")
    def validate_platform(cls, v):
        if v not in ["CPU", "CUDA"]:
            raise ValueError("Invalid platform, must be 'CPU' or 'CUDA'")
        return v

    @validator("platform_properties")
    def validate_platform_properties(cls, v):
        if v["Threads"] is not None:
            if v["Threads"] <= 0:
                raise ValueError("Number of threads must be greater than 0")
        return v

    @validator("epsilon_r")
    def validate_epsilon_r(cls, v):
        if v <= 0:
            raise ValueError("Epsilon_r must be greater than 0")
        return v

    @validator("top_file")
    def validate_top_file(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Topology file {v} does not exist.")
        return v


class OptimizationConfig(BaseModel):
    experiment_name: str
    experiment_type: str
    mode: str
    working_dir: str
    starfile_path: str
    models_fname: str
    ref_model_fname: str
    pipeline: dict
    batch_size: int
    output_path: str
    resolution: float
    n_steps: int
    n_models: int = None
    atom_list_filter: Optional[str] = None

    @validator("mode", pre=True)
    def validate_mode(cls, v):
        if v not in ["all-atom", "resid", "cg"]:
            raise ValueError("Invalid mode, must be 'all-atom', 'resid' or 'cg'")
        return v

    @validator("working_dir", pre=True)
    def validate_working_dir(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Working Directory {v} does not exist.")
        return v

    @validator("starfile_path")
    def validate_starfile_path(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Starfile {v} does not exist.")
        return v

    @root_validator
    def validate_models_fname_and_n_models(cls, values):
        if "*" in values["models_fname"]:
            models_fname = natsorted(glob.glob(values["models_fname"]))
            if len(models_fname) == 0:
                raise FileNotFoundError(
                    f"No files found with pattern {values['models_fname']}"
                )
        else:
            models_fname = [values["models_fname"]]
            if not os.path.exists(models_fname[0]):
                raise FileNotFoundError(f"Model {models_fname[0]} does not exist.")
        for i in range(len(models_fname)):
            models_fname[i] = os.path.join(values["working_dir"], models_fname[i])

        values["models_fname"] = models_fname

        logging.info("Using the following models...")
        for i in range(len(models_fname)):
            logging.info("  ", values["models_fname"][i])

        if values["n_models"] is None:
            values["n_models"] = len(models_fname)
        elif values["n_models"] <= 0:
            raise ValueError("Number of models must be greater than 0")
        elif values["n_models"] > len(models_fname):
            raise ValueError(
                "Number of models must be less than or equal to the number of models found."
            )
        return values

    @validator("ref_model_fname", pre=True)
    def validate_ref_model_fname(cls, v):
        if not os.path.exists(v):
            raise FileNotFoundError(f"Reference model {v} does not exist.")
        return v

    @validator("n_steps")
    def validate_n_steps(cls, v):
        if v <= 0:
            raise ValueError("Number of steps must be greater than 0")
        return v

    @validator("resolution")
    def validate_resolution(cls, v):
        if v <= 0:
            raise ValueError("Resolution must be greater than 0")
        return v

    @root_validator
    def validate_atom_list_filter(cls, values):
        if values["mode"] == "all-atom":
            if values["atom_list_filter"] is None:
                values["atom_list_filter"] = "protein and not name H*"
        elif values["mode"] == "cg":
            if values["atom_list_filter"] is None:
                values["atom_list_filter"] = "protein"

        u = mda.Universe(values["ref_model_fname"])
        try:
            u.select_atoms(values["atom_list_filter"])
        except ValueError:
            raise ValueError(f"Invalid atom list filter {values['atom_list_filter']}")
        return values

    @validator("pipeline")
    def validate_pipeline(cls, v):
        for key in v.keys():
            if v[key]["type"] == "mdsampler":
                if v[key]["mode"] == "all-atom":
                    v[key] = PipelineMDSamplerAllAtomValidator(**v[key])

                elif v[key]["mode"] == "cg":
                    v[key] = PipelineMDSamplerCGValidator(**v[key])

                else:
                    raise ValueError(f"Invalid mode {v[key]['mode']}")

            elif v[key]["type"] == "weight_opt":
                v[key] = PipelineWeightOpValidator(**v[key])

            elif v[key]["type"] == "pos_opt":
                v[key] = PipelinePosOpValidator(**v[key])

            else:
                raise ValueError(f"Invalid pipeline element {v[key]}")

        return v
