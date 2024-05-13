from numbers import Number
import os
from natsort import natsorted
import glob

from .validator_utils import validate_generic_config_req, validate_generic_config_opt

def validate_config_optimization_values(config):
    if config["mode"] not in ["all-atom", "resid", "cg"]:
        raise ValueError("Invalid mode, must be 'all-atom', 'resid' or 'cg'")

    if not os.path.exists(config["working_dir"]):
        raise FileNotFoundError(f"Working Directory {config['working_dir']} does not exist.")

    if not os.path.exists(config["starfile_fname"]):
        raise FileNotFoundError(f"Starfile {config['starfile_fname']} does not exist.")
    
    if "*" in config["init_models_fname"]:
        models_fname = natsorted(glob.glob(config["init_models_fname"]))
        if len(models_fname) == 0:
            raise FileNotFoundError(f"No files found with pattern {config['models_fname']}")
        else:
            models_fname = config["init_models_fname"]
            if not os.path.exists(models_fname):
                raise FileNotFoundError(f"Model {models_fname} does not exist.")
            
    if not os.path.exists(config["ref_model_fname"]):
        raise FileNotFoundError(f"Reference model {config['ref_model_fname']} does not exist.")
    
    if config["n_models"] <= 0:
        raise ValueError("Number of models must be greater than 0")
    
    if config["mdsampler_type"] not in ["all-atom", "cg"]:
        raise ValueError("Invalid mdsampler type, must be 'md-all-atom' or 'md-cg'")
    
    if config["mdsampler_steps"] <= 0:
        raise ValueError("Number of mdsampler steps must be greater than 0")
    
    if config["mdsampler_force_constant"] <= 0:
        raise ValueError("MD sampler force constant must be greater than 0")
    
    if config["optimization_steps"] <= 0:
        raise ValueError("Number of optimization steps must be greater than 0")
    
    if config["weight_opt_steps"] <= 0:
        raise ValueError("Number of weight optimization steps must be greater than 0")
    
    if config["weight_opt_stepsize"] <= 0:
        raise ValueError("Weight optimization stepsize must be greater than 0")
    
    if config["pos_opt_steps"] <= 0:
        raise ValueError("Number of position optimization steps must be greater than 0")
    
    if config["pos_opt_stepsize"] <= 0:
        raise ValueError("Position optimization stepsize must be greater than 0")
    
    if config["resolution"] <= 0:
        raise ValueError("Resolution must be greater than 0")
    
    return

def read_optimization_config(config: dict) -> dict:
    """
    Validate the config dictionary for the preprocessing pipeline.
    """
    req_keys = {
        "experiment_name": str,
        "experiment_type": str,
        "mode": str,
        "working_dir": str,
        "starfile_fname": str,
        "n_models": Number,
        "init_models_fname": str,
        "ref_model_fname": str,
        "mdsampler_type": str,
        "mdsampler_steps": Number,
        "mdsampler_force_constant": Number,
        "optimization_steps": Number,
        "weight_opt_steps": Number,
        "weight_opt_stepsize": Number,
        "pos_opt_steps": Number,
        "pos_opt_stepsize": Number,
        "batch_size": Number,
        "output_path": str,
        "resolution": float,
    }

    validate_generic_config_req(config, req_keys)

    optional_keys = {
        "platform": [str, "CPU"],
        "platform_properties": [dict, {"Threads": 1}],
        "checkpoint_fname": [str, None],
    }
    config = validate_generic_config_opt(config, optional_keys)

    validate_config_optimization_values(config)
    return config