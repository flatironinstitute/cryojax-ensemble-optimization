from numbers import Number
import os
from natsort import natsorted
import glob

from .validator_utils import validate_generic_config_req, validate_generic_config_opt


def validate_config_optimization_values(config):
    if config["mode"] not in ["all-atom", "resid", "cg"]:
        raise ValueError("Invalid mode, must be 'all-atom', 'resid' or 'cg'")

    if not os.path.exists(config["working_dir"]):
        raise FileNotFoundError(
            f"Working Directory {config['working_dir']} does not exist."
        )

    if not os.path.exists(config["starfile_fname"]):
        raise FileNotFoundError(f"Starfile {config['starfile_fname']} does not exist.")

    if "*" in config["models_fname"]:
        models_fname = natsorted(glob.glob(config["models_fname"]))
        if len(models_fname) == 0:
            raise FileNotFoundError(
                f"No files found with pattern {config['models_fname']}"
            )
    else:
        models_fname = [config["models_fname"]]
        if not os.path.exists(models_fname[0]):
            raise FileNotFoundError(f"Model {models_fname[0]} does not exist.")

    if config["n_models"] <= 0:
        raise ValueError("Number of models must be greater than 0")
    if config["n_models"] > len(models_fname):
        raise ValueError(
            "Number of models must be less than or equal to the number of models found."
        )

    if not os.path.exists(config["ref_model_fname"]):
        raise FileNotFoundError(
            f"Reference model {config['ref_model_fname']} does not exist."
        )

    if config["n_steps"] <= 0:
        raise ValueError("Number of steps must be greater than 0")

    if config["resolution"] <= 0:
        raise ValueError("Resolution must be greater than 0")

    return


def validate_pipeline_element(config: dict) -> dict:
    """
    Validate that each element in the pipeline config is valid
    """

    assert "type" in config.keys(), "Pipeline element must have a type"

    if config["type"] == "mdsampler":
        req_keys = {
            "mode": str,
            "mdsampler_steps": Number,
            "mdsampler_force_constant": Number,
            "n_steps": Number,
        }

        optional_keys = {
            "checkpoint_fname": [str, None],
            "platform": [str, "CPU"],
            "platform_properties": [dict, {"Threads": 1}],
        }

        if config["mode"] == "cg":
            req_keys["top_file"] = str
            optional_keys["epsilon_r"] = [float, 15.0]

    elif config["type"] == "weight_opt":
        req_keys = {
            "weight_opt_steps": Number,
            "weight_opt_stepsize": Number,
        }

        optional_keys = {}
    elif config["type"] == "pos_opt":
        req_keys = {
            "pos_opt_steps": Number,
            "pos_opt_stepsize": Number,
        }

        optional_keys = {}

    else:
        raise ValueError(f"Invalid pipeline element {config}")

    validate_generic_config_req(config, req_keys)
    config = validate_generic_config_opt(config, optional_keys)
    return config


def read_pipeline_config(config: dict) -> dict:
    """
    Validate the config dictionary for the optimization pipeline
    """

    n_keys = len(config.keys())
    try:
        req_keys = {f"{i}": dict for i in range(n_keys)}
        validate_generic_config_req(config, req_keys)
    except ValueError:
        req_keys = {i: dict for i in range(n_keys)}
        validate_generic_config_req(config, req_keys)

    for key in config.keys():
        config[key] = validate_pipeline_element(config[key])

    return config


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
        "models_fname": str,
        "ref_model_fname": str,
        "pipeline": dict,
        "batch_size": Number,
        "output_path": str,
        "resolution": float,
        "n_steps": Number,
    }

    validate_generic_config_req(config, req_keys)
    n_models = len(glob.glob(config["models_fname"]))
    optional_keys = {
        "checkpoint_fname": [str, None],
        "n_models": [Number, n_models],
    }
    config = validate_generic_config_opt(config, optional_keys)
    config["pipeline"] = read_pipeline_config(config["pipeline"])

    validate_config_optimization_values(config)
    return config
