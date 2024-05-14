from numbers import Number
import numpy as np
import os
from natsort import natsorted
import glob

from .validator_utils import validate_generic_config_req, validate_generic_config_opt


def validate_config_generator_req_values(config):
    if config["mode"] not in ["all-atom", "resid", "cg"]:
        raise ValueError("Invalid mode, must be 'all-atom', 'resid' or 'cg'")

    if config["box_size"] <= 0:
        raise ValueError("Box size must be greater than 0")

    if config["resolution"] <= 0:
        raise ValueError("Resolution must be greater than 0")

    if config["pixel_size"] <= 0:
        raise ValueError("Pixel size must be greater than 0")

    if np.all(np.array(config["particles_per_model"]) <= 0):
        raise ValueError("Particles per model must be greater than 0")

    if np.all(np.array(config["defocus_u"]) < 0):
        raise ValueError("Defocus u must be greater or equal to 0")

    if np.all(np.array(config["defocus_v"]) < 0):
        raise ValueError("Defocus v must be greater or equal to 0")

    if np.all(np.array(config["noise_snr"]) <= 0):
        raise ValueError("Noise snr must be greater than 0")

    if config["batch_size"] <= 0:
        raise ValueError("Batch size must be greater than 0")

    if config["amp_contrast"] < 0 or config["amp_contrast"] > 1:
        raise ValueError("Amplitude contrast must be between 0 and 1")

    if config["volt"] <= 0:
        raise ValueError("Voltage must be greater than 0")

    if config["spherical_aberr"] <= 0:
        raise ValueError("Spherical aberration must be greater than 0")

    if config["noise_radius_mask"] <= 0:
        raise ValueError("Noise raidus mask must be greater than 0")

    if config["noise_radius_mask"] > config["box_size"]:
        raise ValueError("Noise raidus mask must be less than half of the box size")
    
    if not os.path.exists(config["working_dir"]):
        raise FileNotFoundError(f"Working directory {config['working_dir']} does not exist.")
    
    if "*" in config["models_fname"]:
        models_fname = natsorted(glob.glob(config["models_fname"]))
        if len(models_fname) == 0:
            raise FileNotFoundError(f"No files found with pattern {config['models_fname']}")
    else:
        models_fname = config["models_fname"]
        if not os.path.exists(models_fname):
            raise FileNotFoundError(f"Model {models_fname} does not exist.")

    return

def read_generator_config(config: dict) -> dict:
    """
    Validate the config dictionary for the preprocessing pipeline.
    """
    req_keys = {
        "experiment_name": str,
        "experiment_type": str,
        "mode": str,
        "particles_per_model": (Number, list),
        "box_size": Number,
        "resolution": Number,
        "pixel_size": Number,
        "defocus_u": (Number, list),
        "noise_snr": (Number, list),
        "working_dir": str,
        "output_path": str,
        "models_fname": str,
        "starfile_fname": str,
        "batch_size": Number,
        "atom_list_filter": str,
    }

    validate_generic_config_req(config, req_keys)

    optional_keys = {
        "defocus_v": [(Number, list), config["defocus_u"]],
        "defocus_ang": [(Number, list), 0],
        "bfactor": [(Number, list), 0.0],
        "scalefactor": [Number, 1.0],
        "phaseshift": [Number, 0.0],
        "amp_contrast": [Number, 0.01],
        "volt": [Number, 300],
        "spherical_aberr": [Number, 2.7],
        "noise_radius_mask": [Number, config["box_size"] / 2.0],
        "noise_seed": [Number, 0],
    }
    config = validate_generic_config_opt(config, optional_keys)

    validate_config_generator_req_values(config)
    return config

