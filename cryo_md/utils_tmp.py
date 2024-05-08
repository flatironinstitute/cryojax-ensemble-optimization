# write a json config loader

import json
import os
import sys
from numbers import Number
import numpy as np


def load_config(config_file):
    """
    Load a json config file

    Parameters
    ----------
    config_file : str
        Path to the config file

    Returns
    -------
    config : dict
    """
    if not os.path.isfile(config_file):
        print("Config file not found: {}".format(config_file))
        sys.exit(1)

    with open(config_file, 'r') as f:
        config = json.load(f)

    config = parse_config(config)

    return config

def parse_config_generator(config):

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
        "root_path": str,
        "output_path": str,
        "models_fname": str,
        "starfile_fname": str,
        "batch_size": Number,
        "atom_list_filter": str
    }

    for key in req_keys:
        if key not in config:
            raise ValueError("{} not found in config file".format(key))
        
        if not isinstance(config[key], req_keys[key]):
            raise ValueError("{} must be of type {}".format(key, req_keys[key]))

    
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

    for key in optional_keys:
        if key not in config:
            config[key] = optional_keys[key][1]
        elif not isinstance(config[key], optional_keys[key][0]):
            raise ValueError("{} must be of type {}".format(key, optional_keys[key][0]))

    return config

def validate_config_generator(config):

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
    
    if np.all(np.array(config["defocus_u"]) <= 0):
        raise ValueError("Defocus u must be greater than 0")
    
    if np.all(np.array(config["defocus_v"]) <= 0):
        raise ValueError("Defocus v must be greater than 0")
    
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
    
    return

def parse_config_optimizer(config):

    req_keys = {
        "experiment_name": str,
        "experiment_type": str,
        "root_path": str,
        "starfile_fname": str,
        "mode": str,
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
    }

    for key in req_keys:
        if key not in config:
            raise ValueError("{} not found in config file".format(key))
        
        if not isinstance(config[key], req_keys[key]):
            raise ValueError("{} must be of type {}".format(key, req_keys[key]))
        
    optional_keys = {
        "platform": [str, "CPU"],
        "platform_properties": [dict, {"Threads": 1}],
        "checkpoint_fname": [str, None],
    }

    for key in optional_keys:
        if key not in config:
            config[key] = optional_keys[key][1]
        elif not isinstance(config[key], optional_keys[key][0]):
            raise ValueError("{} must be of type {}".format(key, optional_keys[key][0]))
        
    return config

def parse_config(config):

    if "experiment_type" not in config:
        raise ValueError("experiment_type not found in config file")
    
    if config["experiment_type"] == "generator":
        config = parse_config_generator(config)
    
    elif config["experiment_type"] == "optimization":
        config = parse_config_optimizer(config)

    else:
        raise ValueError("experiment_type must be either generator or optimizer")
    
    return config


def help_config_generator():

    print("Required keys for generator:")
    print("experiment_name: name of the experiment")
    print("experiment_type: generator")
    print("particles_per_model: number of particles per model")
    print("box_size: box size of the particles")
    print("resolution: resolution of the particles")
    print("pixel_size: pixel size of the particles")
    print("defocus_u: defocus_u of the particles")
    print("noise_snr: noise_snr of the particles")
    print("root_path: path where the atomic models (pdb files) are located")
    print("output_path: output path for the generated data")
    print("models_fname: path to the atomic models (pdb files). The path should be relative to root_path, you can use * to indicate multiple files")
    print("starfile_fname: name of the starfile for output")
    print("batch_size: batch size of the output dataset")

    print("Optional keys for generator:")
    print("defocus_v: defocus_v of the particles")
    print("defocus_ang: defocus_ang of the particles")
    print("bfactor: bfactor of the particles")
    print("scalefactor: scalefactor of the particles")
    print("phase_shift: phase_shift of the particles")
    print("amp_contrast: amp_contrast of the particles")
    print("volt: volt of the particles")
    print("spherical_aberr: spherical_aberr of the particles")
    print("noise_radius_mask: noise_radius_mask of the particles")
    print("noise_seed: noise_seed of the particles")

    return

def help_config_optimizer():

    print("Required keys for optimizer:")
    print("experiment_name: name of the experiment")
    print("experiment_type: optimizer")
    print("root_path: root path of the dataset")
    print("starfile_fname: name of the starfile for output")
    print("init_models_fname: path to the initial atomic models (pdb files). The path should be relative to root_path, you can use * to indicate multiple files")
    print("ref_models_fname: path to the reference atomic models (pdb files). The path should be relative to root_path, you can use * to indicate multiple files")
    print("mdsampler_type: type of the MD sampler, can be either 'md' or 'mdgpu'")
    print("mdsampler_steps: number of MD steps")
    print("mdsampler_force_constant: force constant of the MD sampler")
    print("optimization_steps: number of optimization steps")
    print("weight_opt_steps: number of weight optimization steps")
    print("weight_opt_stepsize: stepsize of weight optimization")
    print("pos_opt_steps: number of position optimization steps")
    print("pos_opt_stepsize: stepsize of position optimization")
    print("batch_size: batch size of the output dataset")
    print("output_path: path for the output of the optimization")

    print("Optional keys for optimizer:")
    print("platform: platform of the MD sampler, can be either 'CPU' or 'GPU'")
    print("platform_properties: platform properties of the MD sampler, default is {'Threads': 1}")

    return