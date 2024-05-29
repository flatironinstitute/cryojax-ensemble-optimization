import os
import sys
import json
import numpy as np
import textwrap
import glob
from natsort import natsorted
import logging

from ._io_validators.validate_generation_config import read_generator_config
from ._io_validators.validate_optimization_config import read_optimization_config

def parse_structure_fnames(config):
    if "*" in config["models_fname"]:
        models_fname = natsorted(glob.glob(config["models_fname"]))
    else:
        models_fname = [config["models_fname"]]

    for i in range(len(models_fname)):
        models_fname[i] = os.path.join(config["working_dir"], models_fname[i])

    config["models_fname"] = models_fname
    logging.info(f"Using the following models...")
    for i in range(len(models_fname)):
        logging.info("  ", config["models_fname"][i])
        
    return config

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

    with open(config_file, "r") as f:
        config = json.load(f)

    if "experiment_type" not in config:
        raise ValueError("experiment_type not found in config file")

    if config["experiment_type"] == "generator":
        config = read_generator_config(config)
        config["defocus_ang"] = list(np.radians(config["defocus_ang"]))

    elif config["experiment_type"] == "optimization":
        config = read_optimization_config(config)

    else:
        raise ValueError("experiment_type must be either generator or optimizer")

    config = parse_structure_fnames(config)
    config["output_path"] = os.path.join(config["output_path"], config["experiment_name"])
    
    return config


def help_config_generator():
    string = textwrap.dedent(
        """\
        Required keys in config file:
            experiment_name: name of the experiment, used for logging
            particles_per_model: number of particles per model as a list [n_particles_1, n_particles_2, ...]
            box_size: box size of the particles
            pixel_size: pixel size of the particles
            defocus_u: defocus_u of the particles (Angstrom)
            working_dir: path where the atomic models (pdb files) are located
            output_path: output path for the generated data
            models_fname: path to the atomic models (pdb files). The path should be relative to working_dir, you can use * to indicate multiple files
            starfile_fname: name of the starfile for output
            batch_size: batch size of the output dataset

        Optional keys in config file:
            defocus_v: defocus_v of the particles (Angstrom) (default: defocus_u)
            defocus_ang: defocus_ang of the particles (degrees) (default: 0°)
            bfactor: bfactor of the particles (Angstrom^2) (default: 0.0 Angstrom^2)
            scalefactor: scalefactor of the particles (default: 1.0)
            phase_shift: phase_shift of the particles (default: 0.0)
            amp_contrast: amp_contrast of the particles (default: 0.01)
            volt: volt of the particles (kv) (default: 300 kV)
            spherical_aberr: spherical_aberr of the particles (mm) (default: 2.7 mm)
            seed: seed for parameter generation (default: 0)

    """
    )
    return string
