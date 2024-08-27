import os
import sys
import json
import textwrap
import logging

from ._io_validators.validate_generation_config import GeneratorConfig
from ._io_validators.validate_optimization_config import OptimizationConfig


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
    logging.info("Loaded config file: {}".format(config_file))

    if "experiment_type" not in config:
        raise ValueError("experiment_type not found in config file")

    if config["experiment_type"] == "generator":
        logging.info("Loading config in generator mode")
        config = GeneratorConfig(**config).dict()

    elif config["experiment_type"] == "optimization":
        logging.info("Loading config in optimization mode")
        config = OptimizationConfig(**config).dict()

    else:
        raise ValueError("experiment_type must be either generator or optimizer")

    config["output_path"] = os.path.join(
        config["output_path"], config["experiment_name"]
    )

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
