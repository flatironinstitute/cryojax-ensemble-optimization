import numpy as np
import os
from natsort import natsorted
import glob
from typing import List, Union, Optional
from pydantic import BaseModel, root_validator
import logging


class GeneratorConfig(BaseModel):
    experiment_name: str
    experiment_type: str
    mode: str
    particles_per_model: List[int]
    box_size: int
    resolution: float
    pixel_size: float
    defocus_u: Union[float, List[float]]
    noise_snr: Union[float, List[float]]
    working_dir: str
    output_path: str
    models_fname: str
    starfile_fname: str
    batch_size: int
    atom_list_filter: str
    defocus_v: Optional[Union[float, List[float]]] = None
    defocus_ang: Union[float, List[float]] = 0.0
    bfactor: Union[float, List[float]] = 0.0
    scalefactor: float = 1.0
    phaseshift: float = 0.0
    amp_contrast: float = 0.01
    volt: float = 300
    spherical_aberr: float = 2.7
    noise_radius_mask: Optional[float] = None
    noise_seed: int = 0

    @root_validator(pre=True)
    def set_none_defaults(cls, values):
        if values["defocus_v"] is None:
            values["defocus_v"] = values["defocus_u"]

        if values["noise_radius_mask"] is None:
            values["noise_radius_mask"] = values["box_size"] / 2.0

        return values

    @root_validator
    def validate_config_generator_req_values(cls, values):
        if values["mode"] not in ["all-atom", "resid", "coarse-grained"]:
            raise ValueError(
                "Invalid mode, must be 'all-atom', 'resid' or 'coarse-grained'"
            )

        if values["box_size"] <= 0:
            raise ValueError("Box size must be greater than 0")

        if values["resolution"] <= 0:
            raise ValueError("Resolution must be greater than 0")

        if values["pixel_size"] <= 0:
            raise ValueError("Pixel size must be greater than 0")

        if np.all(np.array(values["particles_per_model"]) <= 0):
            raise ValueError("Particles per model must be greater than 0")

        if np.all(np.array(values["defocus_u"]) < 0):
            raise ValueError("Defocus u must be greater or equal to 0")

        if np.all(np.array(values["defocus_v"]) < 0):
            raise ValueError("Defocus v must be greater or equal to 0")

        if np.all(np.array(values["noise_snr"]) <= 0):
            raise ValueError("Noise snr must be greater than 0")

        if values["batch_size"] <= 0:
            raise ValueError("Batch size must be greater than 0")

        if values["amp_contrast"] < 0 or values["amp_contrast"] > 1:
            raise ValueError("Amplitude contrast must be between 0 and 1")

        if values["volt"] <= 0:
            raise ValueError("Voltage must be greater than 0")

        if values["spherical_aberr"] <= 0:
            raise ValueError("Spherical aberration must be greater than 0")

        if values["noise_radius_mask"] <= 0:
            raise ValueError("Noise raidus mask must be greater than 0")

        if values["noise_radius_mask"] > values["box_size"]:
            raise ValueError("Noise raidus mask must be less than half of the box size")

        if not os.path.exists(values["working_dir"]):
            raise FileNotFoundError(
                f"Working directory {values['working_dir']} does not exist."
            )

        if "*" in values["models_fname"]:
            models_fname = natsorted(glob.glob(values["models_fname"]))
            if len(models_fname) == 0:
                raise FileNotFoundError(
                    f"No files found with pattern {values['models_fname']}"
                )
        else:
            models_fname = values["models_fname"]
            if not os.path.exists(models_fname):
                raise FileNotFoundError(f"Model {models_fname} does not exist.")

        for i in range(len(models_fname)):
            models_fname[i] = os.path.join(values["working_dir"], models_fname[i])

        values["models_fname"] = models_fname

        logging.info("Using the following models...")
        for i in range(len(models_fname)):
            logging.info("  ", values["models_fname"][i])

        return values
