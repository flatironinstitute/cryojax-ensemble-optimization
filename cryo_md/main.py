import MDAnalysis as mda
import logging
import pathlib
from typing import Optional

from cryo_md.pipeline.pipeline import Pipeline
from cryo_md.image.image_stack import NumpyLoader
from cryo_md.utils.parser import pdb_parser


def run_cryomd(
    pipeline: Pipeline,    
    image_stack: NumpyLoader,
    config: dict,
):

    logging.basicConfig(filename="cryo_md.log", level=logging.INFO)

    if config["optimization_steps"] <= 0:
        logging.warning("Number of steps must be greater than 0")
        raise ValueError("Number of steps must be greater than 0")

    if config["mode"] not in ["all-atom", "resid", "cg"]:
        logging.warning("Invalid mode, must be 'all-atom' or 'resid'")
        raise ValueError("Invalid mode, must be 'all-atom' or 'resid'")

    if config["mode"] in ["all-atom", "resid"]:
        filetype = "pdb"
    else:
        filetype = "gro"

    init_universes = []
    for i in range(config["n_models"]):
        init_universes.append(
            mda.Universe(config["init_models_fname"][i])
        )

    ref_universe = mda.Universe(config["ref_model_fname"])

    struct_info = pdb_parser(config["init_models_fname"][0], mode=config["mode"])

    logging.info("Preparing pipeline for run...")
    pipeline.prepare_for_run_(
        config["optimization_steps"], init_universes, struct_info, config["mode"], config["output_fname"], ref_universe
    )

    pipeline.run(image_stack)

    return
