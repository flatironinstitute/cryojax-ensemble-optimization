import MDAnalysis as mda
import logging
import pathlib
from typing import Optional

from cryo_md.pipeline import Pipeline
from cryo_md.image.image_stack import ImageStack
from cryo_md.utils.parser import pdb_parser


def run_cryomd(
    pipeline: Pipeline,
    mode: str,
    image_stack: ImageStack,
    n_models: int,
    n_steps: int,
    ref_universe: mda.Universe,
    path_to_models: Optional[str] = None,
    output_file: str = "outputs.h5",
):
    """
    Run cryo-MD pipeline

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline object containing workflow steps for the cryoMD method.
    mode : str
        Mode for parsing PDB files, either "all-atom" or "resid"
    image_stack : ImageStack
        Image stack object containing cryo-EM images
    n_models : int
        Number of models to optimize
    n_steps : int
        Number of optimization steps
    ref_universe : mda.Universe
        Reference universe for alignment
    path_to_models : str, optional
        Path to directory containing PDB files, by default None
    output_file : str, optional
        Name of output file, by default "outputs.h5"

    Raises
    ------
    ValueError
        If number of steps is less than or equal to 0
    ValueError
        If mode is not "all-atom" or "resid"

    Returns
    -------
    None
        Results will be saved to output file
    """

    logging.basicConfig(filename="cryo_md.log", level=logging.INFO)

    if n_steps <= 0:
        logging.warning("Number of steps must be greater than 0")
        raise ValueError("Number of steps must be greater than 0")

    if mode not in ["all-atom", "resid"]:
        logging.warning("Invalid mode, must be 'all-atom' or 'resid'")
        raise ValueError("Invalid mode, must be 'all-atom' or 'resid'")

    if path_to_models is None:
        path_to_models = str(pathlib.Path().resolve())
        logging.info(f"Path to models not specified, using {path_to_models}")

    init_universes = []
    for i in range(n_models):
        init_universes.append(mda.Universe(f"{path_to_models}/init_system_{i}.pdb"))

    struct_info = pdb_parser(f"{path_to_models}/init_system_0.pdb", mode=mode)

    logging.info("Preparing pipeline for run...")
    pipeline.prepare_for_run_(
        n_steps, init_universes, struct_info, mode, output_file, ref_universe
    )

    pipeline.run(image_stack)

    return
