import logging
import os

import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align

from ._pdb import pdb_parser


def _load_models_for_data_generator(config: dict) -> tuple[np.ndarray, dict]:
    logging.info(
        "Models will be loaded in the following order: {}".format(config["models_fnames"])
    )

    n_models = len(config["models_fnames"])

    model_0 = mda.Universe(config["models_fnames"][0])
    model_0.atoms.translate(-model_0.atoms.center_of_mass())

    logging.info(f"Using model {config['models_fnames'][0]} as reference.")
    path_ref_model = (
        os.path.join(config["path_to_relion_project"], "ref_model.")
        + os.path.basename(config["models_fnames"][0]).split(".")[-1]
    )
    model_0.atoms.write(path_ref_model)
    logging.info(f"Reference model written to {path_ref_model}")

    logging.info("Confirming that all models have the same structural information.")
    struct_info = pdb_parser(config["models_fnames"][0])
    for filename in config["models_fnames"]:
        tmp_struct_info = pdb_parser(filename)

        for key, value in tmp_struct_info.items():
            assert (
                struct_info[key] == value
            ).all(), f"Structural information mismatch between given models: {key} {value} {struct_info[key]}"
    logging.info("Structural information consistent across all models.")

    logging.info("Loading models.")
    models = []
    for i in range(0, n_models):
        uni = mda.Universe(config["models_fnames"][i])
        align.alignto(uni, model_0, select="protein and not name H*", weights="mass")
        models.append(uni.select_atoms("protein and not name H*").positions)
    logging.info("Models loaded.")

    models = np.array(models)

    return models, struct_info


def _load_models_for_optimizer(config: dict) -> dict[str, np.ndarray]:
    logging.info("Loading structural information for the first model.")
    struct_info = pdb_parser(config["models_fnames"][0])

    logging.info("Confirming that all models have the same structural information.")
    for filename in config["models_fnames"]:
        tmp_struct_info = pdb_parser(filename)

        for key, value in tmp_struct_info.items():
            assert (
                struct_info[key] == value
            ).all(), f"Structural information mismatch between given models: {key} {value} {struct_info[key]}"

    logging.info("Structural information consistent across all models.")

    logging.info("Confirming that reference model has the same structural information.")
    struct_info_ref = pdb_parser(config["ref_model_fname"])
    for key, value in struct_info_ref.items():
        assert (
            struct_info[key] == value
        ).all(), f"Structural information mismatch between reference model and given models: {key} {value} {struct_info[key]}"
    logging.info(
        "Structural information consistent between reference model and given models."
    )

    return struct_info
