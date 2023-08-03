import MDAnalysis as mda
import numpy as np
import jax.numpy as jnp
from jax.typing import ArrayLike
from MDAnalysis.analysis import align, rms
from typing import Tuple


def get_MD_outputs(
    n_models: int,
    directory_path: str,
    ref_universe: mda.Universe,
    filter: str,
) -> ArrayLike:
    """
    Get MD outputs from OpenMM simulations

    Parameters
    ----------
    n_models : int
        Number of models involved in the optimization
    directory_path : str
        Path to directory containing PDB files
    ref_universe : mda.Universe
        Reference universe for alignment
    filter : str
        Atom filter for MDAnalysis, e.g. "not name H*" or "name CA

    Returns
    -------
    opt_models : np.ndarray
        Aligned optimized models
    """

    opt_models = np.zeros(
        (n_models, *ref_universe.select_atoms(filter).atoms.positions.T.shape)
    )

    for i in range(n_models):
        pulling_traj = mda.Universe(
            f"{directory_path}/curr_system_{i}.pdb",
            f"{directory_path}/pull_traj_{i}.pdb",
        )

        pulling_traj.trajectory[-1]
        pulling_traj.atoms.write(f"{directory_path}/curr_system_{i}.pdb")

        opt_univ = pulling_traj.select_atoms("protein")
        align.alignto(opt_univ, ref_universe, select=filter, match_atoms=True)
        opt_models[i] = opt_univ.select_atoms(filter).atoms.positions.T

    return jnp.array(opt_models)


def dump_optimized_models(
    directory_path: str,
    opt_models,
    ref_universe: mda.Universe,
    filter: str,
    unit_cell: np.ndarray,
):
    """
    Dump optimized models to PDB files

    Parameters
    ----------
    directory_path : str
        Path to directory containing PDB files
    opt_models : np.ndarray
        Optimized models
    closest_indices : np.ndarray
        Indices of the closest frames in the trajectory to the optimized models
    unit_cell : np.ndarray
        Unit cell dimensions for PDB files

    Returns
    -------
    None
        New models are saved as ala_model_{i}.pdb, where i is the model number in the working directory
    """

    for i in range(opt_models.shape[0]):
        # Solvated ref
        solv_univ_ref = mda.Universe(f"{directory_path}/curr_system_{i}.pdb")

        # Replace in solvated model
        opt_sys = mda.Universe(f"{directory_path}/curr_system_{i}.pdb")
        opt_univ = opt_sys.select_atoms("protein")
        align.alignto(opt_univ, ref_universe, select=filter, match_atoms=True)
        opt_univ_subset = opt_univ.select_atoms(filter)
        opt_univ_subset.positions = opt_models[i].T

        # de-align for MD simulation and write
        align.alignto(opt_univ, solv_univ_ref, select="all", match_atoms=True)
        opt_univ.atoms.dimensions = unit_cell
        opt_univ.atoms.write(f"{directory_path}/curr_system_{i}_ref.pdb")

    return
