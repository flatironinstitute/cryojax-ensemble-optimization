import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis import align, rms
from typing import Tuple


def get_MD_outputs(
    n_models: int,
    directory_path: str,
    ref_universe: mda.Universe,
    model_topfile_prefix: str = "curr_system_",
    traj_fname_prefix: str = "system_traj_",
    filter: str = "not name H*",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get MD outputs from OpenMM simulations

    Parameters
    ----------
    n_models : int
        Number of models involved in the optimization
    directory_path : str
        Path to directory containing PDB files
    model_topfile_prefix: str
        Prefix for the PDB file name, by defaul "curr_system_"
    ref_topfile_prefix: str
        Prefix for the reference PDB file name, by default "ref_system_"
    traj_fname_prefix: str
        Prefix for the trajectory file name, by default "system_traj_"
    ref_universe : mda.Universe
        Reference universe for alignment
    filter : str, optional
        Atom filter for MDAnalysis, by default "not name H*"

    Returns
    -------
    samples : np.ndarray
        Aligned trajectory samples
    opt_models : np.ndarray
        Aligned optimized models
    """

    n_frames = mda.Universe(
        f"{directory_path}/{model_topfile_prefix}0.pdb",
        f"{directory_path}/{traj_fname_prefix}0.pdb",
    ).trajectory.n_frames

    samples = np.zeros(
        (n_frames, n_models, *ref_universe.select_atoms(filter).atoms.positions.T.shape)
    )

    opt_models = np.zeros(
        (n_models, *ref_universe.select_atoms(filter).atoms.positions.T.shape)
    )

    for i in range(n_models):
        traj_prot = mda.Universe(
            f"{directory_path}/{model_topfile_prefix}{i}.pdb",
            f"{directory_path}/{traj_fname_prefix}{i}.pdb",
        )

        align.AlignTraj(
            traj_prot,  # trajectory to align,
            ref_universe,  # reference,
            select=f"protein and {filter}",  # selection of atoms to align,
            in_memory=True,
            match_atoms=True,  # whether to match atoms based on mass
        ).run()

        for j in range(n_frames):
            traj_prot.trajectory[j]
            samples[j, i] = traj_prot.select_atoms(filter).atoms.positions.T

        opt_traj = mda.Universe(f"{directory_path}/{model_topfile_prefix}{i}.pdb")
        align.alignto(opt_traj, ref_universe, select=filter, match_atoms=True)
        opt_models[i] = opt_traj.select_atoms(filter).atoms.positions.T
        opt_traj.atoms.write(f"{directory_path}/ala_prot_{i}.pdb")

    return samples, opt_models


def dump_optimized_models(
    directory_path: str,
    opt_models,
    ref_universe: mda.Universe,
    unit_cell: np.ndarray,
    model_topfile_prefix: str = "curr_system_",
    pull_model_topfile_prefix: str = "pull_model_",
    traj_fname_prefix: str = "system_traj_",
    filter: str = "not name H*",
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
        # Dump pulling model
        optim_univ = mda.Universe(f"{directory_path}/{model_topfile_prefix}{i}.pdb")
        align.alignto(optim_univ, ref_universe, select=filter, match_atoms=True)

        curr_model = mda.Universe(f"{directory_path}/{model_topfile_prefix}{i}.pdb")

        optim_univ_subset = optim_univ.select_atoms(filter)
        optim_univ_subset.positions = opt_models[i].T

        align.alignto(optim_univ, curr_model, select=filter, match_atoms=True)

        optim_univ.atoms.dimensions = unit_cell
        optim_univ.atoms.write(f"{directory_path}/{pull_model_topfile_prefix}{i}.pdb")

        # Dump new model for pulling
        traj_prot = mda.Universe(
            f"{directory_path}/{model_topfile_prefix}{i}.pdb",
            f"{directory_path}/{traj_fname_prefix}{i}.pdb",
        )

        traj_prot.trajectory[-1]
        traj_prot.atoms.dimensions = unit_cell
        traj_prot.atoms.write(f"{directory_path}/{model_topfile_prefix}{i}.pdb")

    return


def process_pulling_trajectory(directory_path: str, n_models, unit_cell: np.ndarray):
    for i in range(n_models):
        pull_traj = mda.Universe(
            f"{directory_path}/ala_model_{i}.pdb",
            f"{directory_path}/ala_traj_{i}_pull.pdb",
            in_memory=True,
        )
        pull_ref = mda.Universe(f"{directory_path}/ala_model_{i}_pull.pdb")

        align.AlignTraj(
            pull_traj, pull_ref, select="not name H*", in_memory=True, match_atoms=True
        ).run()

        rmsd_pulling = (
            rms.RMSD(pull_traj, pull_ref, select="not name H*", weights="mass")
            .run()
            .rmsd[:, 2]
        )

        pull_traj.trajectory[np.argmin(rmsd_pulling)]
        pull_traj.atoms.dimensions = unit_cell
        pull_traj.atoms.write(f"{directory_path}/ala_model_{i}.pdb")

    return
