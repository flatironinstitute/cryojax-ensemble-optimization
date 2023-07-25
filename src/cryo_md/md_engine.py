import openmm
import openmm.app as openmm_app
import openmm.unit as openmm_unit
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
from tqdm import tqdm


def run_md_openmm(
    n_models: int,
    directory_path: str,
    nsteps: int = 1000,
    stride: int = 5,
    device: str = "CPU",
):
    """
    Run MD simulations using OpenMM

    Parameters
    ----------
    n_models : int
        Number of models to run
    directory_path : str
        Path to directory containing PDB files
    nsteps : int, optional
        Number of MD steps, by default 1000
    stride : int, optional
        Stride for saving trajectory frames, by default 5
    device : str, optional
        Device to run MD on, by default "CPU"

    Returns
    -------
    None
        Trajectories are saved as ala_traj_{i}.pdb, where i is the model number in the working directory
    """

    for i in range(n_models):
        # Running dynamics

        forcefield = openmm_app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        pdb = openmm_app.PDBFile(f"{directory_path}/ala_model_{i}.pdb")
        pdb_ref = openmm_app.PDBFile(f"{directory_path}/ala_model_{i}_ref.pdb")

        # Create index list for non-solvent and non-ion atoms
        indexlist_protein = []
        indexlist_protein_H = []

        for atom in pdb.topology.atoms():
            if (
                atom.residue.name != "HOH"
                and atom.residue.name != "NA"
                and atom.residue.name != "CL"
            ):
                if "H" not in atom.name:
                    indexlist_protein_H.append(atom.index)

                elif "CH" in atom.name:
                    indexlist_protein_H.append(atom.index)

                indexlist_protein.append(atom.index)

        pdb_reporter = openmm_app.PDBReporter(
            f"{directory_path}/ala_traj_{i}.pdb", stride
        )
        pdb_reporter._atomSubset = indexlist_protein

        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=openmm_app.PME,
            nonbondedCutoff=1.0 * openmm_unit.nanometers,
            constraints=openmm_app.HBonds,
        )

        # Add RMSD restraint

        RMSD_value = openmm.RMSDForce(pdb_ref.positions, indexlist_protein_H)

        force_RMSD = openmm.CustomCVForce("0.5 * k * (RMSD - r0)^2")
        force_RMSD.addGlobalParameter("k", 100000.0)
        force_RMSD.addGlobalParameter("r0", 0.0)
        force_RMSD.addCollectiveVariable("RMSD", RMSD_value)

        system.addForce(force_RMSD)

        integrator = openmm.LangevinIntegrator(
            300 * openmm_unit.kelvin,
            1 / openmm_unit.picosecond,
            0.002 * openmm_unit.picoseconds,
        )

        platform = openmm.Platform.getPlatformByName(device)
        properties = {"Threads": "16"}
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform, properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(pdb_reporter)
        # simulation.reporters.append(openmm_app.StateDataReporter(f"ala_dynamics_{i}.csv", 5, step=True, potentialEnergy=True, temperature=True))
        simulation.step(nsteps)

    return


def process_outputs(n_models: int, directory_path: str, filter: str = "not name H*"):
    """
    Process MD outputs

    Parameters
    ----------
    n_models : int
        Number of models to run
    directory_path : str
        Path to directory containing PDB files
    filter : str, optional
        Atom filter for MDAnalysis, by default "not name H*"

    Returns
    -------
    samples : np.ndarray
        Aligned trajectory samples
    opt_models : np.ndarray
        Aligned optimized models
    """

    ala_start = mda.Universe(f"{directory_path}/ala_start.pdb")
    n_frames = mda.Universe(
        f"{directory_path}/ala_model_0.pdb", f"{directory_path}/ala_traj_0.pdb"
    ).trajectory.n_frames
    samples = np.zeros(
        (n_frames, n_models, *ala_start.select_atoms(filter).atoms.positions.T.shape)
    )
    opt_models = np.zeros(
        (n_models, *ala_start.select_atoms(filter).atoms.positions.T.shape)
    )

    # prot_trajs = []

    for i in range(n_models):
        traj = mda.Universe(
            f"{directory_path}/ala_model_{i}.pdb", f"{directory_path}/ala_traj_{i}.pdb"
        )
        traj.select_atoms("protein").write(
            f"{directory_path}/ala_traj_prot_{i}.pdb", frames="all"
        )
        traj_prot = mda.Universe(
            f"{directory_path}/ala_start.pdb",
            f"{directory_path}/ala_traj_prot_{i}.pdb",
            in_memory=True,
        )

        align.AlignTraj(
            traj_prot,  # trajectory to align,
            ala_start,  # reference,
            select="protein",  # selection of atoms to align,
            in_memory=True,
            match_atoms=True,  # whether to match atoms based on mass
        ).run()

        for j in range(len(traj.trajectory)):
            traj_prot.trajectory[j]
            samples[j, i] = traj_prot.select_atoms(filter).atoms.positions.T

        opt_traj = mda.Universe(f"{directory_path}/ala_prot_{i}.pdb")
        align.alignto(opt_traj, ala_start, select="all", match_atoms=True)
        opt_models[i] = opt_traj.select_atoms(filter).atoms.positions.T

        # prot_trajs.append(traj_prot)

    return samples, opt_models


def dump_new_models(
    directory_path: str,
    opt_models: np.ndarray,
    closest_indices: np.ndarray,
    unit_cell: np.ndarray,
):
    """
    Dump new models to PDB files

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
        ala_prot = mda.Universe(f"{directory_path}/ala_prot_{i}.pdb")
        ala_system = mda.Universe(f"{directory_path}/ala_model_{i}.pdb")

        ala_prot_noH = ala_prot.select_atoms("not name H*")
        ala_prot_noH.positions = opt_models[i].T

        align.alignto(
            ala_prot, ala_system.select_atoms("protein"), select="all", match_atoms=True
        )

        ala_system_prot = ala_system.select_atoms("protein")
        ala_system_prot.positions = ala_prot.atoms.positions
        ala_system.atoms.dimensions = unit_cell
        ala_system.atoms.write(f"{directory_path}/ala_model_{i}_ref.pdb")

        traj = mda.Universe(
            f"{directory_path}/ala_model_{i}.pdb", f"{directory_path}/ala_traj_{i}.pdb"
        )

        traj.trajectory[closest_indices[i]]
        traj.atoms.dimensions = unit_cell
        traj.atoms.write(f"{directory_path}/ala_model_{i}.pdb")

    return


"""
def dump_new_models(directory_path, indices, unit_cell):

    ala_start = mda.Universe(f"{directory_path}/ala_start.pdb")

    for i in range(indices.shape[0]):

        traj = mda.Universe(
            f"{directory_path}/ala_model_{i}.pdb", f"{directory_path}/ala_traj_{i}.pdb"
        )

        traj.trajectory[indices[i]]
        traj.atoms.dimensions = unit_cell
        traj.atoms.write(f"{directory_path}/ala_model_{i}.pdb")

        traj_prot = mda.Universe(
            f"{directory_path}/ala_start.pdb",
            f"{directory_path}/ala_traj_prot_{i}.pdb",
            in_memory=True,
        )

        align.AlignTraj(
            traj_prot,  # trajectory to align,
            ala_start,  # reference,
            select="protein",  # selection of atoms to align,
            in_memory=True,
            match_atoms=True,  # whether to match atoms based on mass
        ).run()

        traj_prot.trajectory[indices[i]]
        traj_prot.atoms.write(f"{directory_path}/ala_prot_{i}.pdb")

    return
"""
