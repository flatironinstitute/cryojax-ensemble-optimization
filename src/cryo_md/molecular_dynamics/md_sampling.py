"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""
import openmm
import openmm.app as openmm_app
import openmm.unit as openmm_unit


def run_md_openmm(
    n_models: int,
    atom_indices_restraint: list,
    directory_path: str,
    nsteps: int = 1000,
    stride: int = 5,
    restrain_force_constant: float = 1000.0,
    **kwargs,
):
    """
    Run MD simulations using OpenMM

    Parameters
    ----------
    n_models : int
        Number of models to run
    atom_indices_restraint : list
        List of atom indices for the RMSD restraint
    directory_path : str
        Path to directory containing PDB files
    nsteps : int, optional
        Number of MD steps, by default 1000
    stride : int, optional
        Stride for saving trajectory frames, by default 5
    restrain_force_constant : float, optional
        Force constant for the RMSD restraint, by default 1000.0
    **kwargs
        Additional arguments. Includes
        - model_topfile_prefix: str
            Prefix for the PDB file name, by defaul "curr_system_"
        - ref_topfile_prefix: str
            Prefix for the reference PDB file name, by default "ref_system_"
        - traj_fname_prefix: str
            Prefix for the trajectory file name, by default "system_traj_"
        - platform: str
            Platform to run MD on, by default "CPU". Other options include "CUDA" and "OpenCL".
        - properties: dict
            Properties for the platform, by default {"Threads": "1"}. Check other properties in the OpenMM documentation.
        - Temperature: float
            Temperature for the Langevin integrator, by default 300.0 * openmm.unit.kelvin
        - friction: float
            Friction coefficient for the Langevin integrator, by default 1.0 / openmm.unit.picosecond
        - timestep: float
            Timestep for the Langevin integrator, by default 0.002 * openmm.unit.picoseconds

    Returns
    -------
    None
        Trajectory files are saved in the directory specified by directory_path with name {traj_fname_prefix}_{i}.pdb
    """

    default_kwargs = {
        "model_topfile_prefix": "curr_system_",
        "ref_topfile_prefix": "ref_system_",
        "traj_fname_prefix": "system_traj_",
        "platform": "CPU",
        "properties": {"Threads": "1"},
        "Temperature": 300.0 * openmm_unit.kelvin,
        "friction": 1.0 / openmm_unit.picosecond,
        "timestep": 0.002 * openmm_unit.picoseconds,
    }

    for key in kwargs:
        if key not in default_kwargs:
            raise ValueError(f"Invalid argument {key}")

    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

    for i in range(n_models):
        # Running dynamics

        forcefield = openmm_app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        pdb = openmm_app.PDBFile(
            f"{directory_path}/curr_system_{i}.pdb"
        )
        pdb_ref = openmm_app.PDBFile(
            f"{directory_path}/curr_system_{i}_ref.pdb"
        )

        pdb_reporter = openmm_app.PDBReporter(
            f"{directory_path}/pull_traj_{i}.pdb", stride
        )

        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=openmm_app.PME,
            nonbondedCutoff=1.0 * openmm_unit.nanometers,
            constraints=openmm_app.HBonds,
        )

        # Add RMSD restraint

        RMSD_value = openmm.RMSDForce(pdb_ref.positions, atom_indices_restraint)

        force_RMSD = openmm.CustomCVForce("0.5 * k * (RMSD - r0)^2")
        force_RMSD.addGlobalParameter("k", restrain_force_constant)
        force_RMSD.addGlobalParameter("r0", 0.0)
        force_RMSD.addCollectiveVariable("RMSD", RMSD_value)

        system.addForce(force_RMSD)

        integrator = openmm.LangevinIntegrator(
            kwargs["Temperature"], kwargs["friction"], kwargs["timestep"]
        )

        platform = openmm.Platform.getPlatformByName(kwargs["platform"])
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform, kwargs["properties"]
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(pdb_reporter)
        # simulation.reporters.append(openmm_app.StateDataReporter(f"ala_dynamics_{i}.csv", 5, step=True, potentialEnergy=True, temperature=True))
        simulation.step(nsteps)

    return
