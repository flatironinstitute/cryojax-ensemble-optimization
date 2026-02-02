"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""

import logging
import os
import pathlib
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from typing_extensions import override

import jax.numpy as jnp
import mdtraj
import numpy as np
from jaxtyping import Array, Float, Int


try:
    import openmm
    import openmm.app as openmm_app
    import openmm.unit as openmm_unit

    _HAS_OPENMM = True

except ImportError:
    _HAS_OPENMM = False
    warnings.warn(
        "OpenMM is not installed. Please install OpenMM if using any features "
        + "that use molecular dynamics, e.g., ensemble optimization "
        + "or flexible fitting."
    )


from ..base_prior_projector import AbstractEnsemblePriorProjector, AbstractPriorProjector


def _get_default_md_params() -> Dict:
    return {
        "forcefield": "amber14-all.xml",
        "water_model": "amber14/tip3p.xml",
        "nonbondedMethod": openmm_app.PME,
        "nonbondedCutoff": 1.0 * openmm_unit.nanometer,
        "constraints": openmm_app.HBonds,
        "temperature": 300.0 * openmm_unit.kelvin,
        "friction": 1.0 / openmm_unit.picosecond,
        "timestep": 0.002 * openmm_unit.picoseconds,
        "platform": "CPU",
        "properties": {"Threads": "1"},
    }


class SteeredMDSimulator(AbstractPriorProjector, strict=True):
    n_steps: Int
    simulation: openmm_app.Simulation
    restrain_atom_list: List[Int]
    base_state_file_path: str

    def __init__(
        self,
        path_to_initial_pdb: str | pathlib.Path,
        n_steps: Int,
        restrain_atom_list: List[Int],
        parameters_for_md: Dict,
        base_state_file_path: str,
        *,
        make_simulation_fn: Optional[
            Callable[[Dict, openmm_app.Topology], openmm_app.Simulation]
        ] = None,
    ):
        if not _HAS_OPENMM:
            raise ImportError(
                "OpenMM is not installed. Please install OpenMM if using any features "
                + "that use molecular dynamics, e.g., the ensemble optimization pipeline "
                + "or flexible fitting."
            )
        pdb = openmm_app.PDBFile(str(path_to_initial_pdb))
        self.restrain_atom_list = restrain_atom_list

        self.base_state_file_path = _validate_base_state_file_path(base_state_file_path)

        self.n_steps = n_steps

        if make_simulation_fn is None:
            logging.info("Using default make_simulation_fn for OpenMM simulation.")
            parameters_for_md = _validate_and_set_params_for_md(parameters_for_md)
            self.simulation = _default_make_sim_fn(parameters_for_md, pdb.topology)

        else:
            self.simulation = make_simulation_fn(parameters_for_md, pdb.topology)

        self.simulation.context.setPositions(pdb.positions)

    @override
    def initialize(self, init_state: Optional[str] = None) -> str:
        if init_state is not None:
            assert pathlib.Path(init_state).exists(), (
                "init_state does not exist. "
                "Please set to None or provide valid state file."
            )
            self.simulation.loadState(str(init_state))
            path_to_state_file = f"{self.base_state_file_path}0.xml"
            if os.path.exists(path_to_state_file):
                if Path(init_state).samefile(path_to_state_file):
                    Warning(
                        "The provided init_state has the same base name as the "
                        + "base_state_file_path. "
                        + "This may cause overwriting of the state file."
                    )
                    path_to_state_file = f"{self.base_state_file_path}1.xml"

            logging.info(f"Initialized simulation from state file {init_state}.")

        else:
            logging.info("No initial state provided. Initializing from scratch.")
            path_to_state_file = f"{self.base_state_file_path}0.xml"
            logging.info("Minimizing energy and equilibrating...")
            self.simulation.minimizeEnergy()
            self.simulation.step(1000)
            logging.info("Equilibration done.")

        logging.info(f"Saving initial state to {path_to_state_file}...")
        self.simulation.saveState(path_to_state_file)

        return path_to_state_file

    @override
    def __call__(
        self, ref_walkers: Float[Array, "n_atoms 3"], state: str, bias_constant: float
    ) -> Tuple[Float[Array, "n_atoms 3"], str]:
        _assert_is_valid_state_file(state, self.base_state_file_path)

        simulation = _add_restraint_force_to_simulation(
            self.simulation,
            mdtraj.Trajectory(
                ref_walkers / 10.0, self.simulation.topology
            ).openmm_positions(0),
            self.restrain_atom_list,
            bias_constant,
        )

        # print("Reinitialize")
        simulation.context.reinitialize()

        # print("Loading state")
        simulation.loadState(state)

        # print("Running Simulation")
        simulation.step(self.n_steps)
        positions = simulation.context.getState(getPositions=True).getPositions()
        velocities = simulation.context.getState(getVelocities=True).getVelocities()
        # print("Cleaning up")
        simulation = _remove_last_force_from_simulation(simulation)
        simulation.context.reinitialize()  # preserveState=True)

        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)

        state = _get_next_state_file_path(self.base_state_file_path, state)
        simulation.saveState(state)

        # print("Saved states... Finishing.")

        positions = (
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(openmm_unit.angstrom)
        )
        return jnp.array(positions), state


class EnsembleSteeredMDSimulator(AbstractEnsemblePriorProjector, strict=True):
    projectors: List[SteeredMDSimulator]

    def __init__(self, md_simulators: List[SteeredMDSimulator]):
        if not _HAS_OPENMM:
            raise ImportError(
                "OpenMM is not installed. Please install OpenMM if using any features "
                + "that use molecular dynamics, e.g., the ensemble optimization pipeline "
                + "or flexible fitting."
            )
        self.projectors = md_simulators

    @override
    def __call__(
        self,
        ref_positions: Float[Array, "n_walkers n_atoms 3"],
        states: List[str],
        bias_constant: float,
    ) -> Tuple[Float[Array, "n_walkers n_atoms 3"], List[str]]:
        projected_walkers = np.zeros_like(ref_positions)
        for i, projector in enumerate(self.projectors):
            projected_walkers[i], states[i] = projector(
                ref_positions[i], states[i], bias_constant
            )
        return jnp.array(projected_walkers), states


def compute_biasing_constant(
    target_proportion: float,
    path_to_target_pdb: str,
    n_atoms_for_bias: int,
    *,
    equib_steps: int = 1000,
    steps_for_estimation: int = 500,
    make_simulation_fn: Optional[
        Callable[[Dict, openmm_app.Topology], openmm_app.Simulation]
    ] = None,
    parameters_for_md: Dict = {},
) -> float:
    """
    Compute the biasing constant `k` for a molecular dynamics simulation such that the
    average magnitude of the biasing force is a specified proportion of the average
    magnitude of the regular MD force.

    **Arguments:**
    - `target_proportion`:
        The desired proportion between the average magnitude of
        the biasing force and the average magnitude of the regular MD force.
        Recommended to use a value less than 1.0.
    - `path_to_target_pdb`:
        Path to the initial PDB file to set up the molecular dynamics simulation.
    - `n_atoms_for_bias`:
        Number of atoms to consider for biasing force computation.
    - `equib_steps`:
        Number of equilibration steps to run before force estimation. Defaults to 1000.
    - `steps_for_estimation`:
        Number of simulation steps to run for force estimation. Defaults to 2000.
    - `make_simulation_fn`:
        Optional function to create an OpenMM Simulation object. If not provided, a
        default function will be used.
    - `parameters_for_md`:
        Dictionary of parameters for setting up the molecular dynamics simulation.

    **Returns:**
    - `k_value`:
        The computed biasing constant `k` such that the average biasing force magnitude
        is `target_proportion` times the average regular MD force magnitude.
    """
    if not _HAS_OPENMM:
        raise ImportError(
            "OpenMM is not installed. Please install OpenMM if using any features "
            + "that use molecular dynamics, e.g., the ensemble optimization pipeline "
            + "or flexible fitting."
        )
    assert n_atoms_for_bias > 0, (
        "The number of atoms for biasing force computation must be greater than zero."
        " Please provide a " + "valid number of atoms."
    )

    if make_simulation_fn is None:
        parameters_for_md = _validate_and_set_params_for_md(parameters_for_md)
        make_simulation_fn = _default_make_sim_fn

    logging.info("  Computing biasing constant...")

    pdb = openmm_app.PDBFile(str(path_to_target_pdb))
    simulation = make_simulation_fn(parameters_for_md, pdb.topology)
    simulation.context.setPositions(pdb.positions)

    # Equilibrate system
    logging.info("    Equilibrating system...")
    logging.info("    Minimizing energy...")
    simulation.minimizeEnergy()
    logging.info("    Running MD steps...")
    simulation.step(equib_steps)
    logging.info("    Equilibration done.")

    # Run simulation for estimation, save trajectory
    dir_exists = pathlib.Path("./tmp_biasing_comp").exists()
    os.makedirs("./tmp_biasing_comp", exist_ok=True)
    path_to_traj = "./tmp_biasing_comp/traj_for_force.xtc"
    simulation.reporters.append(openmm_app.XTCReporter(path_to_traj, reportInterval=1))
    simulation.step(steps_for_estimation)

    # Load trajectory and compute value of the MD force
    traj = mdtraj.load(path_to_traj, top=path_to_target_pdb)

    simulation = make_simulation_fn(parameters_for_md, pdb.topology)
    md_forces = _compute_md_force(traj, simulation)
    md_forces_norm = np.linalg.norm(md_forces, axis=(1, 2))

    k_value = target_proportion * np.sqrt(n_atoms_for_bias) * md_forces_norm.mean()

    os.remove(path_to_traj)
    if not dir_exists:
        shutil.rmtree("./tmp_biasing_comp")

    return float(k_value)


def _compute_md_force(trajectory: mdtraj.Trajectory, simulation):
    forces = np.zeros((trajectory.n_frames, trajectory.n_atoms, 3))
    for i in range(trajectory.n_frames):
        simulation.context.setPositions(trajectory.openmm_positions(i))
        forces[i] = np.array(
            simulation.context.getState(getForces=True).getForces(asNumpy=True)
        )
    return forces


def _validate_base_state_file_path(base_state_file_path: str) -> str:
    # check if the path exists
    base_dir = os.path.dirname(base_state_file_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not base_state_file_path.endswith("_"):
        return f"{base_state_file_path}_it"
    else:
        return f"{base_state_file_path}it"


def _get_next_state_file_path(
    base_state_file_path: str,
    curr_state_file: str,
) -> str:
    # get the run counter from the last file
    last_counter = int(curr_state_file.split(base_state_file_path)[-1].split(".xml")[0])
    return f"{base_state_file_path}{last_counter + 1}.xml"


def _assert_is_valid_state_file(
    state_file: str,
    base_state_file_path: str,
) -> None:
    assert base_state_file_path in state_file, (
        "State file does not match base state file path. "
        "Please provide a valid state file."
    )

    counter = state_file.split(base_state_file_path)[-1].split(".xml")[0]
    try:
        int(counter)
    except ValueError:
        raise ValueError(
            f"State file should be formatted as base_state_file_path + <int>.xml. "
            f"Got {state_file} instead."
        )
    return


def _add_restraint_force_to_simulation(
    simulation: openmm_app.Simulation,
    positions: openmm_unit.Quantity,
    restrain_atom_list: List[int],
    bias_constant_in_kj_per_mol_angs: float,
) -> openmm_app.Simulation:
    RMSD_value = openmm.RMSDForce(
        positions,
        restrain_atom_list,
    )

    force_RMSD = openmm.CustomCVForce("k * RMSD")
    force_RMSD.addGlobalParameter("k", bias_constant_in_kj_per_mol_angs)
    force_RMSD.addCollectiveVariable("RMSD", RMSD_value)
    simulation.system.addForce(force_RMSD)

    return simulation


def _remove_last_force_from_simulation(
    simulation: openmm_app.Simulation,
) -> openmm_app.Simulation:
    n_forces = len(simulation.system.getForces())
    simulation.system.removeForce(n_forces - 1)
    return simulation


def _default_make_sim_fn(parameters_for_md: dict, topology) -> openmm_app.Simulation:
    forcefield = _create_forcefield(parameters_for_md)
    integrator = _create_integrator(parameters_for_md)
    platform = _create_platform(parameters_for_md)
    system = _create_system(parameters_for_md, forcefield, topology)

    simulation = openmm_app.Simulation(
        topology,
        system,
        integrator,
        platform,
        parameters_for_md["properties"],
    )

    return simulation


def _create_forcefield(parameters_for_md: dict) -> openmm_app.ForceField:
    return openmm_app.ForceField(
        parameters_for_md["forcefield"], parameters_for_md["water_model"]
    )


def _create_integrator(parameters_for_md: dict) -> openmm.Integrator:
    return openmm.LangevinIntegrator(
        parameters_for_md["temperature"],
        parameters_for_md["friction"],
        parameters_for_md["timestep"],
    )


def _create_system(
    parameters_for_md: dict,
    forcefield: openmm_app.ForceField,
    topology: openmm_app.Topology,
) -> openmm.System:
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=parameters_for_md["nonbondedMethod"],
        nonbondedCutoff=parameters_for_md["nonbondedCutoff"],
        constraints=parameters_for_md["constraints"],
    )

    return system


def _create_platform(parameters_for_md: dict) -> openmm.Platform:
    return openmm.Platform.getPlatformByName(parameters_for_md["platform"])


def _validate_and_set_params_for_md(
    parameters_for_md: dict,
) -> dict:
    default_md_params = _get_default_md_params()
    assert set(parameters_for_md.keys()).issubset(default_md_params)
    for key, value in default_md_params.items():
        if key not in parameters_for_md:
            parameters_for_md[key] = value

    return parameters_for_md
