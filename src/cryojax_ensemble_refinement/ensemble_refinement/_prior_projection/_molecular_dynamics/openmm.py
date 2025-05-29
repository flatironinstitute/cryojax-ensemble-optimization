"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""
import os
import pathlib
from functools import partial
from pathlib import Path
import shutil
from typing import Callable, Dict, List, Optional, Tuple
from typing_extensions import override

import jax
import jax.numpy as jnp
import mdtraj
import numpy as np
import openmm
import openmm.app as openmm_app
import openmm.unit as openmm_unit
from jaxtyping import Array, Float, Int, PRNGKeyArray


from ..base_prior_projector import AbstractEnsemblePriorProjector, AbstractPriorProjector


DEFAULT_MD_PARAMS = {
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
    _bias_constant_in_kj_per_mol_angs: Float
    simulation: openmm_app.Simulation
    restrain_atom_list: List[Int]
    base_state_file_path: str

    def __init__(
        self,
        path_to_initial_pdb: str | pathlib.Path,
        bias_constant_in_kj_per_mol_angs: Float,
        n_steps: Int,
        restrain_atom_list: List[Int],
        parameters_for_md: Dict,
        base_state_file_path: str,
        *,
        make_simulation_fn: Optional[
            Callable[[Dict, openmm_app.Topology], openmm_app.Simulation]
        ] = None,
    ):
        pdb = openmm_app.PDBFile(str(path_to_initial_pdb))
        self._bias_constant_in_kj_per_mol_angs = bias_constant_in_kj_per_mol_angs
        self.restrain_atom_list = restrain_atom_list

        self.base_state_file_path = _validate_base_state_file_path(base_state_file_path)

        self.n_steps = n_steps
        if make_simulation_fn is None:
            parameters_for_md = _validate_and_set_params_for_md(parameters_for_md)
            self.simulation = _default_make_sim_fn(parameters_for_md, pdb.topology)

        else:
            self.simulation = make_simulation_fn(parameters_for_md, pdb.topology)

        self.simulation.context.setPositions(pdb.positions)

        # self.simulation = _add_restraint_force_to_simulation(
        #     self.simulation,
        #     self.simulation.context.getState(getPositions=True).getPositions(),
        #     self.restrain_atom_list,
        #     1.0,
        # )

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

        else:
            path_to_state_file = f"{self.base_state_file_path}0.xml"
            self.simulation.minimizeEnergy()

        self.simulation.saveState(path_to_state_file)

        return path_to_state_file

    @property
    def bias_constant_in_kj_per_mol_angs(self) -> float:
        return self._bias_constant_in_kj_per_mol_angs

    @bias_constant_in_kj_per_mol_angs.setter
    def bias_constant_in_kj_per_mol(self, value: Float):
        self._bias_constant_in_kj_per_mol_angs = value

    @override
    def __call__(
        self,
        key: PRNGKeyArray,
        ref_walkers: Float[Array, "n_atoms 3"],
        state: str,
    ) -> Tuple[Float[Array, "n_atoms 3"], str]:
        _assert_is_valid_state_file(state, self.base_state_file_path)

        simulation = _add_restraint_force_to_simulation(
            self.simulation,
            mdtraj.Trajectory(
                ref_walkers / 10.0, self.simulation.topology
            ).openmm_positions(0),
            self.restrain_atom_list,
            self.bias_constant_in_kj_per_mol_angs,
        )

        print("Reinitialize")
        simulation.context.reinitialize()

        print("Loading state")
        simulation.loadState(state)

        platform = simulation.context.getPlatform()
        print(platform.getPropertyValue(simulation.context, "Threads"))

        print("Running Simulation")
        simulation.step(self.n_steps)
        positions = simulation.context.getState(getPositions=True).getPositions()
        velocities = simulation.context.getState(getVelocities=True).getVelocities()
        print("Cleaning up")
        simulation = _remove_last_force_from_simulation(simulation)
        simulation.context.reinitialize()  # preserveState=True)

        simulation.context.setPositions(positions)
        simulation.context.setVelocities(velocities)

        state = _get_next_state_file_path(self.base_state_file_path, state)
        simulation.saveState(state)

        print("Saved states... Finishing.")

        positions = (
            simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(openmm_unit.angstrom)
        )
        return jnp.array(positions), state


class EnsembleSteeredMDSimulator(AbstractEnsemblePriorProjector, strict=True):
    projectors: List[SteeredMDSimulator]

    def __init__(self, md_simulators: List[SteeredMDSimulator]):
        self.projectors = md_simulators

    @override
    def __call__(
        self,
        key: PRNGKeyArray,
        ref_positions: Float[Array, "n_walkers n_atoms 3"],
        states: List[str],
    ) -> Tuple[Float[Array, "n_walkers n_atoms 3"], List[str]]:
        keys = jax.random.split(key, len(self.projectors))
        projected_walkers = np.zeros_like(ref_positions)
        for i, projector in enumerate(self.projectors):
            projected_walkers[i], states[i] = projector(
                keys[i], ref_positions[i], states[i]
            )
        return jnp.array(projected_walkers), states


def compute_biasing_constant(
    target_percentage: Float,
    path_to_initial_pdb: str,
    positions_for_bias,
    n_steps: Int,
    stride: Int = 1,
    atom_selection: str = "not element H",
    *,
    make_simulation_fn: Optional[
        Callable[[Dict, openmm_app.Topology], openmm_app.Simulation]
    ] = None,
    parameters_for_md: Dict = {},
):  
    if make_simulation_fn is None:
        parameters_for_md = _validate_and_set_params_for_md(parameters_for_md)
        make_simulation_fn = _default_make_sim_fn

    restrain_atom_list = mdtraj.load(str(path_to_initial_pdb)).topology.select(atom_selection)

    pdb = openmm_app.PDBFile(str(path_to_initial_pdb))
    simulation = make_simulation_fn(
        parameters_for_md, pdb.topology
    )
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()

    # initial_positions = simulation.context.getState(
    #     getPositions=True
    # ).getPositions()

    #initial_positions = mdtraj.load(str(path_to_initial_pdb)).openmm_positions(0)

    dir_exists = pathlib.Path("./tmp_biasing_comp").exists()
    os.makedirs("./tmp_biasing_comp", exist_ok=True)
    path_to_traj = "./tmp_biasing_comp/traj_for_force.xtc"
    simulation.reporters.append(
        openmm_app.XTCReporter(path_to_traj, reportInterval=stride)
    )
    simulation.step(n_steps)

    traj = mdtraj.load(path_to_traj, top=path_to_initial_pdb)

    simulation = make_simulation_fn(parameters_for_md, pdb.topology)
    md_forces = _compute_regular_force(traj, simulation)

    simulation = make_simulation_fn(parameters_for_md, pdb.topology)
    bias_forces = _compute_biasing_force(
        traj, simulation, restrain_atom_list, positions_for_bias
    )

    k_value = _compute_k_value(
        jnp.array(md_forces),
        jnp.array(bias_forces),
        target_percentage,
        restrain_atom_list
    ).mean()

    os.remove(path_to_traj)
    if not dir_exists:
        shutil.rmtree("./tmp_biasing_comp")

    return k_value

@partial(jax.vmap, in_axes=(0, 0, None, None))
def _compute_k_value(base_force, bias_force, target_percentage, restrain_atom_list):
    force1 = base_force[restrain_atom_list, :].flatten()
    force2 = bias_force[restrain_atom_list, :].flatten()

    rho = jnp.dot(force1, force2) / jnp.sum(force2 ** 2)
    R = jnp.sum(force1 ** 2) / jnp.sum(force2 ** 2)

    a = (1.0 - target_percentage ** 2)
    b = - 2.0 * target_percentage ** 2 * rho
    c = - R * target_percentage ** 2

    return -b + jnp.sqrt(b ** 2 - 4.0 * a * c) / (2.0 * a)


def _compute_regular_force(trajectory: mdtraj.Trajectory, simulation):
    forces = np.zeros((trajectory.n_frames, trajectory.n_atoms, 3))
    for i in range(trajectory.n_frames):
        simulation.context.setPositions(trajectory.openmm_positions(i))
        forces[i] = np.array(
            simulation.context.getState(getForces=True).getForces(asNumpy=True)
        )
    return forces


def _compute_biasing_force(
    trajectory: mdtraj.Trajectory,
    simulation,
    restrain_atom_list,
    bias_positions,
):
    RMSD_value = openmm.RMSDForce(
        bias_positions,
        restrain_atom_list,
    )

    force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
    force_RMSD.addGlobalParameter("k", 1.0)
    force_RMSD.addCollectiveVariable("RMSD", RMSD_value)
    simulation.system.addForce(force_RMSD)

    n_forces = len(simulation.system.getForces())
    for i in range(n_forces - 1):
        simulation.system.removeForce(0)

    simulation.context.reinitialize()
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

    force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
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
    assert set(parameters_for_md.keys()).issubset(DEFAULT_MD_PARAMS)
    for key, value in DEFAULT_MD_PARAMS.items():
        if key not in parameters_for_md:
            parameters_for_md[key] = value

    return parameters_for_md
