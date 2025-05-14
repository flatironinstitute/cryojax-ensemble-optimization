"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""

import pathlib
from abc import abstractmethod
from typing import List, Optional, Callable, Dict, Any

import equinox as eqx
import jax.numpy as jnp
import mdtraj
import openmm
import openmm.app as openmm_app
import openmm.unit as openmm_unit
from jaxtyping import Array, Float, Int


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


class AbstractMolecularDynamicsSimulator(eqx.Module, strict=True):
    @abstractmethod
    def __call__(self):
        return NotImplementedError


class SteeredMolecularDynamicsSimulator:
    n_steps: Int
    bias_constant_in_units: Float
    restrain_atom_list: List[Int]
    path_to_checkpoint: str | pathlib.Path

    def __init__(
        self,
        path_to_initial_pdb: str | pathlib.Path,
        bias_constant_in_units: Float,
        n_steps: Int,
        restrain_atom_list: List[Int],
        parameters_for_md: Dict,
        path_to_checkpoint: str | pathlib.Path,
        *,
        make_simulation_fn: Optional[Callable[[Dict, openmm_app.Topology], openmm_app.Simulation]] = None,
        continue_from_checkpoint: bool = False,
    ):
        
        if continue_from_checkpoint:
            assert pathlib.Path(path_to_checkpoint).exists(), (
                "Checkpoint file does not exist. "
                "Please set continue_from_checkpoint to False or provide a valid checkpoint file."
            )

        pdb = openmm_app.PDBFile(str(path_to_initial_pdb))
        self.bias_constant_in_units = bias_constant_in_units
        self.restrain_atom_list = restrain_atom_list

        self.n_steps = n_steps
        parameters_for_md = _validate_and_set_params_for_md(parameters_for_md)
    
        if make_simulation_fn is None:
            self.simulation = _default_make_sim_fn(
                parameters_for_md, pdb.topology
            )
        else:
            self.simulation = make_simulation_fn(
                parameters_for_md, pdb.topology
            )

        self.path_to_checkpoint = path_to_checkpoint



        if continue_from_checkpoint:
            self.simulation.loadCheckpoint(self.path_to_checkpoint)
    
        else:
            self.simulation.context.setPositions(pdb.positions)
            self.simulation.minimizeEnergy()
            self.simulation.saveCheckpoint(self.path_to_checkpoint)

        self.simulation = _add_restraint_force_to_simulation(
            self.simulation, pdb.positions, self.restrain_atom_list, self.bias_constant_in_units
        )



    def __call__(self, positions_for_bias_in_angstroms: Float[Array, "n_atoms 3"]):

        simulation = _remove_last_force_from_simulation(self.simulation)
        simulation = _add_restraint_force_to_simulation(
            simulation,
            mdtraj.Trajectory(
                positions_for_bias_in_angstroms / 10.0, self.simulation.topology
            ).openmm_positions(0),
            self.restrain_atom_list,
            self.bias_constant_in_units,
        )

        simulation.context.reinitialize()
        simulation.loadCheckpoint(self.path_to_checkpoint)

        simulation.step(self.n_steps)

        simulation.saveCheckpoint(self.path_to_checkpoint)
        positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )
        return jnp.array(positions) * 10.0  # Convert to angstroms


def _add_restraint_force_to_simulation(
    simulation: openmm_app.Simulation,
    positions: openmm_unit.Quantity,
    restrain_atom_list: List[int],
    bias_constant_in_units: float,
) -> openmm_app.Simulation:
    
    RMSD_value = openmm.RMSDForce(
        positions,
        restrain_atom_list,
    )

    force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
    force_RMSD.addGlobalParameter("k", bias_constant_in_units)
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
    system = _create_system(
        parameters_for_md, forcefield, topology
    )    

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
