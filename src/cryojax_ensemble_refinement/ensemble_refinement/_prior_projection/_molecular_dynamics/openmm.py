"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""

import glob
import os
import pathlib
from typing import Callable, Dict, List, Optional
from typing_extensions import override

import jax.numpy as jnp
import mdtraj
import openmm
import openmm.app as openmm_app
import openmm.unit as openmm_unit
from jaxtyping import Array, Float, Int
from natsort import natsorted

from ..base_prior_projector import AbstractPriorProjector


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


class SteeredMolecularDynamicsSimulator(AbstractPriorProjector):
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
        path_to_old_state_file: Optional[str] = None,
    ):
        pdb = openmm_app.PDBFile(str(path_to_initial_pdb))
        self._bias_constant_in_kj_per_mol_angs = bias_constant_in_kj_per_mol_angs
        self.restrain_atom_list = restrain_atom_list

        self.base_state_file_path = _validate_base_state_file_path(base_state_file_path)

        self.n_steps = n_steps
        parameters_for_md = _validate_and_set_params_for_md(parameters_for_md)

        if make_simulation_fn is None:
            self.simulation = _default_make_sim_fn(parameters_for_md, pdb.topology)
        else:
            self.simulation = make_simulation_fn(parameters_for_md, pdb.topology)

        if path_to_old_state_file is not None:
            assert pathlib.Path(path_to_old_state_file).exists(), (
                "path_to_old_state_file does not exist. "
                "Please set to None or provide valid state file."
            )
            self.simulation.loadState(str(path_to_old_state_file))

        else:
            self.simulation.context.setPositions(pdb.positions)
            self.simulation.minimizeEnergy()

        path_to_state_file = f"{self.base_state_file_path}0.xml"
        self.simulation.saveState(path_to_state_file)

        self.simulation = _add_restraint_force_to_simulation(
            self.simulation,
            self.simulation.context.getState(getPositions=True).getPositions(),
            self.restrain_atom_list,
            self.bias_constant_in_kj_per_mol_angs,
        )

    @property
    def bias_constant_in_kj_per_mol_angs(self) -> bool:
        return self._bias_constant_in_kj_per_mol_angs

    @bias_constant_in_kj_per_mol_angs.setter
    def bias_constant_in_kj_per_mol(self, value: Float):
        self._bias_constant_in_kj_per_mol_angs = value

    @override
    def __call__(
        self,
        positions_for_bias_in_angstroms: Float[Array, "n_atoms 3"],
    ):
        simulation = _remove_last_force_from_simulation(self.simulation)
        simulation = _add_restraint_force_to_simulation(
            simulation,
            mdtraj.Trajectory(
                positions_for_bias_in_angstroms / 10.0, self.simulation.topology
            ).openmm_positions(0),
            self.restrain_atom_list,
            self.bias_constant_in_kj_per_mol_angs,
        )

        simulation.context.reinitialize()

        path_to_state_file = _get_curr_state_file_path(self.base_state_file_path)
        simulation.loadState(path_to_state_file)
        simulation.step(self.n_steps)

        path_to_state_file = _get_new_state_file_path(self.base_state_file_path)
        simulation.saveState(path_to_state_file)

        positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )
        return jnp.array(positions) * 10.0  # Convert to Angstroms


def _validate_base_state_file_path(base_state_file_path: str) -> str:
    # check if the path exists
    base_dir = os.path.dirname(base_state_file_path)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if not base_state_file_path.endswith("_"):
        return f"{base_state_file_path}_it"
    else:
        return f"{base_state_file_path}it"


def _get_curr_state_file_path(
    base_state_file_path: str,
) -> str:
    # load all files in the directory that match the base path
    return natsorted(glob.glob(f"{base_state_file_path}*.xml"))[-1]


def _get_new_state_file_path(
    base_state_file_path: str,
) -> str:
    # load all files in the directory that match the base path
    last_file = natsorted(glob.glob(f"{base_state_file_path}*.xml"))[-1]
    # get the run counter from the last file
    last_counter = int(last_file.split(base_state_file_path)[-1].split(".xml")[0])
    return f"{base_state_file_path}{last_counter + 1}.xml"


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
