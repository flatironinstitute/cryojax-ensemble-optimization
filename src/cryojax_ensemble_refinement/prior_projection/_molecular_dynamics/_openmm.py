"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""

import pathlib
from abc import abstractmethod
from typing import List, Optional

import equinox as eqx
import mdtraj
import numpy as np
import jax.numpy as jnp
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
    modeller: openmm_app.Modeller
    bias_constant_in_units: Float
    restrain_atom_list: List[Int]
    forcefield: openmm_app.ForceField
    platform: openmm.Platform
    platform_properties: dict
    path_to_checkpoint: str | pathlib.Path
    parameters_for_md: dict

    def __init__(
        self,
        path_to_initial_pdb: str | pathlib.Path,
        bias_constant_in_units: Float,
        n_steps: Int,
        restrain_atom_list: List[Int],
        parameters_for_md: dict,
        path_to_sim_checkpoint: Optional[str | pathlib.Path] = None,
    ):
        pdb = openmm_app.PDBFile(str(path_to_initial_pdb))
        self.modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
        self.bias_constant_in_units = bias_constant_in_units
        self.restrain_atom_list = restrain_atom_list

        self.n_steps = n_steps
        self.parameters_for_md = _validate_and_set_params_for_md(parameters_for_md)

        self.forcefield = _create_forcefield(self.parameters_for_md)
        self.platform = _create_platform(self.parameters_for_md)
        self.platform_properties = self.parameters_for_md["properties"]

        self.path_to_checkpoint = path_to_sim_checkpoint
        if not pathlib.Path(path_to_sim_checkpoint).exists():
            _generate_checkpoint(
                path_to_sim_checkpoint,
                self.modeller,
                self.forcefield,
                self.parameters_for_md,
                self.platform,
                self.platform_properties,
            )
            self.path_to_checkpoint = path_to_sim_checkpoint

    def __call__(self, positions_for_bias: Float[Array, "n_atoms 3"]):

        RMSD_value = openmm.RMSDForce(
            mdtraj.Trajectory(
                positions_for_bias / 10.0, self.modeller.topology
            ).openmm_positions(0),
            self.restrain_atom_list,
        )

        force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
        force_RMSD.addGlobalParameter("k", self.bias_constant_in_units)
        force_RMSD.addCollectiveVariable("RMSD", RMSD_value)

        system = _create_system(
            self.parameters_for_md, self.forcefield, self.modeller.topology
        )
        system.addForce(force_RMSD)

        integrator = _create_integrator(self.parameters_for_md)
        simulation = openmm_app.Simulation(
            self.modeller.topology,
            system,
            integrator,
            self.platform,
            self.parameters_for_md["properties"],
        )

        simulation.loadCheckpoint(self.path_to_checkpoint)

        simulation.step(self.n_steps)

        simulation.saveCheckpoint(self.path_to_checkpoint)
        positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )
        return jnp.array(positions) * 10.0  # Convert to angstroms


def _generate_checkpoint(
    path_to_output_checkpoint: str | pathlib.Path,
    modeller: openmm_app.Modeller,
    forcefield: openmm_app.ForceField,
    parameters_for_md: dict,
    platform: openmm.Platform,
    platform_properties: dict,
) -> None:
    
    integrator = _create_integrator(parameters_for_md)
    system = _create_system(parameters_for_md, forcefield, modeller.topology)
    simulation = openmm_app.Simulation(
        modeller.topology,
        system,
        integrator,
        platform,
        platform_properties,
    )

    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    simulation.step(1)
    simulation.saveCheckpoint(path_to_output_checkpoint)
    return


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
