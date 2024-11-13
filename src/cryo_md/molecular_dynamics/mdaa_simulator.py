"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""

import logging
import os
from typing import List, Optional
import numpy as np
import openmm
import openmm.app as openmm_app
import openmm.unit as openmm_unit


class MDSimulatorRMSDConstraint:
    def __init__(
        self,
        models_fname: List[str],
        restrain_force_constant: float,
        n_steps: int,
        n_models: int,
        checkpoint_fnames: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        if checkpoint_fnames is None:
            self.checkpoint_fnames = [""] * n_models

        else:
            self.checkpoint_fnames = checkpoint_fnames

        self.models_fname = models_fname
        self.restrain_force_constant = restrain_force_constant
        self.n_steps = n_steps
        self.n_models = n_models

        self.parse_kwargs(**kwargs)
        self.define_forcefield()
        self.define_platform()

        for i in range(self.n_models):
            if self.checkpoint_fnames[i] == "":
                logging.info(f"Generating checkpoint for model {i}")

                self.generate_checkpoint(
                    self.models_fname[i], f"checkpoint_model_{i}_tmp.chk"
                )
                self.checkpoint_fnames[i] = f"checkpoint_model_{i}_tmp.chk"
                logging.info(
                    f"Checkpoint for model {i} generated and saved as {self.checkpoint_fnames[i]}."
                )

            else:
                logging.info(
                    f"Checkpoint for model {i} found at {self.checkpoint_fnames[i]}."
                )
                logging.info("Creating copy...")
                os.system(
                    f"cp {self.checkpoint_fnames[i]} checkpoint_model_{i}_tmp.chk"
                )
                self.checkpoint_fnames[i] = f"checkpoint_model_{i}_tmp.chk"

    def parse_kwargs(self, **kwargs):
        default_kwargs = {
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

        units = {
            "nonbondedCutoff": 1.0 * openmm_unit.nanometers,
            "temperature": 1.0 * openmm_unit.kelvin,
            "friction": 1.0 / openmm_unit.picosecond,
            "timestep": 1.0 * openmm_unit.picoseconds,
        }

        for key in kwargs:
            if key not in default_kwargs:
                raise ValueError(f"Invalid argument {key}")
            if key in units:
                kwargs[key] = kwargs[key] * units[key]

        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        self.md_params = kwargs

        return

    def generate_checkpoint(self, pdb_fname, fname):
        integrator = openmm.LangevinIntegrator(
            self.md_params["temperature"],
            self.md_params["friction"],
            self.md_params["timestep"],
        )

        pdb = openmm_app.PDBFile(pdb_fname)
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)

        system = self.forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=self.md_params["nonbondedMethod"],
            nonbondedCutoff=self.md_params["nonbondedCutoff"],
            constraints=self.md_params["constraints"],
        )

        simulation = openmm_app.Simulation(
            modeller.topology,
            system,
            integrator,
            self.platform,
            self.md_params["properties"],
        )

        simulation.context.setPositions(modeller.positions)
        simulation.minimizeEnergy()
        simulation.step(1)
        simulation.saveCheckpoint(fname)

        return

    def define_forcefield(self):
        self.forcefield = openmm_app.ForceField(
            self.md_params["forcefield"], self.md_params["water_model"]
        )
        return

    def define_platform(self):
        self.platform = openmm.Platform.getPlatformByName(self.md_params["platform"])
        return

    def update_system(self, process_id, ref_position_file, restrain_atom_list):
        integrator = openmm.LangevinIntegrator(
            self.md_params["temperature"],
            self.md_params["friction"],
            self.md_params["timestep"],
        )

        pdb = openmm_app.PDBFile(self.models_fname[process_id])
        pdb_ref = openmm_app.PDBFile(ref_position_file)
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)

        system = self.forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=self.md_params["nonbondedMethod"],
            nonbondedCutoff=self.md_params["nonbondedCutoff"],
            constraints=self.md_params["constraints"],
        )

        RMSD_value = openmm.RMSDForce(pdb_ref.positions, restrain_atom_list)

        force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
        force_RMSD.addGlobalParameter("k", self.restrain_force_constant)
        force_RMSD.addCollectiveVariable("RMSD", RMSD_value)

        system.addForce(force_RMSD)

        simulation = openmm_app.Simulation(
            modeller.topology,
            system,
            integrator,
            self.platform,
            self.md_params["properties"],
        )

        simulation.loadCheckpoint(self.checkpoint_fnames[process_id])

        return simulation

    def run(self, process_id, ref_positions_file, restrain_atom_list):
        logging.info("Running MD simulation...")

        simulation = self.update_system(
            process_id, ref_positions_file, restrain_atom_list
        )
        logging.info("  Positions updated.")

        # tolerance = 100 * openmm_unit.kilojoules_per_mole / openmm_unit.nanometer
        # simulation.minimizeEnergy(tolerance)
        # simulation.minimizeEnergy(maxIterations=self.n_steps)
        # logging.info("  Energy minimized.")

        logging.info(f"  Running simulation for {self.n_steps} steps...")
        simulation.step(self.n_steps)

        simulation.saveCheckpoint(self.checkpoint_fnames[process_id])

        positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )

        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        logging.info(f"Simulation complete. Final Energy: {energy}")

        return np.array(positions) * 10.0
