"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""

"""
import numpy as np
import logging
import openmm
import openmm.unit as openmm_unit
import openmm.app as openmm_app
import martini_openmm as martini
from typing import Any
import os


class MDCGSampler:
    def __init__(
        self,
        models_fname: str,
        top_file: str,
        restrain_force_constant: float,
        n_steps: int,
        n_models: int,
        checkpoint_fnames: Any,
        epsilon_r: float,
        **kwargs,
    ) -> None:
        if checkpoint_fnames is None:
            checkpoint_fnames = [None] * n_models

        else:
            self.checkpoint_fnames = checkpoint_fnames

        self.models_fname = models_fname
        self.top_file = top_file
        self.restrain_force_constant = restrain_force_constant
        self.n_steps = n_steps
        self.epsilon_r = epsilon_r
        self.n_models = n_models

        self.parse_kwargs(**kwargs)

        self.define_platform()

        for i in range(self.n_models):
            if self.checkpoint_fnames[i] is None:
                logging.info(f"Generating checkpoint for model {i}")

                self.generate_checkpoint(
                    models_fname[i], f"checkpoint_model_{i}_tmp.chk"
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
            "nonbondedCutoff": 1.1 * openmm_unit.nanometers,
            "temperature": 310.0 * openmm_unit.kelvin,
            "friction": 10.0 / openmm_unit.picosecond,
            "timestep": 20 * openmm_unit.femtosecond,
            "platform": "CPU",
            "properties": {"Threads": "1"},
        }

        units = {
            "nonbondedCutoff": 1.0 * openmm_unit.nanometers,
            "temperature": 1.0 * openmm_unit.kelvin,
            "friction": 1.0 / openmm_unit.picosecond,
            "timestep": 1.0 * openmm_unit.femtosecond,
        }

        for key in kwargs:
            if key not in default_kwargs:
                raise ValueError(f"Invalid argument {key}")

            if key in units:
                kwargs[key] = kwargs[key] * units[key]

        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        self.config = kwargs

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

    def define_platform(self):
        self.platform = openmm.Platform.getPlatformByName(self.config["platform"])
        return

    def update_system(self, process_id, ref_position_file, restrain_atom_list):
        conf = openmm_app.GromacsGroFile(self.models_fname[process_id])
        conf_ref = openmm_app.GromacsGroFile(ref_position_file)

        box_vectors = conf.getPeriodicBoxVectors()
        top = martini.MartiniTopFile(
            self.top_file,
            periodicBoxVectors=box_vectors,
            defines={},
            epsilon_r=self.epsilon_r,
        )

        system = top.create_system(nonbonded_cutoff=self.config["nonbondedCutoff"])

        integrator = openmm.LangevinIntegrator(
            self.config["temperature"], self.config["friction"], self.config["timestep"]
        )

        RMSD_value = openmm.RMSDForce(conf_ref.positions, restrain_atom_list)

        force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
        force_RMSD.addGlobalParameter("k", self.restrain_force_constant)
        force_RMSD.addCollectiveVariable("RMSD", RMSD_value)

        system.addForce(force_RMSD)

        simulation = openmm_app.Simulation(
            top.topology, system, integrator, self.platform, self.config["properties"]
        )

        simulation.loadCheckpoint(f"sim_model_{process_id}.chk")

        return simulation

    def run(self, process_id, ref_positions_file, restrain_atom_list):
        logging.info("Running MD simulation...")

        simulation = self.update_system(
            process_id, ref_positions_file, restrain_atom_list
        )
        logging.info("  Positions updated.")

        simulation.minimizeEnergy()
        logging.info("  Energy minimized.")

        logging.info(f"  Running simulation for {self.n_steps} steps...")
        simulation.step(self.n_steps)

        simulation.saveState(f"sim_model_{process_id}.state")
        simulation.saveCheckpoint(f"sim_model_{process_id}.chk")

        positions = simulation.context.getState(getPositions=True).getPositions(
            asNumpy=True
        )

        energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        logging.info(f"Simulation complete. Final Energy: {energy}")

        return np.array(positions) * 10.0
"""
