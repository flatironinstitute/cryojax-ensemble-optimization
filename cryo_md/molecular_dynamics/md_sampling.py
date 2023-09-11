"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""
import numpy as np
import openmm
import logging
import openmm.app as openmm_app
import openmm.unit as openmm_unit


class MDSampler:
    def __init__(
        self,
        pdb_file: str,
        restrain_force_constant: float,
        n_steps: int,
        **kwargs,
    ) -> None:
        self.pdb_file = pdb_file
        self.restrain_force_constant = restrain_force_constant
        self.n_steps = n_steps

        self.parse_kwargs(**kwargs)

        self.define_forcefield()
        self.define_platform()
        # self.define_integrator()

    def parse_kwargs(self, **kwargs):
        default_kwargs = {
            "forcefield": "amber14-all.xml",
            "water_model": "amber14/tip3p.xml",
            "nonbondedMethod": openmm_app.PME,
            "nonbondedCutoff": 1.0 * openmm_unit.nanometers,
            "constraints": openmm_app.HBonds,
            "temperature": 300.0 * openmm_unit.kelvin,
            "friction": 1.0 / openmm_unit.picosecond,
            "timestep": 0.002 * openmm_unit.picoseconds,
            "platform": "CPU",
            "properties": {"Threads": "1"},
        }

        for key in kwargs:
            if key not in default_kwargs:
                raise ValueError(f"Invalid argument {key}")

        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        self.config = kwargs

        return

    def define_forcefield(self):
        self.forcefield = openmm_app.ForceField(
            self.config["forcefield"], self.config["water_model"]
        )
        return

    def define_platform(self):
        self.platform = openmm.Platform.getPlatformByName(self.config["platform"])
        return

    def update_system(self, process_id, ref_position_file, restrain_atom_list):
        integrator = openmm.LangevinIntegrator(
            self.config["temperature"], self.config["friction"], self.config["timestep"]
        )

        pdb = openmm_app.PDBFile(self.pdb_file)
        pdb_ref = openmm_app.PDBFile(ref_position_file)
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)

        system = self.forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=self.config["nonbondedMethod"],
            nonbondedCutoff=self.config["nonbondedCutoff"],
            constraints=self.config["constraints"],
        )

        RMSD_value = openmm.RMSDForce(
            pdb_ref.positions, restrain_atom_list
        )

        force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
        force_RMSD.addGlobalParameter("k", self.restrain_force_constant)
        force_RMSD.addCollectiveVariable("RMSD", RMSD_value)

        system.addForce(force_RMSD)

        simulation = openmm_app.Simulation(
            modeller.topology,
            system,
            integrator,
            self.platform,
            self.config["properties"],
        )

        simulation.loadCheckpoint(f"sim_model_{process_id}.chk")
        
        return simulation

    def run(self, process_id, ref_positions_file, restrain_atom_list):
        logging.info("Running MD simulation...")

        simulation = self.update_system(process_id, ref_positions_file, restrain_atom_list)
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
