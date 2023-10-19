"""
Functions for running MD simulations using OpenMM.

Functions
---------
run_md_openmm
    Run MD simulations using OpenMM
"""
import numpy as np
import logging
import openmm
import openmm.unit as openmm_unit
import openmm.app as openmm_app
import martini_openmm as martini


class MDCGSampler:
    def __init__(
        self,
        gro_file: str,
        top_file: str,
        restrain_force_constant: float,
        n_steps: int,
        epsilon_r: float = 15.0,
        **kwargs,
    ) -> None:
        self.gro_file = gro_file
        self.top_file = top_file
        self.restrain_force_constant = restrain_force_constant
        self.n_steps = n_steps
        self.epsilon_r = epsilon_r

        self.parse_kwargs(**kwargs)

        self.define_platform()
        # self.define_integrator()

    def parse_kwargs(self, **kwargs):
        default_kwargs = {
            "nonbondedCutoff": 1.1 * openmm_unit.nanometers,
            "temperature": 310.0 * openmm_unit.kelvin,
            "friction": 10.0 / openmm_unit.picosecond,
            "timestep": 20 * openmm_unit.femtosecond,
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


    def define_platform(self):
        self.platform = openmm.Platform.getPlatformByName(self.config["platform"])
        return

    def update_system(self, process_id, ref_position_file, restrain_atom_list):
        
        conf = openmm_app.GromacsGroFile(self.gro_file)
        conf_ref = openmm_app.GromacsGroFile(ref_position_file)

        box_vectors = conf.getPeriodicBoxVectors()
        top = martini.MartiniTopFile(
            self.top_file,
            periodicBoxVectors=box_vectors,
            defines={},
            epsilon_r=self.epsilon_r
        )

        system = top.create_system(nonbonded_cutoff=self.config["nonbondedCutoff"])

        integrator = openmm.LangevinIntegrator(
            self.config["temperature"], self.config["friction"], self.config["timestep"]
        )

        RMSD_value = openmm.RMSDForce(
            conf_ref.positions, restrain_atom_list
        )

        force_RMSD = openmm.CustomCVForce("0.5 * k * RMSD^2")
        force_RMSD.addGlobalParameter("k", self.restrain_force_constant)
        force_RMSD.addCollectiveVariable("RMSD", RMSD_value)

        system.addForce(force_RMSD)

        simulation = openmm_app.Simulation(
            top.topology,
            system,
            integrator,
            self.platform,
            self.config["properties"]
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
