import logging
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis import align
import numpy as np
import jax.numpy as jnp
import os
import h5py
import matplotlib.pyplot as plt

from .molecular_dynamics.mdaa_simulator import MDSimulatorRMSDConstraint

# from .molecular_dynamics.mdcg_simulator import MDCGSampler
from .optimization.optimizers import WeightOptimizer, PositionOptimizer
from .data._output_manager import OutputManager


def plot_loss(loss_values, output_path):
    # do a window averaging of the loss, window of 20

    indices = np.arange(len(loss_values))
    window_size = int(len(loss_values) * 0.1)

    losses_avg = [loss_values[0]]
    indices_avg = [indices[0]]
    for i in range(len(loss_values) - window_size):
        losses_avg.append(sum(loss_values[i : i + window_size]) / window_size)
        indices_avg.append(indices[i])

    losses_avg.append(loss_values[-1])
    indices_avg.append(indices[-1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(indices, loss_values, label="Loss", color="blue", alpha=0.7)
    ax.plot(indices_avg, losses_avg, label="Loss (smoothed)", color="red")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()

    fig_path = os.path.join(output_path, "loss_curve.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    return


def plot_weights(traj_wts, output_path):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for i in range(traj_wts.shape[1]):
        ax.plot(traj_wts[:, i], label=f"Weights Model {i+1}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Weights")

    ax.legend()
    fig_path = os.path.join(output_path, "weights_curve.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    return


def generate_outputs(universe, ref_universe, output_fname, atom_filter):
    """
    Generate outputs from the pipeline

    Parameters
    ----------
    universe : MDAnalysis.Universe
        Universe object with the optimized positions - should include the full system
    ref_universe : MDAnalysis.Universe
        Universe object with the reference structure
    output_fname : str
        Output file name
    atom_filter : str
        Atom selection string

    Returns
    -------
    None

    Generates PDB files for the trajectory of each optimized model. Plots the loss curve.
    """

    universe = universe.select_atoms(atom_filter)
    with h5py.File(output_fname, "r") as file:
        losses = file["losses"][:]
        traj_wts = file["trajs_weights"][:]
        n_frames, n_models, _, n_atoms = file["trajs_positions"].shape

        for i in range(n_models):
            traj_path = os.path.join(os.path.dirname(output_fname), f"traj_{i}.pdb")
            with mda.Writer(traj_path, n_atoms) as W:
                for j in range(n_frames):
                    universe.atoms.positions = file["trajs_positions"][j, i, :, :]
                    align.alignto(
                        universe, ref_universe, select=atom_filter, match_atoms=True
                    )
                    W.write(universe)

        plot_loss(losses, os.path.dirname(output_fname))
        plot_weights(traj_wts, os.path.dirname(output_fname))

    return


class Pipeline:
    def __init__(self, workflow, config):
        self.check_steps(workflow)
        self.workflow = workflow
        self.config = config
        # self.output_manager = output_manager

        return

    def check_steps(self, workflow):
        logging.info("Reading pipeline workflow...")

        self.workflow_types = []
        for i, step in enumerate(workflow):
            if isinstance(step, MDSimulatorRMSDConstraint):
                self.workflow_types.append("MDSampler")
                logging.info(f"Step {i}: MDSampler")

            # elif isinstance(step, MDCGSampler):
            #     self.workflow_types.append("MDCGSampler")
            #     logging.info(f"Step {i}: MDCGSampler")

            elif isinstance(step, WeightOptimizer):
                self.workflow_types.append("WeightOptimizer")
                logging.info(f"Step {i}: WeightOptimizer")

            elif isinstance(step, PositionOptimizer):
                self.workflow_types.append("PositionOptimizer")
                logging.info(f"Step {i}: PositionOptimizer")

            else:
                logging.error(
                    f"Invalid step {i}, must be MDSampler, WeightOptimizer, or PositionOptimizer"
                )
                raise ValueError(
                    f"Invalid step type: {type(step)}, must be MDSampler, WeightOptimizer, or PositionOptimizer"
                )
        logging.info(f"Loaded workflow with steps: {self.workflow_types}")
        logging.info("Checking if workflow is valid")
        self.workflow_is_valid()
        logging.info("Workflow check complete.")

        return

    def workflow_is_valid(self):
        if all(x in ["MDCGSampler", "MDSampler"] for x in self.workflow_types):
            logging.error("Workflow cannot contain MDCGSampler and MDSampler")
            raise ValueError("Workflow cannot contain MDCGSampler and MDSampler")

        if not any(x in ["MDCGSampler", "MDSampler"] for x in self.workflow_types):
            logging.error(
                "Workflow must contain at least one of MDCGSampler or MDSampler"
            )
            raise ValueError(
                "Workflow must contain at least one of MDCGSampler or MDSampler"
            )

        if "WeightOptimizer" not in self.workflow_types:
            logging.error("Workflow must contain at least one WeightOptimizer")
            raise NotImplementedError(
                "Workflow must contain at least one WeightOptimizer"
            )

        if "PositionOptimizer" not in self.workflow_types:
            logging.error("Workflow must contain at least one PositionOptimizer")
            raise NotImplementedError(
                "Workflow must contain at least one PositionOptimizer"
            )

        for i in range(len(self.workflow_types)):
            logging.info(f"Checking step {i} with type: {self.workflow_types[i]}")
            if i == 0 and self.workflow_types[i] != "WeightOptimizer":
                logging.error("First step must be WeightOptimizer")
                raise NotImplementedError("First step must be WeightOptimizer")

            if i == 1 and self.workflow_types[i] != "PositionOptimizer":
                logging.error("Second step must be PositionOptimizer")
                raise NotImplementedError("Second step must be PositionOptimizer")

            if i > 1 and self.workflow_types[i] not in ["MDSampler", "MDCGSampler"]:
                logging.error(f"Step {i} is {self.workflow_types[i]}")
                logging.error("All steps after second must be MDSampler or MDCGSampler")
                raise NotImplementedError(
                    "All steps after second must be MDSampler or MDCGSampler"
                )

        return

    def prepare_for_run_(
        self,
        config,
        init_universes,
        struct_info,
        ref_universe,
        init_weights=None,
    ):
        if "MDSampler" in self.workflow_types:
            if config["mode"] not in ["all-atom", "resid"]:
                logging.error(
                    "Invalid mode, must be 'all-atom' or 'resid' when using MDSampler"
                )
                raise ValueError(
                    "Invalid mode, must be 'all-atom' or 'resid' when using MDSampler"
                )

            self.filetype = "pdb"

        if "MDCGSampler" in self.workflow_types:
            if config["mode"] in ["all-atom", "resid"]:
                logging.error("Cannot run all-atom optimization with CG MD")
                raise ValueError("Cannot run all-atom optimization with CG MD")

            self.filetype = "gro"

        self.atom_list_filter = config["atom_list_filter"]
        self.univ_md = []
        self.univ_pull = []
        self.ref_universe = ref_universe
        self.n_models = len(init_universes)
        self.struct_info = struct_info

        for i in range(len(init_universes)):
            self.univ_md.append(init_universes[i].copy())
            self.univ_pull.append(init_universes[i].copy())

        self.n_steps = config["n_steps"]

        models_shape = (
            len(init_universes),
            *init_universes[0]
            .select_atoms(self.atom_list_filter)
            .atoms.positions.shape,
        )

        output_fname = os.path.join(
            config["output_path"], config["experiment_name"] + ".h5"
        )
        self.output_manager = OutputManager(output_fname, self.n_steps, models_shape)

        if init_weights is None:
            self.weights = jnp.ones(self.n_models) / self.n_models

        else:
            self.weights = init_weights.copy()

        self.opt_atom_list = (
            self.univ_md[0].select_atoms(self.atom_list_filter).atoms.indices
        )
        self.unit_cell = self.univ_md[0].atoms.dimensions

        return

    def run_md_(self, step):
        for i in range(self.n_models):
            self.univ_md[i].atoms.write(f"positions.{self.filetype}")
            self.univ_pull[i].atoms.write(f"ref_positions.{self.filetype}")

            # self.univ_md[i].atoms.write(f"md_init_model_{i}.pdb")
            # self.univ_pull[i].atoms.write(f"md_pull_model_{i}.pdb")
            positions = step.run(
                i, f"ref_positions.{self.filetype}", self.opt_atom_list
            )
            self.univ_md[i].atoms.positions = positions.copy()
            self.univ_md[i].select_atoms(self.atom_list_filter).atoms.write(
                f"positions_after_md_{i}.{self.filetype}"
            )

        return

    def run_wts_opt_(self, step):
        positions = np.zeros(
            (
                self.n_models,
                *self.ref_universe.select_atoms(
                    self.atom_list_filter
                ).atoms.positions.shape,
            )
        )

        for i in range(self.n_models):
            dummy_univ = self.univ_md[i].copy()
            align.alignto(
                dummy_univ,
                self.ref_universe,
                select=self.atom_list_filter,
                match_atoms=True,
            )
            positions[i] = dummy_univ.select_atoms(
                self.atom_list_filter
            ).atoms.positions

        positions = jnp.array(positions)
        self.weights = step.run(positions, self.weights, self.struct_info)

        return

    def run_pos_opt_(self, step):
        positions = np.zeros(
            (
                self.n_models,
                *self.ref_universe.select_atoms(
                    self.atom_list_filter
                ).atoms.positions.shape,
            )
        )

        for i in range(self.n_models):
            dummy_univ = self.univ_md[i].copy()

            # dummy_univ.atoms.write(f"model_before_opt_{i}.pdb")

            align.alignto(
                dummy_univ,
                self.ref_universe,
                select=self.atom_list_filter,
                match_atoms=True,
            )

            positions[i] = dummy_univ.select_atoms(
                self.atom_list_filter
            ).atoms.positions

        logging.debug(f"Optimized_positions: {positions}")

        positions = jnp.array(positions)
        positions, loss = step.run(positions, self.weights, self.struct_info)

        logging.debug(f"Optimized_positions: {positions}")

        for i in range(self.n_models):
            dummy_univ = self.univ_md[i].copy()
            align.alignto(
                dummy_univ,
                self.ref_universe,
                select=self.atom_list_filter,
                match_atoms=True,
            )
            dummy_univ.select_atoms(self.atom_list_filter).atoms.positions = positions[
                i
            ]

            align.alignto(
                dummy_univ,
                self.univ_md[i],
                select=self.atom_list_filter,
                match_atoms=True,
            )
            self.univ_pull[i].atoms.positions = dummy_univ.atoms.positions.copy()
            self.univ_pull[i].select_atoms(self.atom_list_filter).atoms.write(
                f"model_after_opt_{i}.pdb"
            )

        return loss

    def run(self):
        logging.info(f"Running pipeline for {self.n_steps} steps...")

        loss = None
        with tqdm(range(self.n_steps), unit="step") as pbar:
            for counter in pbar:
                for step in self.workflow:
                    if isinstance(
                        step, MDSimulatorRMSDConstraint
                    ):  # or isinstance(step, MDCGSampler):
                        self.run_md_(step)

                    elif isinstance(step, WeightOptimizer):
                        self.run_wts_opt_(step)

                    elif isinstance(step, PositionOptimizer):
                        loss = self.run_pos_opt_(step)

                    else:
                        continue

                pbar.set_postfix(loss=loss)
                self.output_manager.write(
                    np.array(
                        [
                            univ.select_atoms(self.atom_list_filter).atoms.positions
                            for univ in self.univ_md
                        ]
                    ),
                    self.weights,
                    loss,
                    counter,
                )

        logging.info("Pipeline finished.")
        logging.info("Saving last checkpoints to output path...")
        for i in range(self.n_models):
            os.system(
                f"cp checkpoint_model_{i}_tmp.chk {self.config['output_path']}/checkpoint_model_{i}.chk"
            )

            os.system(f"rm checkpoint_model_{i}_tmp.chk")
        logging.info(f"Output saved to {self.output_manager.file_name}.")
        self.output_manager.close()

        generate_outputs(
            self.univ_md[0],
            self.ref_universe,
            self.output_manager.file_name,
            self.atom_list_filter,
        )

        return
