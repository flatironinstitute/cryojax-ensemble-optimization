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
from .likelihood_optimization.deprc_optimizers import EnsembleOptimizer
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

    universe = universe.select_atoms("protein")
    with h5py.File(output_fname, "r") as file:
        losses = file["losses"][1:]
        traj_wts = file["trajs_weights"][1:]
        n_frames, n_models, n_atoms, _ = file["trajs_positions"].shape

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
    def __init__(
        self,
        experiment_name: str,
        ensemble_optimizer: EnsembleOptimizer,
        md_sampler: MDSimulatorRMSDConstraint,
        init_models_path: list[str],
        ref_model_path: str,
        atom_list_filter: str = "protein and not name H*",
    ):
        self.experiment_name = experiment_name
        self.ensemble_optimizer = ensemble_optimizer
        self.md_sampler = md_sampler

        univ_init, self.ref_universe = _load_universes(init_models_path, ref_model_path)
        self.univ_md = [u.copy() for u in univ_init]
        self.univ_pull = [u.copy() for u in univ_init]
        self.atom_list_filter = atom_list_filter
        self.opt_atom_list = (
            self.univ_md[0].select_atoms(self.atom_list_filter).atoms.indices
        )

        # self.output_manager = output_manager

        return

    def prepare_for_run_(
        self,
        n_steps,
        output_path,
        init_weights=None,
    ):
        models_shape = (
            len(self.univ_md),
            *self.univ_md[0].select_atoms("protein").atoms.positions.shape,
        )

        output_fname = os.path.join(output_path, self.experiment_name + ".h5")
        output_manager = OutputManager(output_fname, n_steps + 1, models_shape)

        output_manager.write(
            np.array(
                [univ.select_atoms("protein").atoms.positions for univ in self.univ_md]
            ),
            self.ensemble_optimizer.weights,
            0.0,
            0,
        )

        return output_manager

    def run_md(self):
        for i in range(len(self.univ_md)):
            self.univ_md[i].atoms.write("tmp/positions.pdb")
            self.univ_pull[i].atoms.write("tmp/ref_positions.pdb")

            # self.univ_md[i].atoms.write(f"md_init_model_{i}.pdb")
            # self.univ_pull[i].atoms.write(f"md_pull_model_{i}.pdb")
            positions = self.md_sampler.run(
                i, "tmp/ref_positions.pdb", self.opt_atom_list
            )
            self.univ_md[i].atoms.positions = positions.copy()
            self.univ_md[i].select_atoms(self.atom_list_filter).atoms.write(
                f"tmp/positions_after_md_{i}.pdb"
            )

        return

    def run_ensemble_optimizer(self):
        positions = np.zeros(
            (
                len(self.univ_md),
                *self.ref_universe.select_atoms(
                    self.atom_list_filter
                ).atoms.positions.shape,
            )
        )

        for i in range(len(self.univ_md)):
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
        positions, weights, loss = self.ensemble_optimizer.run(positions)

        logging.debug(f"Optimized_positions: {positions}")

        for i in range(len(self.univ_md)):
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
                f"tmp/model_after_opt_{i}.pdb"
            )

        return loss, weights

    def run(self, n_steps, output_path):
        output_manager = self.prepare_for_run_(n_steps, output_path)

        logging.info(f"Running pipeline for {n_steps} steps...")

        loss = None
        with tqdm(range(n_steps), unit="step") as pbar:
            for counter in pbar:
                # run optimization
                loss, weights = self.run_ensemble_optimizer()

                # run MD
                self.run_md()

                pbar.set_postfix(loss=loss)
                output_manager.write(
                    np.array(
                        [
                            univ.select_atoms("protein").atoms.positions
                            for univ in self.univ_md
                        ]
                    ),
                    weights,
                    loss,
                    counter + 1,
                )

        logging.info("Pipeline finished.")
        logging.info("Saving last checkpoints and atomic structures to output path...")
        for i in range(len(self.univ_md)):
            os.system(
                f"cp checkpoint_model_{i}_tmp.chk {output_path}/checkpoint_model_{i}.chk"
            )

            self.univ_md[i].atoms.write(f"{output_path}/model_{i}.pdb")

            os.system(f"rm checkpoint_model_{i}_tmp.chk")
        logging.info(f"Output saved to {output_manager.file_name}.")
        output_manager.close()

        generate_outputs(
            self.univ_md[0],
            self.ref_universe,
            output_manager.file_name,
            self.atom_list_filter,
        )

        return


def _load_universes(init_models_path, ref_model_path):
    init_universes = []
    for i in range(len(init_models_path)):
        init_universes.append(mda.Universe(init_models_path[i]))

    ref_universe = mda.Universe(ref_model_path)
    return init_universes, ref_universe
