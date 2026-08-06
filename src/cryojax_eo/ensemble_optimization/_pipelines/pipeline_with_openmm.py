import logging
import os
import pathlib
from typing import Any
from typing_extensions import override

import jax
import jax.numpy as jnp
import mdtraj
import numpy as np
import optax
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int, PRNGKeyArray
from mdtraj.formats import XTCTrajectoryFile
from tqdm import tqdm

from cryojax_eo.utils import ModelToVolumeAligner, rigid_align_positions

from .._likelihood_optimization import (
    IterativeEnsembleLikelihoodOptimizer,
    MultGradWeightOptimizer,
)
from .._prior_projection.base_prior_projector import AbstractEnsemblePriorProjector
from .base_pipeline import AbstractEnsembleOptimizationPipeline


class EnsembleOptimizationPipeline(AbstractEnsembleOptimizationPipeline, strict=True):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projector: AbstractEnsemblePriorProjector
    likelihood_optimizer: IterativeEnsembleLikelihoodOptimizer
    n_steps: int
    prealigned_structure: mdtraj.Trajectory
    model_to_volume_aligner: ModelToVolumeAligner | None
    atom_indices_for_opt: Int[Array, " n_atoms_for_opt"]
    runs_postprocessing: bool
    write_buffer_size: int

    def __init__(
        self,
        prior_projector: AbstractEnsemblePriorProjector,
        likelihood_optimizer: IterativeEnsembleLikelihoodOptimizer,
        n_steps: int,
        prealigned_structure: mdtraj.Trajectory,
        atom_indices_for_opt: Int[Array, " n_atoms_for_opt"],
        model_to_volume_aligner: ModelToVolumeAligner | None = None,
        *,
        runs_postprocessing: bool = True,
        write_buffer_size: int = 10,
    ):
        self.prior_projector = prior_projector
        self.likelihood_optimizer = likelihood_optimizer
        self.n_steps = n_steps
        self.prealigned_structure = prealigned_structure
        self.model_to_volume_aligner = model_to_volume_aligner
        self.atom_indices_for_opt = atom_indices_for_opt
        self.runs_postprocessing = runs_postprocessing
        self.write_buffer_size = write_buffer_size

    @override
    def run(
        self,
        key: PRNGKeyArray,
        initial_walkers: Float[Array, "n_walkers n_atoms 3"],
        initial_weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        bias_constant_scheduler: optax.Schedule,
        *,
        output_directory: str | pathlib.Path,
        initial_state_for_projector: Any = None,
    ) -> tuple[
        Float[Array, "n_steps n_walkers n_atoms 3"],
        Float[Array, "n_steps n_walkers"],
    ]:
        logging.info("Initializing projector...")
        md_states = self.prior_projector.initialize(initial_state_for_projector)
        logging.info("Projector initialized.")

        # extract unit cell vectors from simulation
        unit_cell_vectors = np.array(
            self.prior_projector.projectors[0]
            .simulation.topology.getPeriodicBoxVectors()
            ._value
        )

        ref_positions = (
            np.asarray(self.prealigned_structure.xyz[0]) * 10.0
        )  # Convert from nm to Angstroms

        walkers = initial_walkers.copy()
        weights = initial_weights.copy()

        if walkers.ndim == 2:
            walkers = jnp.expand_dims(walkers, axis=0)

        if weights.ndim == 0:
            weights = jnp.expand_dims(weights, axis=0)

        logging.info("Preparing writers for output...")
        writers = [
            XTCTrajectoryFile(os.path.join(output_directory, f"traj_walker_{i}.xtc"), "w")
            for i in range(walkers.shape[0])
        ]
        logging.info("Writers prepared.")

        logging.info("Aligning walkers to reference structure...")
        walkers = _align_walkers_to_reference(
            walkers, ref_positions, self.atom_indices_for_opt
        )
        logging.info("Walkers aligned.")

        # Buffer trajectory frames on the host and flush them to the XTC writers in
        # batches, rather than writing (and forcing a device->host transfer) every
        # step. Shape: (buffer_size, n_walkers, n_atoms, 3), stored in nm.
        n_walkers, n_atoms = walkers.shape[0], walkers.shape[1]
        xtc_buffer = np.empty(
            (self.write_buffer_size, n_walkers, n_atoms, 3), dtype=np.float32
        )
        buffer_count = 0

        # Reusable single-frame Trajectory for the per-step PDB snapshots: set the
        # topology and unit cell once, then only swap the coordinates each write
        # (avoids rebuilding a Trajectory and re-attaching the topology every step).
        pdb_snapshot_traj = mdtraj.Trajectory(
            xyz=np.zeros((1, n_atoms, 3), dtype=np.float32),
            topology=self.prealigned_structure.topology,
        )
        pdb_snapshot_traj.unitcell_vectors = unit_cell_vectors[None, ...]

        progress_bar = tqdm(range(self.n_steps), desc="Optimization Progress")
        # make the tqdm progress bar show the current neg_log_likelihood at each step
        for i in progress_bar:
            logging.info(f"Starting optimization step {i + 1}/{self.n_steps}...")

            logging.info("   Likelihood Optimization: ")
            neg_log_likelihood, tmp_walkers, weights = self.likelihood_optimizer(
                walkers[:, self.atom_indices_for_opt, :],
                weights,
                dataloader,
            )
            logging.info(f"   Negative Log-Likelihood: {neg_log_likelihood}")
            progress_bar.set_description(
                f"Iter {i + 1}/{self.n_steps}, "
                f"Neg Log-Likelihood: {neg_log_likelihood:.4f}"
            )

            walkers = walkers.at[:, self.atom_indices_for_opt, :].set(tmp_walkers)
            walkers.block_until_ready()
            walkers = jax.device_get(walkers)
            logging.info("   Likelihood Optimization done.")

            logging.info("   Prior Projection: ")
            walkers, md_states = self.prior_projector(
                walkers, md_states, bias_constant_scheduler(i)
            )
            logging.info("   Prior Projection done.")

            logging.info("   Aligning walkers to reference structure...")
            walkers = _align_walkers_to_reference(
                walkers, ref_positions, self.atom_indices_for_opt
            )
            logging.info("   Walkers aligned.")

            if self.model_to_volume_aligner is not None:
                logging.info("   Aligning walkers to volume...")
                walkers = _align_walkers_to_volume(
                    walkers,
                    self.model_to_volume_aligner,
                    self.atom_indices_for_opt,
                    self.likelihood_optimizer.ensemble_likelihood_fn.image_to_walker_likelihood_fn.amplitudes,
                    self.likelihood_optimizer.ensemble_likelihood_fn.image_to_walker_likelihood_fn.variances,
                )
                logging.info("   Walkers aligned to volume.")

            logging.info("   Buffering trajectory frame and writing snapshots...")
            # Single device->host transfer per step; divide once (Angstrom -> nm).
            walkers_nm = np.asarray(walkers, dtype=np.float32) / 10.0

            xtc_buffer[buffer_count] = walkers_nm
            buffer_count += 1
            if buffer_count == self.write_buffer_size:
                logging.info("   Flushing buffered trajectory frames to XTC writers...")
                _flush_xtc_buffer(writers, xtc_buffer, buffer_count)
                buffer_count = 0

            # Overwrite the per-walker snapshot PDBs, reusing one Trajectory object
            # and the nm coordinates already computed for the XTC buffer.
            for j in range(n_walkers):
                _write_walker_to_pdb(
                    pdb_snapshot_traj,
                    walkers_nm[j],
                    os.path.join(output_directory, f"curr_walker_{j}.pdb"),
                )

        logging.info("Optimization complete.")

        # Flush any frames still in the buffer before closing the writers.
        if buffer_count > 0:
            _flush_xtc_buffer(writers, xtc_buffer, buffer_count)
            buffer_count = 0

        for writer in writers:
            writer.close()

        for i, walker in enumerate(walkers):
            _write_walker_to_pdb(
                pdb_snapshot_traj,
                np.asarray(walker, dtype=np.float32) / 10.0,
                os.path.join(output_directory, f"final_walker_{i}.pdb"),
            )

        if self.runs_postprocessing:
            logging.info("Running postprocessing...")
            weight_optimizer = MultGradWeightOptimizer(
                ensemble_likelihood_fn=self.likelihood_optimizer.ensemble_likelihood_fn,
                pose_search=self.likelihood_optimizer.pose_search,
            )
            walkers, weights = self.postprocess(
                walkers, weights, dataloader, weight_optimizer
            )
            logging.info("Postprocessing complete.")
        return walkers, weights

    def postprocess(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        weight_optimizer: MultGradWeightOptimizer,
    ):
        """
        Postprocess the walkers and weights.
        """
        # Project the weights
        weights = weight_optimizer(
            walkers[:, self.atom_indices_for_opt],
            dataloader,
        )

        return walkers, weights


def _flush_xtc_buffer(writers, buffer, count):
    """Write the first ``count`` buffered frames to each per-walker XTC writer.

    ``buffer`` has shape ``(buffer_size, n_walkers, n_atoms, 3)`` in nm; each writer
    receives its ``count`` frames in a single batched ``write`` call.
    """
    for j, writer in enumerate(writers):
        writer.write(buffer[:count, j])
    return


def _write_walker_to_pdb(snapshot_traj, positions_nm, filename):
    """Overwrite ``filename`` with a single-frame PDB of ``positions_nm`` (in nm).

    Reuses ``snapshot_traj``'s topology and unit cell by swapping only its
    coordinates, instead of constructing a new ``mdtraj.Trajectory`` (which
    re-attaches the topology) on every call.
    """
    snapshot_traj.xyz = positions_nm[None, ...]
    snapshot_traj.save_pdb(filename)
    return


def _align_walkers_to_reference(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    ref_positions: Float[Array, "n_atoms 3"],
    atom_indices: Int[Array, " n_atoms_for_opt"],
) -> Float[Array, "n_walkers n_atoms 3"]:
    """
    Align the walkers to the reference structure.
    """

    aligned_walkers = np.zeros_like(walkers)
    for i in range(walkers.shape[0]):
        _, rot_matrix, displacement = rigid_align_positions(
            walkers[i, atom_indices], ref_positions[atom_indices]
        )
        aligned_walkers[i] = walkers[i] @ rot_matrix.T + displacement

    return jnp.asarray(aligned_walkers)


def _align_walkers_to_volume(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    model_to_volume_aligner: ModelToVolumeAligner,
    atom_indices: Int[Array, " n_atoms_for_opt"],
    amplitudes,
    variances,
) -> Float[Array, "n_walkers n_atoms 3"]:
    """
    Align the walkers to the volume using the ModelToVolumeAligner.
    """
    for i in range(walkers.shape[0]):
        atomic_positions = walkers[i, atom_indices, :]

        _, solution = model_to_volume_aligner.align(
            atomic_positions,
            amplitudes,
            variances,
        )
        aligned_positions = walkers[i] @ solution.rotation_matrix + solution.offset

        walkers = walkers.at[i].set(aligned_positions)
    return walkers
