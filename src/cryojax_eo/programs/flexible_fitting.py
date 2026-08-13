import logging
import os
import pathlib
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import mdtraj
import numpy as np
import optax
from jaxtyping import Array, Float, Int
from mdtraj.formats import XTCTrajectoryFile
from tqdm import tqdm

from cryojax_eo.utils import EarlyStopping, ModelToVolumeAligner, rigid_align_positions

from .._prior_projection import (
    SteeredMDSimulator,
)
from .._volume_to_model_utils.model_to_volume_loss import ModelToVolumeCorrelationLossFn
from .._volume_to_model_utils.optimizer import AbstractWalkerOptimizer


class FlexibleFittingPipeline(eqx.Module):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projector: SteeredMDSimulator
    walker_optimizer: AbstractWalkerOptimizer
    n_steps: int
    prealigned_structure: mdtraj.Trajectory
    model_to_volume_aligner: ModelToVolumeAligner | None
    atom_indices_for_opt: Int[Array, " n_atoms_for_opt"]
    early_stopping: EarlyStopping | None
    write_buffer_size: int
    write_buffer_size: int

    def __init__(
        self,
        prior_projector: SteeredMDSimulator,
        walker_optimizer: AbstractWalkerOptimizer,
        n_steps: int,
        prealigned_structure: mdtraj.Trajectory,
        atom_indices_for_opt: Int[Array, " n_atoms_for_opt"],
        model_to_volume_aligner: ModelToVolumeAligner | None = None,
        early_stopping: EarlyStopping | None = None,
        *,
        write_buffer_size: int = 10,
    ):
        assert n_steps > 0, "n_steps must be positive"
        assert atom_indices_for_opt.ndim == 1, "atom_indices_for_opt must be a 1D array."
        assert len(atom_indices_for_opt) > 0, (
            "atom_indices_for_opt must contain at least one index."
        )
        assert write_buffer_size > 0, "write_buffer_size must be positive"

        self.prior_projector = prior_projector
        self.walker_optimizer = walker_optimizer
        self.n_steps = n_steps
        self.prealigned_structure = prealigned_structure
        self.atom_indices_for_opt = atom_indices_for_opt
        self.model_to_volume_aligner = model_to_volume_aligner
        self.early_stopping = early_stopping
        self.write_buffer_size = write_buffer_size
        self.write_buffer_size = write_buffer_size

    def run(
        self,
        initial_walker: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
        bias_constant_scheduler: optax.ScalarOrSchedule,
        *,
        output_directory: str | pathlib.Path,
        initial_state_for_projector: Any = None,
    ) -> tuple[Float[Array, "n_atoms 3"], Any]:
        logging.info("Initializing projector...")
        md_state = self.prior_projector.initialize(initial_state_for_projector)
        logging.info("Projector initialized.")

        unit_cell_vectors = np.array(
            self.prior_projector.simulation.topology.getPeriodicBoxVectors()._value
        )

        ref_positions = (
            jnp.asarray(self.prealigned_structure.xyz[0]) * 10.0
        )  # Convert from nm to Angstroms
        walker = initial_walker.copy()

        logging.info("Preparing writers for output...")
        writer = XTCTrajectoryFile(os.path.join(output_directory, "traj_walker.xtc"), "w")
        logging.info("Writers prepared.")

        logging.info("Aligning walker to reference structure...")
        walker = _align_walker_to_reference(
            walker, ref_positions, self.atom_indices_for_opt
        )
        logging.info("Walkers aligned.")

        # Construct initial optimizer state
        opt_state = self.walker_optimizer._initalize_opt_state(
            walker[self.atom_indices_for_opt, :]
        )

        early_stopping_state = (
            self.early_stopping.init() if self.early_stopping is not None else None
        )

        # Buffer trajectory frames on the host and flush them to the XTC writer in
        # batches, rather than writing (and forcing a device->host transfer) every
        # step. Shape: (buffer_size, n_atoms, 3), stored in nm.
        n_atoms = walker.shape[0]
        xtc_buffer = np.empty((self.write_buffer_size, n_atoms, 3), dtype=np.float32)
        buffer_count = 0

        # Reusable single-frame Trajectory for the per-step PDB snapshots: set the
        # topology and unit cell once, then only swap the coordinates each write
        # (avoids rebuilding a Trajectory and re-attaching the topology every step).
        pdb_snapshot_traj = mdtraj.Trajectory(
            xyz=np.zeros((1, n_atoms, 3), dtype=np.float32),
            topology=self.prealigned_structure.topology,
        )
        pdb_snapshot_traj.unitcell_vectors = unit_cell_vectors[None, ...]

        # Buffer trajectory frames on the host and flush them to the XTC writer in
        # batches, rather than writing (and forcing a device->host transfer) every
        # step. Shape: (buffer_size, n_atoms, 3), stored in nm.
        n_atoms = walker.shape[0]
        xtc_buffer = np.empty((self.write_buffer_size, n_atoms, 3), dtype=np.float32)
        buffer_count = 0

        # Reusable single-frame Trajectory for the per-step PDB snapshots: set the
        # topology and unit cell once, then only swap the coordinates each write
        # (avoids rebuilding a Trajectory and re-attaching the topology every step).
        pdb_snapshot_traj = mdtraj.Trajectory(
            xyz=np.zeros((1, n_atoms, 3), dtype=np.float32),
            topology=self.prealigned_structure.topology,
        )
        pdb_snapshot_traj.unitcell_vectors = unit_cell_vectors[None, ...]

        progress_bar = tqdm(range(self.n_steps), desc="Flexible Fitting", leave=True)
        for i in progress_bar:
            """
            if stride_for_pose is True:
                new_dataset = pose_estimation(walker)
                dataloader = create_dataloader...
            """

            logging.info("Likelihood Optimization: ")
            loss, tmp_walker, opt_state = self.walker_optimizer(
                walker[self.atom_indices_for_opt, :],
                reference_volume,
                opt_state,
            )

            ref_walker = walker.at[self.atom_indices_for_opt, :].set(tmp_walker)
            ref_walker.block_until_ready()
            logging.info("Likelihood Optimization done.")

            logging.info("Prior Projection: ")
            walker, md_state = self.prior_projector(
                ref_walker, md_state, bias_constant_scheduler(i)
            )

            logging.info("Aligning walker to reference structure...")
            walker = _align_walker_to_reference(
                walker, ref_positions, self.atom_indices_for_opt
            )
            logging.info("Walker aligned.")

            if self.model_to_volume_aligner is not None:
                logging.info("   Aligning walker to volume...")
                walker = _align_walkers_to_volume(
                    walker,
                    self.model_to_volume_aligner,
                    self.atom_indices_for_opt,
                    self.walker_optimizer.model_to_vol_loss_fn.amplitudes,
                    self.walker_optimizer.model_to_vol_loss_fn.variances,
                )
                logging.info("   Walkers aligned to volume.")

            logging.info("Buffering trajectory frame and writing snapshot...")
            # Single device->host transfer per step; divide once (Angstrom -> nm).
            walker_nm = np.asarray(walker, dtype=np.float32) / 10.0

            xtc_buffer[buffer_count] = walker_nm
            buffer_count += 1
            if buffer_count == self.write_buffer_size:
                logging.info("Flushing buffered trajectory frames to XTC writer...")
                writer.write(xtc_buffer[:buffer_count])
                buffer_count = 0

            # Overwrite the snapshot PDB, reusing one Trajectory object and the nm
            # coordinates already computed for the XTC buffer.
            _write_walker_to_pdb(
                pdb_snapshot_traj,
                walker_nm,
                os.path.join(output_directory, "curr_walker.pdb"),
            )

            loss_str = (
                f"C.C: {1 - loss:.4f}"
                if isinstance(
                    self.walker_optimizer.model_to_vol_loss_fn,
                    ModelToVolumeCorrelationLossFn,
                )
                else f"MSE: {loss:.4e}"
            )
            progress_bar.set_description(f"Flexible Fitting ({loss_str})")

            if self.early_stopping is not None and early_stopping_state is not None:
                early_stopping_state, should_stop = self.early_stopping.step(
                    early_stopping_state, loss
                )
                if should_stop:
                    logging.info("Early stopping triggered. Stopping optimization.")
                    break

        # Flush any frames still in the buffer before closing the writer.
        if buffer_count > 0:
            writer.write(xtc_buffer[:buffer_count])
            buffer_count = 0

        writer.close()

        _write_walker_to_pdb(
            pdb_snapshot_traj,
            np.asarray(walker, dtype=np.float32) / 10.0,
            os.path.join(output_directory, "final_walker.pdb"),
        )

        return walker, md_state


def _write_walker_to_pdb(snapshot_traj, positions_nm, filename):
    """Overwrite ``filename`` with a single-frame PDB of ``positions_nm`` (in nm).

    Reuses ``snapshot_traj``'s topology and unit cell by swapping only its
    coordinates, instead of constructing a new ``mdtraj.Trajectory`` (which
    re-attaches the topology) on every call.
    """
    snapshot_traj.xyz = positions_nm[None, ...]
    snapshot_traj.save_pdb(filename)
    return


@eqx.filter_jit
def _align_walker_to_reference(
    walker: Float[Array, "n_atoms 3"],
    ref_positions: Float[Array, "n_atoms 3"],
    atom_indices: Int[Array, " n_atoms_for_opt"],
) -> Float[Array, "n_atoms 3"]:
    """
    Align the walker to the reference structure.
    """

    _, rot_matrix, displacement = rigid_align_positions(
        walker[atom_indices], ref_positions[atom_indices]
    )
    aligned_walker = walker @ rot_matrix.T + displacement

    return jnp.asarray(aligned_walker)


def _align_walkers_to_volume(
    walker: Float[Array, "n_atoms 3"],
    model_to_volume_aligner: ModelToVolumeAligner,
    atom_indices: Int[Array, " n_atoms_for_opt"],
    amplitudes,
    variances,
) -> Float[Array, "n_atoms 3"]:
    """
    Align the walker to the volume using the ModelToVolumeAligner.
    """
    _, solution = model_to_volume_aligner.align(
        walker[atom_indices, :],
        amplitudes,
        variances,
    )

    return walker @ solution.rotation_matrix + solution.offset
