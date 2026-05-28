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

from cryojax_eo.ensemble_optimization import (
    SteeredMDSimulator,
)
from cryojax_eo.utils import ModelToVolumeAligner, rigid_align_positions

from .._cross_corelation.optimizer import SteepestDescWalkerFlexibleFitting


class FlexibleFittingPipeline(eqx.Module):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projector: SteeredMDSimulator
    walker_optimizer: SteepestDescWalkerFlexibleFitting
    n_steps: int
    prealigned_structure: mdtraj.Trajectory
    model_to_volume_aligner: ModelToVolumeAligner | None
    atom_indices_for_opt: Int[Array, " n_atoms_for_opt"]

    def __init__(
        self,
        prior_projector: SteeredMDSimulator,
        walker_optimizer: SteepestDescWalkerFlexibleFitting,
        n_steps: int,
        prealigned_structure: mdtraj.Trajectory,
        atom_indices_for_opt: Int[Array, " n_atoms_for_opt"],
        model_to_volume_aligner: ModelToVolumeAligner | None = None,
    ):
        assert n_steps > 0, "n_steps must be positive"
        assert atom_indices_for_opt.ndim == 1, "atom_indices_for_opt must be a 1D array."
        assert len(atom_indices_for_opt) > 0, (
            "atom_indices_for_opt must contain at least one index."
        )

        self.prior_projector = prior_projector
        self.walker_optimizer = walker_optimizer
        self.n_steps = n_steps
        self.prealigned_structure = prealigned_structure
        self.atom_indices_for_opt = atom_indices_for_opt
        self.model_to_volume_aligner = model_to_volume_aligner

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

        progress_bar = tqdm(range(self.n_steps), desc="Flexible Fitting", leave=True)
        for i in progress_bar:
            """
            if stride_for_pose is True:
                new_dataset = pose_estimation(walker)
                dataloader = create_dataloader...
            """

            logging.info("Likelihood Optimization: ")
            loss, tmp_walker = self.walker_optimizer(
                walker[self.atom_indices_for_opt, :],
                reference_volume,
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

            logging.info("Write trajectory to files...")
            writer.write(walker / 10.0)
            _write_walker_to_pdb(
                walker,
                os.path.join(output_directory, "curr_walker.pdb"),
                self.prealigned_structure.topology,
                unit_cell_vectors,
            )

            progress_bar.set_description(f"Flexible Fitting (C.C: {1 - loss:.4f})")

        writer.close()
        _write_walker_to_pdb(
            walker,
            os.path.join(output_directory, "final_walker.pdb"),
            self.prealigned_structure.topology,
            unit_cell_vectors,
        )

        return walker, md_state


def _write_walker_to_pdb(walker, filename, topology, unit_cell_vectors):
    walker_as_traj = mdtraj.Trajectory(
        xyz=walker / 10.0,
        topology=topology,
    )
    walker_as_traj.unitcell_vectors = unit_cell_vectors[None, ...]
    walker_as_traj.save_pdb(filename)
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
