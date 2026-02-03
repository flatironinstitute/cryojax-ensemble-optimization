import logging
import os
import pathlib
from typing import Any
from typing_extensions import override

import jax
import jax.numpy as jnp
import mdtraj
import optax
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int, PRNGKeyArray
from mdtraj.formats import XTCTrajectoryFile
from tqdm import tqdm

from ...utils import ModelToVolumeAligner
from .._likelihood_optimization.optimizers import (
    IterativeEnsembleLikelihoodOptimizer,
    ProjGradDescWeightOptimizer,
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
    ):
        self.prior_projector = prior_projector
        self.likelihood_optimizer = likelihood_optimizer
        self.n_steps = n_steps
        self.prealigned_structure = prealigned_structure
        self.model_to_volume_aligner = model_to_volume_aligner
        self.atom_indices_for_opt = atom_indices_for_opt
        self.runs_postprocessing = runs_postprocessing

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

        reference_structure = self.prealigned_structure

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
            walkers, reference_structure, self.atom_indices_for_opt
        )
        logging.info("Walkers aligned.")
        for i in tqdm(range(self.n_steps)):
            logging.info(f"Starting optimization step {i+1}/{self.n_steps}...")

            logging.info("   Likelihood Optimization: ")
            tmp_walkers, weights = self.likelihood_optimizer(
                walkers[:, self.atom_indices_for_opt, :],
                weights,
                dataloader,
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
                walkers, reference_structure, self.atom_indices_for_opt
            )
            logging.info("   Walkers aligned.")

            if self.model_to_volume_aligner is not None:
                logging.info("   Aligning walkers to volume...")
                walkers = _align_walkers_to_volume(
                    walkers,
                    self.model_to_volume_aligner,
                    self.atom_indices_for_opt,
                    self.likelihood_optimizer.likelihood_fn.amplitudes,
                    self.likelihood_optimizer.likelihood_fn.variances,
                )
                logging.info("   Walkers aligned to volume.")

            logging.info("   Writing trajectory to files...")
            for j in range(walkers.shape[0]):
                writers[j].write(walkers[j] / 10.0)

        logging.info("Optimization complete.")

        for writer in writers:
            writer.close()

        for i, walker in enumerate(walkers):
            mdtraj.Trajectory(
                xyz=walker / 10.0,
                topology=reference_structure.topology,
            ).save_pdb(os.path.join(output_directory, f"final_walker_{i}.pdb"))

        if self.runs_postprocessing:
            logging.info("Running postprocessing...")
            weight_optimizer = ProjGradDescWeightOptimizer(
                n_steps=500,
                likelihood_fn=self.likelihood_optimizer.likelihood_fn,
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
        weight_optimizer: ProjGradDescWeightOptimizer,
    ):
        """
        Postprocess the walkers and weights.
        """
        # Project the weights
        weights = weight_optimizer(
            walkers[:, self.atom_indices_for_opt],
            weights,
            dataloader,
        )

        return walkers, weights


def _align_walkers_to_reference(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    reference_structure: mdtraj.Trajectory,
    atom_indices: Int[Array, " n_atoms_for_opt"],
) -> Float[Array, "n_walkers n_atoms 3"]:
    """
    Align the walkers to the reference structure.
    """

    new_walkers = jnp.asarray(walkers.copy())
    for i in range(walkers.shape[0]):
        walker_mdtraj = mdtraj.Trajectory(
            xyz=walkers[i] / 10.0,  # Convert to nm
            topology=reference_structure.topology,
        )
        walker_mdtraj = walker_mdtraj.superpose(
            reference_structure,
            frame=0,
            atom_indices=atom_indices,
        )
        new_walkers = new_walkers.at[i].set(walker_mdtraj.xyz[0] * 10.0)
    return new_walkers


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
            amplitudes[i],
            variances[i],
        )
        aligned_positions = walkers[i] @ solution.rotation_matrix + solution.offset

        walkers = walkers.at[i].set(aligned_positions)
    return walkers
