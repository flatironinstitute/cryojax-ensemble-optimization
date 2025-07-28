import os
import pathlib
from typing import Any, Tuple
from typing_extensions import override

import jax
import jax.numpy as jnp
import mdtraj
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int, PRNGKeyArray
from mdtraj.formats import XYZTrajectoryFile
from tqdm import tqdm

from .._likelihood_optimization.optimizers import (
    IterativeEnsembleLikelihoodOptimizer,
    ProjGradDescWeightOptimizer,
)
from .._prior_projection import ParallelSteeredOverdampedLangevinSampler
from .base_pipeline import AbstractEnsembleOptimizationPipeline


class EnsembleOptimizationLangevinPipeline(
    AbstractEnsembleOptimizationPipeline, strict=True
):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projector: ParallelSteeredOverdampedLangevinSampler
    likelihood_optimizer: IterativeEnsembleLikelihoodOptimizer
    n_steps: int
    runs_postprocessing: bool

    def __init__(
        self,
        prior_projector: ParallelSteeredOverdampedLangevinSampler,
        likelihood_optimizer: IterativeEnsembleLikelihoodOptimizer,
        n_steps: int,
        *,
        runs_postprocessing: bool = True,
    ):
        self.prior_projector = prior_projector
        self.likelihood_optimizer = likelihood_optimizer
        self.n_steps = n_steps
        self.runs_postprocessing = runs_postprocessing

    @override
    def run(
        self,
        key: PRNGKeyArray,
        initial_walkers: Float[Array, "n_walkers n_atoms 3"],
        initial_weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        *,
        output_directory: str | pathlib.Path,
        initial_state_for_projector: Any = None,
    ) -> Tuple[
        Float[Array, "n_steps n_walkers n_atoms 3"],
        Float[Array, "n_steps n_walkers"],
    ]:

        # print("Initializing projetor...")
        if initial_state_for_projector is None:
            initial_state_for_projector = self.prior_projector.initialize(
                (key, initial_walkers)
            )

        proj_states = self.prior_projector.initialize(initial_state_for_projector)
        # print("Projector initialized.")

        walkers = initial_walkers.copy()
        weights = initial_weights.copy()

        if walkers.ndim == 2:
            walkers = jnp.expand_dims(walkers, axis=0)

        if weights.ndim == 0:
            weights = jnp.expand_dims(weights, axis=0)

        writers = [
            XYZTrajectoryFile(os.path.join(output_directory, f"traj_walker_{i}.xyz"), "w")
            for i in range(walkers.shape[0])
        ]

        for i in tqdm(range(self.n_steps)):
            """
            if stride_for_pose is True:
                new_dataset = pose_estimation(walkers)
                dataloader = create_dataloader...
            """

            walkers, weights = self.likelihood_optimizer(
                walkers,
                weights,
                dataloader,
            )

            walkers, proj_states = self.prior_projector(walkers, proj_states)

            for j in range(walkers.shape[0]):
                writers[j].write(walkers[j][None, ...] / 10.0)

        for writer in writers:
            writer.close()

        if self.runs_postprocessing:
            # print("Running postprocessing...")
            weight_optimizer = ProjGradDescWeightOptimizer(
                self.likelihood_optimizer.gaussian_amplitudes,
                self.likelihood_optimizer.gaussian_variances,
                self.likelihood_optimizer.image_to_walker_log_likelihood_fn,
            )
            walkers, weights = self.postprocess(
                walkers, weights, dataloader, weight_optimizer
            )
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
