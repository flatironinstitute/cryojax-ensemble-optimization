import os
from typing import Any, List, Tuple

import jax.numpy as jnp
import mdtraj
from equinox import Module
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int
from pydantic import DirectoryPath
from tqdm import tqdm

from .._likelihood_optimization.optimizers import (
    IterativeEnsembleOptimizer,
    ProjGradDescWeightOptimizer,
)
from .._prior_projection._molecular_dynamics.openmm import (
    EnsembleSteeredMDSimulator,
    SteeredMDSimulator,
)


# class EnsembleRefinementOpenMMPipeline(AbstractEnsembleRefinementPipeline):
class EnsembleRefinementOpenMMPipeline(Module):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projector: EnsembleSteeredMDSimulator | SteeredMDSimulator
    likelihood_optimizer: IterativeEnsembleOptimizer
    n_steps: int
    reference_structure: mdtraj.Trajectory
    atom_indices_for_opt: List[Int]
    runs_postprocessing: bool

    def __init__(
        self,
        prior_projector: EnsembleSteeredMDSimulator | SteeredMDSimulator,
        likelihood_optimizer: IterativeEnsembleOptimizer,
        n_steps: int,
        ref_structure_for_opt: mdtraj.Trajectory,
        atom_indices_for_opt: List[Int],
        *,
        runs_postprocessing: bool = True,
    ):
        self.prior_projector = prior_projector
        self.likelihood_optimizer = likelihood_optimizer
        self.n_steps = n_steps
        self.reference_structure = ref_structure_for_opt
        self.atom_indices_for_opt = atom_indices_for_opt
        self.runs_postprocessing = runs_postprocessing

    def __call__(
        self,
        initial_walkers: Float[Array, "n_walkers n_atoms 3"],
        initial_weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        args_for_likelihood_optimizer: Any,
        *,
        output_directory: DirectoryPath,
    ) -> Tuple[
        Float[Array, "n_steps n_walkers n_atoms 3"],
        Float[Array, "n_steps n_walkers"],
    ]:
        md_states = self.prior_projector.initialize()
        walkers = initial_walkers.copy()
        weights = initial_weights.copy()

        if walkers.ndim == 2:
            walkers = jnp.expand_dims(walkers, axis=0)

        if weights.ndim == 0:
            weights = jnp.expand_dims(weights, axis=0)

        writers = [
            mdtraj.formats.XTCTrajectoryFile(
                os.path.join(output_directory, f"traj_walker_{i}.xtc"), "w"
            )
            for i in range(walkers.shape[0])
        ]

        walkers = _align_walkers_to_reference(
            walkers, self.reference_structure, self.atom_indices_for_opt
        )

        for i in tqdm(range(self.n_steps)):
            tmp_walkers, weights = self.likelihood_optimizer(
                walkers[:, self.atom_indices_for_opt, :],
                weights,
                dataloader,
                args_for_likelihood_optimizer,
            )

            walkers = walkers.at[:, self.atom_indices_for_opt, :].set(tmp_walkers)

            walkers, md_states = self.prior_projector(walkers, md_states)

            walkers = _align_walkers_to_reference(
                walkers, self.reference_structure, self.atom_indices_for_opt
            )
            for j in range(walkers.shape[0]):
                writers[j].write(walkers[j] / 10.0)

        for writer in writers:
            writer.close()

        if self.runs_postprocessing:
            walkers, weights = self.postprocess(
                walkers, weights, dataloader, args_for_likelihood_optimizer
            )
        return walkers, weights

    def postprocess(
        self,
        walkers: Float[Array, "n_walkers n_atoms 3"],
        weights: Float[Array, " n_walkers"],
        dataloader: DataLoader,
        args_for_likelihood_optimizer: Any,
    ):
        """
        Postprocess the walkers and weights.
        """
        # Project the weights
        weights = ProjGradDescWeightOptimizer()(
            walkers[:, self.atom_indices_for_opt],
            weights,
            dataloader,
            args_for_likelihood_optimizer,
        )

        return walkers, weights


def _align_walkers_to_reference(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    reference_structure: mdtraj.Trajectory,
    atom_indices: List[Int],
) -> Float[Array, "n_walkers n_atoms 3"]:
    """
    Align the walkers to the reference structure.
    """
    # Convert walkers to mdtraj format
    walkers_mdtraj = mdtraj.Trajectory(
        xyz=walkers / 10.0,  # Convert to nm
        topology=reference_structure.topology,
    ).superpose(
        reference_structure,
        frame=0,
        atom_indices=atom_indices,
    )
    return jnp.array(walkers_mdtraj.xyz) * 10.0  # Convert back to Angstroms
