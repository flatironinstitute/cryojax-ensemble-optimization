import os
from typing import Any, List, Tuple
from typing_extensions import override

import mdtraj
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int
from pydantic import DirectoryPath

from ..likelihood_optimization.optimizers import (
    IterativeEnsembleOptimizer,
    ProjGradDescWeightOptimizer,
)
from ..prior_projection._molecular_dynamics._openmm import (
    SteeredMolecularDynamicsSimulator,
)
from .base_pipeline import AbstractEnsembleRefinementPipeline


class EnsembleRefinementOpenMMPipeline(AbstractEnsembleRefinementPipeline):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projectors: List[SteeredMolecularDynamicsSimulator]
    likelihood_optimizer: IterativeEnsembleOptimizer
    n_steps: int
    reference_structure: mdtraj.Trajectory
    atom_indices_for_opt: List[Int]
    runs_postprocessing: bool

    def __init__(
        self,
        prior_projectors: List[SteeredMolecularDynamicsSimulator],
        likelihood_optimizer: IterativeEnsembleOptimizer,
        n_steps: int,
        ref_structure_for_opt: mdtraj.Trajectory,
        atom_indices_for_opt: List[Int],
        *,
        runs_postprocessing: bool = True,
    ):
        self.prior_projectors = prior_projectors
        self.likelihood_optimizer = likelihood_optimizer
        self.n_steps = n_steps
        self.ref_structure_for_opt = ref_structure_for_opt
        self.atom_indices_for_opt = atom_indices_for_opt
        self.runs_postprocessing = runs_postprocessing

    @override
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
        walkers = initial_walkers.copy()
        weights = initial_weights.copy()
        writers = [
            mdtraj.formats.XTCTrajectoryFile(
                os.path.join(output_directory, f"traj_walker_{i}.xtc"), "w"
            )
            for i in range(walkers.shape[0])
        ]

        walkers = _align_walkers_to_reference(
            walkers, self.ref_structure_for_opt, self.atom_indices_for_opt
        )

        for i in range(self.n_steps):

            tmp_walkers, weights = self.likelihood_optimizer(
                walkers[:, self.atom_indices_for_opt, :],
                weights,
                dataloader,
                args_for_likelihood_optimizer,
            )

            walkers = walkers.at[:, self.atom_indices_for_opt, :].set(tmp_walkers)

            for i in range(len(self.prior_projectors)):
                walkers = walkers.at[i].set(self.prior_projectors[i](walkers[i]))

            walkers = _align_walkers_to_reference(
                walkers, self.ref_structure_for_opt, self.atom_indices_for_opt
            )
            for j in range(walkers.shape[0]):
                writers[j].write(walkers[j])

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
            walkers, weights, dataloader, args_for_likelihood_optimizer
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
    return walkers_mdtraj.xyz * 10.0  # Convert back to Angstroms
