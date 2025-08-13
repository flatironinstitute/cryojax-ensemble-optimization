import os
import pathlib
from typing import Any, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import mdtraj
from jaxtyping import Array, Float, Int
from mdtraj.formats import XTCTrajectoryFile
from tqdm import tqdm

from ...ensemble_optimization._prior_projection._molecular_dynamics.openmm import (
    SteeredMDSimulator,
)
from .._cross_corelation.optimizer import SteepestDescWalkerFlexibleFitting


class FlexibleFittingPipeline(eqx.Module):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projector: SteeredMDSimulator
    likelihood_optimizer: SteepestDescWalkerFlexibleFitting
    n_steps: int
    reference_structure: mdtraj.Trajectory
    atom_indices_for_opt: Int[Array, " n_atoms_for_opt"]
    runs_postprocessing: bool

    def __init__(
        self,
        prior_projector: SteeredMDSimulator,
        likelihood_optimizer: SteepestDescWalkerFlexibleFitting,
        n_steps: int,
        ref_structure_for_alignment: mdtraj.Trajectory,
        atom_indices_for_opt: Int[Array, " n_atoms_for_opt"],
        *,
        runs_postprocessing: bool = True,
    ):
        self.prior_projector = prior_projector
        self.likelihood_optimizer = likelihood_optimizer
        self.n_steps = n_steps
        self.reference_structure = ref_structure_for_alignment
        self.atom_indices_for_opt = atom_indices_for_opt
        self.runs_postprocessing = runs_postprocessing

    def run(
        self,
        initial_walker: Float[Array, "n_atoms 3"],
        reference_volume: Float[Array, "n_pixels n_pixels n_pixels"],
        *,
        output_directory: str | pathlib.Path,
        initial_state_for_projector: Any = None,
    ) -> Tuple[Float[Array, "n_atoms 3"], Any]:
        # print("Initializing projetor...")
        md_state = self.prior_projector.initialize(initial_state_for_projector)
        # print("Projector initialized.")

        walker = initial_walker.copy()

        # print("Preparing writers for output...")
        writer = XTCTrajectoryFile(os.path.join(output_directory, "traj_walker.xtc"), "w")
        # print("Writers prepared.")

        # print("Aligning walker to reference structure...")
        walker = _align_walkers_to_reference(
            walker, self.reference_structure, self.atom_indices_for_opt
        )
        # print("Walkers aligned.")

        progress_bar = tqdm(range(self.n_steps), desc="Flexible Fitting", leave=True)
        for i in progress_bar:
            """
            if stride_for_pose is True:
                new_dataset = pose_estimation(walker)
                dataloader = create_dataloader...
            """

            # print("Likelihood Optimization: ")
            loss, tmp_walker = self.likelihood_optimizer(
                walker[self.atom_indices_for_opt, :],
                reference_volume,
            )

            walker = walker.at[self.atom_indices_for_opt, :].set(tmp_walker)
            walker.block_until_ready()
            walker = jax.device_get(walker)
            # print("Likelihood Optimization done.")

            # print("Prior Projection: ")
            walker, md_state = self.prior_projector(walker, md_state)

            walker = _align_walkers_to_reference(
                walker, self.reference_structure, self.atom_indices_for_opt
            )

            # print("Write trajectory to files...")
            writer.write(walker / 10.0)

            progress_bar.set_description(f"Flexible Fitting (C.C: {1 - loss:.4f})")

        writer.close()

        return walker, md_state


def _align_walkers_to_reference(
    walker: Float[Array, "n_atoms 3"],
    reference_structure: mdtraj.Trajectory,
    atom_indices: Int[Array, " n_atoms_for_opt"],
) -> Float[Array, "n_atoms 3"]:
    """
    Align the walker to the reference structure.
    """
    # Convert walker to mdtraj format
    walkers_mdtraj = mdtraj.Trajectory(
        xyz=walker / 10.0,  # Convert to nm
        topology=reference_structure.topology,
    ).superpose(
        reference_structure,
        frame=0,
        atom_indices=atom_indices,
    )
    return jnp.array(walkers_mdtraj.xyz[0]) * 10.0  # Convert back to Angstroms
