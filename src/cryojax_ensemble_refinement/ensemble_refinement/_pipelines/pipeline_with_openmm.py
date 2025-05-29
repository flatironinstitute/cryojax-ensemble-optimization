import os
import pathlib
from typing import Any, Tuple
from typing_extensions import override
import time
import jax
import jax.numpy as jnp
import mdtraj
from jax_dataloader import DataLoader
from jaxtyping import Array, Float, Int, PRNGKeyArray
from tqdm import tqdm

from .._likelihood_optimization.optimizers import (
    IterativeEnsembleOptimizer,
    ProjGradDescWeightOptimizer,
)
from .._prior_projection.base_prior_projector import AbstractEnsemblePriorProjector
from .base_pipeline import AbstractEnsembleRefinementPipeline


# os.environ["XLA_FLAGS"] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1'


class EnsembleRefinementPipeline(AbstractEnsembleRefinementPipeline, strict=True):
    """
    Ensemble refinement pipeline using OpenMM for molecular dynamics simulation.
    """

    prior_projector: AbstractEnsemblePriorProjector
    likelihood_optimizer: IterativeEnsembleOptimizer
    n_steps: int
    reference_structure: mdtraj.Trajectory
    atom_indices_for_opt: Int[Array, " n_atoms_for_opt"]
    runs_postprocessing: bool

    def __init__(
        self,
        prior_projector: AbstractEnsemblePriorProjector,
        likelihood_optimizer: IterativeEnsembleOptimizer,
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
        print("Initializing projetor...")
        md_states = self.prior_projector.initialize(initial_state_for_projector)
        print("Projector initialized.")


        walkers = initial_walkers.copy()
        weights = initial_weights.copy()

        if walkers.ndim == 2:
            walkers = jnp.expand_dims(walkers, axis=0)

        if weights.ndim == 0:
            weights = jnp.expand_dims(weights, axis=0)

        print("Preparing writers for output...")
        writers = [
            mdtraj.formats.XTCTrajectoryFile(
                os.path.join(output_directory, f"traj_walker_{i}.xtc"), "w"
            )
            for i in range(walkers.shape[0])
        ]
        print("Writers prepared.")

        print("Aligning walkers to reference structure...")
        walkers = _align_walkers_to_reference(
            walkers, self.reference_structure, self.atom_indices_for_opt
        )
        print("Walkers aligned.")

        for i in tqdm(range(self.n_steps)):
            key, subkey = jax.random.split(key)

            print("Likelihood Optimization: ")
            tmp_walkers, weights = self.likelihood_optimizer(
                walkers[:, self.atom_indices_for_opt, :],
                weights,
                dataloader,
            )

            walkers = walkers.at[:, self.atom_indices_for_opt, :].set(tmp_walkers)
            walkers.block_until_ready()
            walkers = jax.device_get(walkers)
            print("Likelihood Optimization done.")

            # give some time for the cores to be freed up
            time.sleep(10.0)

            print("Prior Projection: ")
            walkers, md_states = self.prior_projector(subkey, walkers, md_states)

            walkers = _align_walkers_to_reference(
                walkers, self.reference_structure, self.atom_indices_for_opt
            )

            print("Write trajectory to files...")
            for j in range(walkers.shape[0]):
                writers[j].write(walkers[j] / 10.0)

        for writer in writers:
            writer.close()

        if self.runs_postprocessing:

            print("Running postprocessing...")
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


def _align_walkers_to_reference(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    reference_structure: mdtraj.Trajectory,
    atom_indices: Int[Array, " n_atoms_for_opt"],
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
