import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis import align
import h5py

from cryo_md.likelihood.calc_lklhood import (
    calc_lkl_and_grad_wts,
    calc_lkl_and_grad_struct,
)

from cryo_md.molecular_dynamics.md_utils import get_MD_outputs, dump_optimized_models
from cryo_md.molecular_dynamics.md_sampling import run_md_openmm


def optimize_weights(models, weights, steps, step_size, image_stack):
    losses = np.zeros(steps)
    for i in range(steps):
        loss, grad_wts = calc_lkl_and_grad_wts(
            models,
            weights,
            image_stack.images,
            image_stack.constant_params[0],
            image_stack.constant_params[1],
            image_stack.constant_params[2],
            image_stack.variable_params,
        )

        weights = weights + step_size * grad_wts
        weights /= jnp.sum(weights)

        losses[i] = loss

    return weights, losses


def run_optimizer(
    directory_path: str,
    ref_universe: mda.Universe,
    image_stack,
    filter: str,
    n_models: int,
    unit_cell: np.ndarray,
    n_steps,
    step_size,
    batch_size=None,
    init_weights=None,
):
    
    outputs = h5py.File(f"{directory_path}/outputs.h5", "w")

    trajs_full = outputs.create_dataset("trajs_full", (n_steps, n_models, *ref_universe.atoms.positions.T.shape), dtype=np.float64)
    trajs_wts = outputs.create_dataset("trajs_wts", (n_steps, n_models), dtype=np.float64)
    losses = outputs.create_dataset("losses", (n_steps, ), dtype=np.float64)

    losses_np = np.zeros(n_steps)


    if init_weights is None:
        opt_weights = jnp.ones(n_models) / n_models

    else:
        opt_weights = init_weights.copy()

    if batch_size is None:
        batch_size = image_stack.images.shape[0]

    opt_models = np.zeros((n_models, *ref_universe.select_atoms(filter).atoms.positions.T.shape))

    for i in range(n_models):
        univ_system = mda.Universe(
            f"{directory_path}/curr_system_{i}.pdb",
        )

        align.alignto(univ_system.select_atoms("protein"), ref_universe, select=filter, match_atoms=True)
        opt_models[i] = univ_system.select_atoms(filter).atoms.positions.T

    with tqdm(range(n_steps), unit="step") as pbar:
        for counter in pbar:

            for i in range(n_models):
                opt_univ = mda.Universe(f"{directory_path}/curr_{i}.pdb")
                align.alignto(opt_univ, ref_universe, select="not name H*", match_atoms=True)
                trajs_full[counter] = opt_univ.atoms.positions.T
                
            trajs_wts[counter] = opt_weights.copy()

            opt_weights, _ = optimize_weights(opt_models, opt_weights, 10, 0.1, image_stack)

            random_batch = np.arange(0, image_stack.images.shape[0], 1)
            np.random.shuffle(random_batch)
            random_batch = random_batch[:batch_size]

            loss, grad_str = calc_lkl_and_grad_struct(
                opt_models,
                opt_weights,
                image_stack.images[random_batch],
                image_stack.constant_params[0],
                image_stack.constant_params[1],
                image_stack.constant_params[2],
                image_stack.variable_params[random_batch],
            )

            pbar.set_postfix(loss=loss)
            losses[counter] = loss

            if loss is jnp.nan:
                print("Loss is nan. Exiting.")
                break
            
            norms = jnp.max(jnp.abs(grad_str), axis=(1))[:, None, :]
            grad_str /= jnp.maximum(norms, jnp.ones_like(norms))

            opt_models = opt_models + step_size * grad_str

            dump_optimized_models(directory_path=directory_path, opt_models=opt_models, ref_universe=ref_universe, unit_cell=unit_cell, filter=filter)

            run_md_openmm(n_models, atom_indices, directory_path, nsteps_md, stride_md, restrain_force_constant)

            opt_models = get_MD_outputs(n_models, directory_path, ref_universe, filter=filter, )
        

        losses[i] = loss

        grad_str /= jnp.max(jnp.abs(grad_str), axis=(1))[:, None, :]


        traj[i] = opt_models.copy()
        traj_wts[i] = opt_weights.copy()

    return traj, traj_wts, losses
