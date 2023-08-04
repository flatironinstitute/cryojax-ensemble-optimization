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
from cryo_md.utils.parser import pdb_parser


def optimize_weights_(models, weights, steps, step_size, image_stack):
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

def parse_initial_models_(directory_path: str, n_models: int, ref_universe: mda.Universe, filter: str):

    
    if filter == "name CA":
        struct_info = pdb_parser(f"{directory_path}/init_system_0.pdb", mode="resid")
        
    elif filter == "not name H*":
        struct_info = pdb_parser(f"{directory_path}/init_system_0.pdb", mode="all_atom")

    else:
        raise NotImplementedError(
            "Only CA and all-atom models are supported at the moment."
        )

    opt_models = np.zeros(
        (n_models, *ref_universe.select_atoms(filter).atoms.positions.T.shape)
    )
        
    for i in range(n_models):
        univ_system = mda.Universe(
            f"{directory_path}/init_system_{i}.pdb",
        )

        univ_prot = univ_system.select_atoms("protein")
        align.alignto(univ_prot, ref_universe, select=filter, match_atoms=True)
        opt_models[i] = univ_prot.select_atoms(filter).atoms.positions.T

    atom_indices = (
        mda.Universe(f"{directory_path}/init_system_0.pdb")
        .select_atoms(f"protein and {filter}")
        .atoms.indices
    )

    unit_cell = mda.Universe(f"{directory_path}/init_system_0.pdb").atoms.dimensions

    return opt_models, struct_info, atom_indices, unit_cell

def run_optimizer(
    n_models: int,
    ref_universe: mda.Universe,
    image_stack,
    filter: str,
    n_steps,
    step_size,
    nsteps_md,
    stride_md,
    restrain_force_constant,
    directory_path: str,
    batch_size=None,
    init_weights=None,
):
    
    assert n_models > 0, "Number of models must be greater than 0."
    assert n_steps > 0, "Number of steps must be greater than 0."
    assert step_size > 0, "Step size must be greater than 0."

    opt_models, struct_info, atom_indices, unit_cell = parse_initial_models_(directory_path, n_models, ref_universe, filter)

    if init_weights is None:
        opt_weights = jnp.ones(n_models) / n_models

    else:
        opt_weights = init_weights.copy()

    if batch_size is None:
        batch_size = image_stack.images.shape[0]

    else:
        assert (
            batch_size <= image_stack.n_images
        ), "Batch size must be smaller than the number of images in the stack."

        assert batch_size > 0, "Batch size must be greater than 0."

    outputs = h5py.File(f"{directory_path}/outputs.h5", "w")
    trajs_full = outputs.create_dataset(
        "trajs_full",
        (n_steps, n_models, *ref_universe.atoms.positions.T.shape),
        dtype=np.float64,
    )
    trajs_wts = outputs.create_dataset(
        "trajs_wts", (n_steps, n_models), dtype=np.float64
    )
    losses = outputs.create_dataset("losses", (n_steps,), dtype=np.float64)

    losses_np = np.zeros(n_steps)

    with tqdm(range(n_steps), unit="step") as pbar:
        for counter in pbar:

            opt_weights, _ = optimize_weights_(
                opt_models, opt_weights, 10, 0.1, image_stack
            )

            random_batch = np.random.choice(image_stack.n_images, batch_size, replace=False)

            loss, grad_str = calc_lkl_and_grad_struct(
                opt_models,
                opt_weights,
                image_stack.images[random_batch],
                struct_info,
                image_stack.constant_params[0],
                image_stack.constant_params[1],
                image_stack.constant_params[2],
                image_stack.variable_params[random_batch],
            )

            pbar.set_postfix(loss=loss)


            if loss is jnp.nan:
                print("Loss is nan. Exiting.")
                break

            norms = jnp.max(jnp.abs(grad_str), axis=(1))[:, None, :]
            grad_str /= jnp.maximum(norms, jnp.ones_like(norms))

            opt_models = opt_models + step_size * grad_str

            dump_optimized_models(
                directory_path=directory_path,
                opt_models=opt_models,
                ref_universe=ref_universe,
                unit_cell=unit_cell,
                filter=filter,
            )

            run_md_openmm(
                n_models,
                atom_indices,
                directory_path,
                nsteps_md,
                stride_md,
                restrain_force_constant,
            )

            opt_models = get_MD_outputs(
                n_models,
                directory_path,
                ref_universe,
                filter=filter,
            )

        for i in range(n_models):
            opt_univ = mda.Universe(f"{directory_path}/curr_sytem_{i}.pdb")
            align.alignto(
                opt_univ, ref_universe, select="not name H*", match_atoms=True
            )
            trajs_full[counter] = opt_univ.atoms.positions.T

        losses[counter] = loss
        losses_np[counter] = loss
        trajs_wts[counter] = opt_weights.copy()

        np.save("losses.npy", losses_np[:counter])

    return
