import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.dataset import ParticleStackInfo
from jaxtyping import Array, Float

from .._likelihood_optimization._loss_functions.common_functions import (
    compute_optimal_scale_and_offset,
)
from .geometry import (
    getbestneighbors_base_SO3,
    getbestneighbors_next_SO3,
    grid_SO3,
)


@eqx.filter_vmap(in_axes=(0, None, None))
def loss_for_grid_search(
    quat: Float[Array, "4"],
    volume: cxs.AbstractVolumeRepresentation,
    relion_stack: ParticleStackInfo,
) -> Float:
    pose = cxs.QuaternionPose(
        offset_x_in_angstroms=0.0, offset_y_in_angstroms=0.0, wxyz=quat
    )

    computed_image = cxs.make_image_model(
        volume,
        relion_stack["parameters"]["image_config"],
        pose,
        relion_stack["parameters"]["transfer_theory"],
        simulates_quantity=False,
        normalizes_signal=True,
    ).simulate()

    observed_image = jnp.asarray(relion_stack["images"])

    scale, offset = compute_optimal_scale_and_offset(
        computed_image,
        observed_image,
    )

    return jnp.sum((scale * computed_image + offset - observed_image) ** 2)


@eqx.filter_jit
def global_SO3_hier_search(
    volume: cxs.AbstractVolumeRepresentation,
    relion_stack: ParticleStackInfo,
    base_grid: int,
    n_rounds: int,
    N_candidates: int,
) -> cxs.QuaternionPose:
    """
    Perform a global search on the SO3 grid using a hierarchical approach.

    Args:
        lossfn: A function that computes the loss for a given set of quaternions,
        return a numpy array
        base_grid: The base resolution of the SO3 grid. 1 -> 30, 2 -> 15
        n_rounds: The number of rounds to perform the search.
        N_candidates: The number of candidate quaternions to consider in each round.

    Returns:
        best_quats: The best quaternions found during the search.
        best_loss: The loss associated with the best quaternions.
    """

    def body_fun(i, val):
        allnb_quats, allnb_s2s1 = val

        loss = loss_for_grid_search(allnb_quats, volume, relion_stack)
        allnb_quats, allnb_s2s1 = getbestneighbors_next_SO3(
            loss, allnb_quats, allnb_s2s1, curr_res=base_grid + i, N=N_candidates
        )
        return (allnb_quats, allnb_s2s1)

    # Initialize the base SO3 grid
    base_quats = grid_SO3(base_grid)

    # Compute the initial loss for the base grid
    loss = loss_for_grid_search(base_quats, volume, relion_stack)  # numpy array

    if n_rounds <= 0:
        # If no rounds are to be performed, return the base quaternions and their loss
        best_index = jnp.argmin(loss)
        return cxs.QuaternionPose(wxyz=base_quats[best_index])

    else:
        allnb_quats, allnb_s2s1 = getbestneighbors_base_SO3(
            loss, base_quats, N=N_candidates, base_resol=base_grid
        )

        # Just in case n_rounds = 1
        allnb_quats, allnb_s2s1 = jax.lax.cond(
            n_rounds > 1,
            lambda _: jax.lax.fori_loop(1, n_rounds, body_fun, (allnb_quats, allnb_s2s1)),
            lambda _: (allnb_quats, allnb_s2s1),
            None,
        )

        loss = loss_for_grid_search(allnb_quats, volume, relion_stack)

        # Find the best quaternion and its associated loss
        best_index = jnp.argmin(loss)
        best_quats = allnb_quats[best_index]

        return cxs.QuaternionPose(wxyz=best_quats)


def local_SO3_hier_search(lossfn, base_grid=1, n_rounds=5, N_candidates=40):
    raise NotImplementedError(
        "Local SO3 hierarchical search is not implemented yet. "
        "Please use global SO3 hierarchical search instead."
    )
