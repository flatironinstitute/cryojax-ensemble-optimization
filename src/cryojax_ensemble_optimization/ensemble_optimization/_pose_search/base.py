import numpy as np

from .geometry import (
    getbestneighbors_base_SO3,
    getbestneighbors_next_SO3,
    grid_SO3,
)


def global_SO3_hier_search(lossfn, base_grid=1, n_rounds=5, N_candidates=40):
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
    # Initialize the base SO3 grid
    base_quats = grid_SO3(base_grid)

    # Compute the initial loss for the base grid
    loss = lossfn(base_quats)  # numpy array

    # Iterate through the specified number of rounds
    # if n_rounds == 1, skip the whole for loop
    assert (
        n_rounds >= 1
    ), "n_rounds must be greater or equal than 1 for hierarchical search"
    for i in range(n_rounds - 1):
        if i == 0:
            # Get the best neighbors from the base SO3 grid, minimize the loss
            allnb_quats, allnb_s2s1 = getbestneighbors_base_SO3(
                loss, base_quats, N=N_candidates, base_resol=base_grid
            )
        else:
            # Get the best neighbors from the current SO3 grid
            allnb_quats, allnb_s2s1 = getbestneighbors_next_SO3(
                loss, allnb_quats, allnb_s2s1, curr_res=base_grid + i, N=N_candidates
            )

        # Compute the loss for the neighbors
        loss = lossfn(allnb_quats)

    # Find the best quaternion and its associated loss
    best_index = np.argmin(loss)
    best_quats = allnb_quats[best_index]
    best_loss = loss[best_index]

    return best_quats, best_loss


def local_SO3_hier_search(lossfn, base_grid=1, n_rounds=5, N_candidates=40):
    raise NotImplementedError(
        "Local SO3 hierarchical search is not implemented yet. "
        "Please use global SO3 hierarchical search instead."
    )
