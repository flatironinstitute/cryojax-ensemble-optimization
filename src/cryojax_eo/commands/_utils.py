"""Helpers shared by the command-line entry points."""

import cryojax_eo as cxeo


def make_pose_search(
    estimates_pose: bool,
    pose_search_params: dict,
) -> cxeo.HierarchicalSO3GridSearch | None:
    """Build the pose search from its config, or `None` if it was not requested.

    **Arguments:**
    - `estimates_pose`:
        Whether poses should be estimated at all. If False, `pose_search_params`
        is ignored.
    - `pose_search_params`:
        A dictionary formatted by the `PoseSearchConfig` class.

    **Returns:**
        The `HierarchicalSO3GridSearch` described by the config, or `None` if
        `estimates_pose` is False.
    """
    if not estimates_pose:
        return None

    return cxeo.HierarchicalSO3GridSearch(
        base_grid_res=pose_search_params["initial_resolution"],
        n_rounds=pose_search_params["n_rounds"],
        n_candidates=pose_search_params["n_candidates"],
        n_angles_in_parallel=pose_search_params["n_angles_in_parallel"],
        shift_search_range_in_angstroms=pose_search_params[
            "shift_search_range_in_angstroms"
        ],
    )
