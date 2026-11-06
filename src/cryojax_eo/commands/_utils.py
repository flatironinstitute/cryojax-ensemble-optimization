"""Helpers shared by the command-line entry points."""

from pathlib import Path

import cryojax.simulator as cxs
import jax.numpy as jnp
import mrcfile
from jaxtyping import Array, Float

import cryojax_eo as cxeo
from cryojax_eo.simulator import DilatedMask


def make_pose_search(
    estimates_poses: bool,
    pose_search_params: dict,
    *,
    dilated_mask: DilatedMask | None = None,
    mask_frame_positions: Float[Array, "n_atoms 3"] | None = None,
) -> cxeo.HierarchicalSO3GridSearch | None:
    """Build the pose search from its config, or `None` if it was not requested.

    **Arguments:**
    - `estimates_poses`:
        Whether poses should be estimated at all. If False, `pose_search_params`
        is ignored.
    - `pose_search_params`:
        A dictionary formatted by the `PoseSearchConfig` class.
    - `dilated_mask`:
        Optional mask used to score the candidate orientations.
    - `mask_frame_positions`:
        Optional positions of a model aligned to `dilated_mask`, to which the
        walkers are aligned before each pose search.

    **Returns:**
        The `HierarchicalSO3GridSearch` described by the config, or `None` if
        `estimates_poses` is False.
    """
    if not estimates_poses:
        return None

    return cxeo.HierarchicalSO3GridSearch(
        base_grid_res=pose_search_params["initial_resolution"],
        n_rounds=pose_search_params["n_rounds"],
        n_candidates=pose_search_params["n_candidates"],
        n_angles_in_parallel=pose_search_params["n_angles_in_parallel"],
        shift_search_range_in_angstroms=pose_search_params[
            "shift_search_range_in_angstroms"
        ],
        dilated_mask=dilated_mask,
        mask_frame_positions=mask_frame_positions,
    )


def read_pose_search_mask(
    path_to_mask: str | Path, image_config: cxs.BasicImageConfig
) -> Float[Array, "z y x"]:
    """Read the 3D mask used by the pose search, checking that it is on the
    same box as the particle images.
    """
    with mrcfile.open(path_to_mask, mode="r") as mrc:
        mask = jnp.asarray(mrc.data).copy()

    box_size = image_config.shape[0]
    if mask.shape != (box_size, box_size, box_size):
        raise ValueError(
            f"The pose search mask {path_to_mask} has shape {mask.shape}, but the "
            f"particle images need a mask of shape {(box_size,) * 3}."
        )
    return mask
