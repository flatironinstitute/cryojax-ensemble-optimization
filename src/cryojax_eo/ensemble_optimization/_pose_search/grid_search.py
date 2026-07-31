from typing import Any

import cryojax.ndimage as cxim
import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.jax_util import filter_bmap
from jaxtyping import Array, Float

from .geometry import (
    getbestneighbors_base_SO3,
    getbestneighbors_next_SO3,
    grid_SO3,
)
from .pose_offset import compute_correlation_at_optimal_offset


@eqx.filter_vmap(in_axes=(0, None, None, None, None, None))
def _loss_for_grid_search(
    quat: Float[Array, "4"],
    volume: cxs.AbstractVolumeRepresentation,
    target_image: Float[Array, "H W"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    shift_search_area: Float[Array, "H W"] | None,
) -> tuple[Float[Array, ""], Float[Array, " 2"]]:
    """
    Computes the correlation at the optimal shift using Plancherel's theorem and
    Fourier's convolution theorem for a given quaternion.

    The function outputs the negative correlation as this is a loss function.
    """
    pose = cxs.QuaternionPose(
        offset_x_in_angstroms=0.0,
        offset_y_in_angstroms=0.0,
        wxyz=quat,
    )

    computed_image_no_shif_fft = cxs.make_image_model(
        volume,
        image_config,
        pose,
        transfer_theory,
        normalizes_signal=True,
    ).simulate(outputs_real_space=True)

    # get the optimal shift, restricted to `shift_search_area`
    correlation, optimal_offset = compute_correlation_at_optimal_offset(
        target_image,
        computed_image_no_shif_fft,
        image_config.get_coordinate_grid(physical=True),
        shift_search_area,
    )
    return -correlation, optimal_offset


def _batched_loss_for_grid_search(
    quats: Float[Array, "N 4"],
    volume: cxs.AbstractVolumeRepresentation,
    target_image: Float[Array, "H W"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    shift_search_area: Float[Array, "H W"] | None,
    n_angles_in_parallel: int,
) -> tuple[Float[Array, " N"], Float[Array, "N 2"]]:
    """
    Evaluate `_loss_for_grid_search` over a set of quaternions, computing
    `n_angles_in_parallel` of them at a time (vmapped) in a loop over batches.
    """
    return filter_bmap(
        lambda x: _loss_for_grid_search(
            x,
            volume,
            target_image,
            image_config,
            transfer_theory,
            shift_search_area,
        ),
        xs=quats,
        batch_size=n_angles_in_parallel,
    )


class _HierSO3GriSearchCarry(eqx.Module):
    losses: Float[Array, " N"]
    quats: Float[Array, "N 4"]
    offsets: Float[Array, "N 2"]
    grid_indices: Any
    curr_resolution: int


class HierarchicalSO3GridSearch(eqx.Module):
    """
    Perform a hierarchical grid search over SO3 to find the optimal pose
    using a shift-invariant metric based on correlation. Also provides
    the optimal shift for the pose found.

    **Attributes:**
    - `base_quats`:
        Base quaternions for the SO3 grid at the coarsest resolution.
    - `n_rounds`:
        Number of rounds to perform the hierarchical search.
    - `n_candidates`:
        Number of candidate quaternions to consider in each round.
    - `base_grid_res`:
        Base grid resolution for the SO3 grid.
    - `n_angles_in_parallel`:
        Number of orientations whose losses are evaluated in parallel
        (i.e. vmapped) within each batch of the search.
    - `shift_search_range_in_angstroms`:
        Half-width of the square region of shifts that is searched, in
        angstroms. If `None`, all shifts are searched.
    """

    base_quats: Float[Array, "N 4"]
    n_rounds: int
    n_candidates: int
    base_grid_res: int
    n_angles_in_parallel: int
    shift_search_range_in_angstroms: Float[Array, ""] | None

    def __init__(
        self,
        base_grid_res,
        n_rounds,
        n_candidates,
        n_angles_in_parallel=10,
        shift_search_range_in_angstroms=None,
    ):
        """
        Initialize the HierarchicalSO3GridSearch.

        **Arguments:**
        - `base_grid_res`:
            Base grid resolution for the SO3 grid.
        - `n_rounds`:
            Number of rounds to perform the hierarchical search.
        - `n_candidates`:
            Number of candidate quaternions to consider in each round.
        - `n_angles_in_parallel`:
            Number of orientations whose losses are evaluated in parallel
            (i.e. vmapped) within each batch of the search. Larger values
            are faster but use more memory.
        - `shift_search_range_in_angstroms`:
            Half-width of the square region of shifts that is searched, in
            angstroms. Only shifts with `|x| <= shift_search_range_in_angstroms`
            and `|y| <= shift_search_range_in_angstroms` are considered. If
            `None`, all shifts are searched.
        """
        assert base_grid_res >= 1, "Base grid must be at least 1."
        assert n_rounds >= 0, "Number of rounds must be non-negative."
        assert n_candidates >= 1, "Number of candidates must be at least 1."
        assert n_angles_in_parallel >= 1, (
            "Number of angles in parallel must be at least 1."
        )
        assert (
            shift_search_range_in_angstroms is None
            or shift_search_range_in_angstroms > 0.0
        ), "Shift search range must be positive."

        self.base_grid_res = base_grid_res
        self.base_quats = grid_SO3(base_grid_res)
        self.n_rounds = n_rounds
        self.n_candidates = n_candidates
        self.n_angles_in_parallel = n_angles_in_parallel
        self.shift_search_range_in_angstroms = (
            None
            if shift_search_range_in_angstroms is None
            else jnp.asarray(shift_search_range_in_angstroms)
        )

    def __call__(
        self,
        volume: cxs.AbstractVolumeRepresentation,
        target_image: Float[Array, "H W"],
        image_config: cxs.BasicImageConfig,
        transfer_theory: cxs.ContrastTransferTheory,
    ) -> cxs.QuaternionPose:
        """Perform the hierarchical SO3 grid search to find the optimal pose.

        **Arguments:**
        - `volume`:
            `cryojax` `AbstractVolumeRepresentation` representing a volumetric density.
        - `target_image`:
            The target image in Real space to compare against through the search.
        - `image_config`:
            Configuration for target_image simulation.
        - `transfer_theory`:
            Contrast transfer theory to use for target_image simulation.

        **Returns:**
        - `optimal_pose`:
            The optimal pose found through the hierarchical SO3 grid search.
        """

        # target_image = jnp.asarray(target_image)
        shift_search_area = _make_shift_search_area(
            self.shift_search_range_in_angstroms, image_config
        )
        losses, offsets = _batched_loss_for_grid_search(
            self.base_quats,
            volume,
            target_image,
            image_config,
            transfer_theory,
            shift_search_area,
            self.n_angles_in_parallel,
        )
        # jax.debug.print("Iter: 1. Best loss: {loss}", loss=losses.min())

        if self.n_rounds == 0:
            best_index = jnp.argmin(losses)
            optimal_pose = cxs.QuaternionPose(
                wxyz=self.base_quats[best_index],
                offset_x_in_angstroms=offsets[best_index, 0],
                offset_y_in_angstroms=offsets[best_index, 1],
            )

        else:
            # Do the first iteration outside the loop
            quats, grid_indices = getbestneighbors_base_SO3(
                losses,
                self.base_quats,
                N=self.n_candidates,
                base_resol=self.base_grid_res,
            )
            losses, offsets = _batched_loss_for_grid_search(
                quats,
                volume,
                target_image,
                image_config,
                transfer_theory,
                shift_search_area,
                self.n_angles_in_parallel,
            )
            carry = _HierSO3GriSearchCarry(
                losses=losses,
                quats=quats,
                grid_indices=grid_indices,
                offsets=offsets,
                curr_resolution=self.base_grid_res + 1,
            )
            # Then do the rest of the rounds in a loop
            carry = jax.lax.fori_loop(
                lower=2,
                upper=self.n_rounds,
                body_fun=lambda _, x: _run_global_SO3_step(
                    x,
                    self.n_candidates,
                    volume,
                    target_image,
                    image_config,
                    transfer_theory,
                    shift_search_area,
                    self.n_angles_in_parallel,
                ),
                init_val=carry,
            )
            best_index = jnp.argmin(carry.losses)
            optimal_pose = cxs.QuaternionPose(
                wxyz=carry.quats[best_index],
                offset_x_in_angstroms=carry.offsets[best_index, 0],
                offset_y_in_angstroms=carry.offsets[best_index, 1],
            )

        return optimal_pose


def _run_global_SO3_step(
    hier_grid_search_carry: _HierSO3GriSearchCarry,
    n_candidates: int,
    volume: cxs.AbstractVolumeRepresentation,
    target_image: Float[Array, "H W"],
    image_config: cxs.BasicImageConfig,
    transfer_theory: cxs.ContrastTransferTheory,
    shift_search_area: Float[Array, "H W"] | None,
    n_angles_in_parallel: int,
):
    quats, grid_indices = getbestneighbors_next_SO3(
        hier_grid_search_carry.losses,
        hier_grid_search_carry.quats,
        hier_grid_search_carry.grid_indices,
        N=n_candidates,
        curr_res=hier_grid_search_carry.curr_resolution,
    )
    losses, offsets = _batched_loss_for_grid_search(
        quats,
        volume,
        target_image,
        image_config,
        transfer_theory,
        shift_search_area,
        n_angles_in_parallel,
    )
    # jax.debug.print(
    #     "Iter: {iter}. Best loss: {loss}",
    #     iter=hier_grid_search_carry.curr_resolution,
    #     loss=losses.min(),
    # )
    return _HierSO3GriSearchCarry(
        losses=losses,
        quats=quats,
        offsets=offsets,
        grid_indices=grid_indices,
        curr_resolution=hier_grid_search_carry.curr_resolution + 1,
    )


def local_SO3_hier_search(lossfn, base_grid_res=1, n_rounds=5, n_candidates=40):
    raise NotImplementedError(
        "Local SO3 hierarchical search is not implemented yet. "
        "Please use global SO3 hierarchical search instead."
    )


def _make_shift_search_area(
    shift_search_range_in_angstroms: Float[Array, ""] | None,
    image_config: cxs.BasicImageConfig,
) -> Float[Array, "H W"] | None:
    """Build the mask restricting which shifts the search considers."""
    if shift_search_range_in_angstroms is None:
        return None
    return cxim.Rectangular2DCosineMask(
        image_config.get_coordinate_grid(physical=True),
        x_width=shift_search_range_in_angstroms * 2,
        y_width=shift_search_range_in_angstroms * 2,
        rolloff_width=0.1,
    ).get()
