from typing import Any, Optional
from typing_extensions import Literal

import equinox as eqx
import jax
from cryojax.jax_util import error_if_not_positive
from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex
from jaxtyping import Array, Float, Int

from ...._custom_types import ConstantT, LossFn
from ....simulator import DilatedMask
from .ensemble_losses import (
    compute_likelihood_matrix,
    compute_neg_log_likelihood,
    compute_neg_log_likelihood_from_weights,
)
from .euclidean_losses import (
    likelihood_isotropic_gaussian,
    likelihood_isotropic_gaussian_marginalized,
)
from .sliced_wasserstein import likelihood_sliced_wasserstein


class AbstractLikelihoodFn(eqx.Module, strict=True):
    variances: eqx.AbstractVar[Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]]
    amplitudes: eqx.AbstractVar[Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]]
    image_to_walker_log_likelihood_fn: eqx.AbstractVar[LossFn]
    loss_fn_constant_args: eqx.AbstractVar[ConstantT]
    dilated_mask: eqx.AbstractClassVar[Optional[DilatedMask]]
    estimates_pose: eqx.AbstractVar[bool]

    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        relion_batch: Any,
    ) -> Float:
        raise NotImplementedError


class LikelihoodFn(AbstractLikelihoodFn, strict=True):
    variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    image_to_walker_log_likelihood_fn: LossFn
    loss_fn_constant_args: ConstantT
    dilated_mask: Optional[DilatedMask] = None
    estimates_pose: bool = False

    def __init__(
        self,
        amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        image_to_walker_log_likelihood_fn: Literal[
            "iso_gaussian", "iso_gaussian_var_marg", "sliced_wasserstein"
        ]
        | LossFn,
        loss_fn_constant_args: Optional[ConstantT] = None,
        dilated_mask: Optional[DilatedMask] = None,
        estimates_pose: bool = False,
    ):
        self.variances = error_if_not_positive(variances)
        self.amplitudes = error_if_not_positive(amplitudes)
        if image_to_walker_log_likelihood_fn == "iso_gaussian":
            self.image_to_walker_log_likelihood_fn = likelihood_isotropic_gaussian
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_walker_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_walker_log_likelihood_fn = (
                likelihood_isotropic_gaussian_marginalized
            )
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_walker_log_likelihood_fn == "sliced_wasserstein":
            self.image_to_walker_log_likelihood_fn = likelihood_sliced_wasserstein
            self.loss_fn_constant_args = (
                (18, 2) if loss_fn_constant_args is None else loss_fn_constant_args
            )
        else:
            assert callable(image_to_walker_log_likelihood_fn), (
                "If `image_to_walker_log_likelihood_fn` is not 'iso_gaussian' or "
                + "'iso_gaussian_var_marg', it must be a callable function."
            )
            self.image_to_walker_log_likelihood_fn = image_to_walker_log_likelihood_fn
            self.loss_fn_constant_args = loss_fn_constant_args

        self.dilated_mask = dilated_mask
        self.estimates_pose = estimates_pose

    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        relion_batch: Any,
    ):
        return compute_neg_log_likelihood(
            walkers,
            weights,
            relion_batch["particle_stack"],
            self.amplitudes,
            self.variances,
            self.image_to_walker_log_likelihood_fn,
            self.dilated_mask,
            self.estimates_pose,
            constant_args=self.loss_fn_constant_args,
            per_particle_args=relion_batch["per_particle_args"],
        )


class LikelihoodOptimalWeightsFn(AbstractLikelihoodFn, strict=True):
    variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"]
    image_to_walker_log_likelihood_fn: LossFn
    loss_fn_constant_args: ConstantT
    dilated_mask: Optional[DilatedMask] = None
    estimates_pose: bool = False

    def __init__(
        self,
        amplitudes: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        variances: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        image_to_walker_log_likelihood_fn: Literal[
            "iso_gaussian", "iso_gaussian_var_marg"
        ]
        | LossFn,
        loss_fn_constant_args: Optional[ConstantT] = None,
        dilated_mask: Optional[DilatedMask] = None,
        estimates_pose: bool = False,
    ):
        self.variances = error_if_not_positive(variances)
        self.amplitudes = error_if_not_positive(amplitudes)
        if image_to_walker_log_likelihood_fn == "iso_gaussian":
            self.image_to_walker_log_likelihood_fn = likelihood_isotropic_gaussian
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_walker_log_likelihood_fn == "iso_gaussian_var_marg":
            self.image_to_walker_log_likelihood_fn = (
                likelihood_isotropic_gaussian_marginalized
            )
            self.loss_fn_constant_args = (
                1.0 if loss_fn_constant_args is None else loss_fn_constant_args
            )
        elif image_to_walker_log_likelihood_fn == "sliced_wasserstein":
            self.image_to_walker_log_likelihood_fn = likelihood_sliced_wasserstein
            self.loss_fn_constant_args = (
                (18, 2) if loss_fn_constant_args is None else loss_fn_constant_args
            )
        else:
            assert callable(image_to_walker_log_likelihood_fn), (
                "If `image_to_walker_log_likelihood_fn` is not 'iso_gaussian' or "
                + "'iso_gaussian_var_marg', it must be a callable function."
            )
            self.image_to_walker_log_likelihood_fn = image_to_walker_log_likelihood_fn
            self.loss_fn_constant_args = loss_fn_constant_args

        self.dilated_mask = dilated_mask
        self.estimates_pose = estimates_pose

    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        relion_batch: Any,
    ):
        likelihood_matrix = compute_likelihood_matrix(
            walkers,
            relion_batch["particle_stack"],
            self.amplitudes,
            self.variances,
            self.image_to_walker_log_likelihood_fn,
            self.dilated_mask,
            self.estimates_pose,
            constant_args=self.loss_fn_constant_args,
            per_particle_args=relion_batch["per_particle_args"],
        )
        weights = _optimize_weights(weights, likelihood_matrix)
        weights = jax.nn.softmax(weights)
        return compute_neg_log_likelihood_from_weights(
            weights, likelihood_matrix
        ), weights


@eqx.filter_jit
def _optimize_weights(
    weights: Float[Array, " n_walkers"],
    likelihood_matrix: Float[Array, "n_images n_walkers"],
    n_steps: Int = 500,
) -> Float[Array, " n_walkers"]:
    pg = ProjectedGradient(
        fun=compute_neg_log_likelihood_from_weights,
        projection=projection_simplex,
        maxiter=n_steps,
    )
    return pg.run(weights, likelihood_matrix=likelihood_matrix).params
