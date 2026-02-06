from typing import Any

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.jax_util import filter_bmap
from jaxtyping import Array, Float

from .single_likelihood import AbstractImageToWalkerLogLikelihoodFn


class AbstractImagesToEnsembleLikelihoodFn(eqx.Module):
    image_to_walker_likelihood_fn: eqx.AbstractVar[AbstractImageToWalkerLogLikelihoodFn]
    n_walkers_in_parallel: eqx.AbstractVar[int]
    n_images_in_parallel: eqx.AbstractVar[int]

    def compute_log_likelihood_matrix(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        images: Float[Array, "n_images y x"],
        image_config: cxs.BasicImageConfig,
        poses_per_walker: cxs.AbstractPose,
        transfer_theories: cxs.ContrastTransferTheory,
        per_particle_args: Any,
    ) -> Float[Array, "n_images n_walkers"]:
        raise NotImplementedError

    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        images: Float[Array, "n_images y x"],
        image_config: cxs.BasicImageConfig,
        poses_per_walker: cxs.AbstractPose,
        transfer_theory: cxs.ContrastTransferTheory,
        per_particle_args: Any,
    ):
        raise NotImplementedError


class ImagesToEnsembleLikelihoodFn(AbstractImagesToEnsembleLikelihoodFn):
    image_to_walker_likelihood_fn: AbstractImageToWalkerLogLikelihoodFn
    n_walkers_in_parallel: int = 1
    n_images_in_parallel: int = 1

    def compute_log_likelihood_matrix(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        images: Float[Array, "n_images y x"],
        image_config: cxs.BasicImageConfig,
        poses_per_walker: cxs.AbstractPose,
        transfer_theories: cxs.ContrastTransferTheory,
        per_particle_args: Any,
    ) -> Float[Array, "n_images n_walkers"]:
        vmap_over_images_fn = eqx.filter_vmap(
            lambda w, img, ic, p, tf, ppa: self.image_to_walker_likelihood_fn(
                w, img, ic, p, tf, ppa
            ),
            in_axes=(
                None,
                0,
                eqx.if_array(0),
                0,
                0,
                0,
            ),
        )

        vmap_over_walkers_fn = eqx.filter_vmap(
            lambda w, img, ic, p, tf, ppa: filter_bmap(
                f=lambda x: vmap_over_images_fn(w, x[0], x[1], x[2], x[3], x[4]),
                xs=(img, ic, p, tf, ppa),
                batch_size=self.n_images_in_parallel,
            ),
            in_axes=(0, None, None, 0, None, None),
        )
        return filter_bmap(
            lambda x: vmap_over_walkers_fn(
                x[0], images, image_config, x[1], transfer_theories, per_particle_args
            ),
            (walkers, poses_per_walker),
            batch_size=self.n_walkers_in_parallel,
        ).T

    def __call__(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        images: Float[Array, "n_images y x"],
        image_config: cxs.BasicImageConfig,
        poses_per_walker: cxs.AbstractPose,
        transfer_theories: cxs.ContrastTransferTheory,
        per_particle_args: Any,
    ):
        likelihood_matrix = self.compute_log_likelihood_matrix(
            walkers,
            images,
            image_config,
            poses_per_walker,
            transfer_theories,
            per_particle_args,
        )
        return jnp.mean(
            jax.scipy.special.logsumexp(a=likelihood_matrix, b=weights[None, :], axis=1)
        )
