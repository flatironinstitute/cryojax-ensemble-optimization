from typing import Any, cast

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
from cryojax.jax_util import filter_bmap
from jax_dataloader import DataLoader
from jaxtyping import Array, Float
from tqdm import tqdm

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
        return _compute_likelihoods_over_walkers(
            self.image_to_walker_likelihood_fn,
            walkers,
            images,
            image_config,
            poses_per_walker,
            transfer_theories,
            per_particle_args,
            self.n_walkers_in_parallel,
            self.n_images_in_parallel,
        ).T

    def compute_from_log_likelihood_matrix(
        self,
        log_likelihood_matrix: Float[Array, "n_images n_walkers"],
        weights: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    ):
        return jnp.mean(
            jax.scipy.special.logsumexp(
                a=log_likelihood_matrix, b=weights[None, :], axis=1
            )
        )

    def compute_full_log_likelihood_matrix(
        self,
        walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
        dataloader: DataLoader,
        *,
        prints_progress=False,
    ) -> Float[Array, "n_images n_walkers"]:
        compute_likelihood_matrix_fn = eqx.filter_jit(self.compute_log_likelihood_matrix)

        shuffle = dataloader.dataloader.shuffle  # save the original shuffle state
        dataloader.dataloader.shuffle = False
        # Compute the likelihood matrix for each batch in the dataloader
        likelihood_matrix = []

        iterable = (
            dataloader
            if not prints_progress
            else tqdm(dataloader, desc="Computing full likelihood matrix", unit=" batch")
        )
        for batch in iterable:
            poses_per_walker = jax.tree.map(
                lambda x: jnp.repeat(x[None, :], repeats=walkers.shape[0], axis=0),
                batch["particle_stack"]["parameters"]["pose"],
            )
            poses_per_walker = cast(cxs.EulerAnglePose, poses_per_walker)

            lklhood_matrix = compute_likelihood_matrix_fn(
                walkers,
                batch["particle_stack"]["images"],
                batch["particle_stack"]["parameters"]["image_config"],
                poses_per_walker,
                batch["particle_stack"]["parameters"]["transfer_theory"],
                batch["per_particle_args"],
            )
            likelihood_matrix.append(lklhood_matrix)

        # restore the original shuffle state
        dataloader.dataloader.shuffle = shuffle

        # Concatenate the likelihood matrices from all batches
        return jnp.concatenate(likelihood_matrix, axis=0)

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
        return self.compute_from_log_likelihood_matrix(likelihood_matrix, weights)


def _compute_likelihoods_single_walker_bmap(
    likelihood_fn: AbstractImageToWalkerLogLikelihoodFn,
    walker: Float[Array, "n_atoms n_gaussians_per_atom"],
    images: Float[Array, "n_images y x"],
    image_config: cxs.BasicImageConfig,
    poses_per_walker: cxs.AbstractPose,
    transfer_theories: cxs.ContrastTransferTheory,
    per_particle_args: Any,
    n_images_in_parallel: int,
):
    @eqx.filter_vmap(in_axes=(0, None, 0, 0, 0))
    def _body_fn(img, ic, p, tf, ppa):
        return likelihood_fn(
            walker,
            img,
            ic,
            p,
            tf,
            ppa,
        )

    return filter_bmap(
        lambda x: _body_fn(x[0], image_config, x[1], x[2], x[3]),
        (images, poses_per_walker, transfer_theories, per_particle_args),
        batch_size=n_images_in_parallel,
    )


def _compute_likelihoods_over_walkers(
    likelihood_fn: AbstractImageToWalkerLogLikelihoodFn,
    walkers: Float[Array, "n_walkers n_atoms n_gaussians_per_atom"],
    images: Float[Array, "n_images y x"],
    image_config: cxs.BasicImageConfig,
    poses_per_walker: cxs.AbstractPose,
    transfer_theories: cxs.ContrastTransferTheory,
    per_particle_args: Any,
    n_walkers_in_parallel: int,
    n_images_in_parallel: int,
):
    @eqx.filter_vmap(in_axes=(0, None, None, 0, None, None))
    def _body_fn(w, im, ic, p, tf, ppa):
        return _compute_likelihoods_single_walker_bmap(
            likelihood_fn,
            w,
            im,
            ic,
            p,
            tf,
            ppa,
            n_images_in_parallel,
        )

    return filter_bmap(
        lambda x: _body_fn(
            x[0], images, image_config, x[1], transfer_theories, per_particle_args
        ),
        xs=(walkers, poses_per_walker),
        batch_size=n_walkers_in_parallel,
    )
