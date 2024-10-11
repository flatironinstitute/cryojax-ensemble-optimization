from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.internal as eqxi


import cryojax.simulator as cxs

from ..simulator._distributions import VarianceMarginalizedWhiteGaussianNoise


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(None, 0, None), out_axes=eqxi.if_mapped(axis=0))
@partial(eqx.filter_vmap, in_axes=(0, None, None), out_axes=eqxi.if_mapped(axis=0))
def compute_lklhood_matrix(atom_positions, relion_stack_vmap, args):
    atom_identities, b_factors, noise_variance, relion_stack_novmap = args

    relion_stack = eqx.combine(relion_stack_vmap, relion_stack_novmap)

    potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
    structural_ensemble = cxs.SingleStructureEnsemble(potential, relion_stack.pose)

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble=structural_ensemble,
        potential_integrator=cxs.GaussianMixtureProjection(),
        transfer_theory=relion_stack.transfer_theory,
    )

    imaging_pipeline = cxs.ContrastImagingPipeline(
        relion_stack.instrument_config, scattering_theory
    )
    distribution = VarianceMarginalizedWhiteGaussianNoise(imaging_pipeline)
    # distribution = WhiteGaussianNoise(imaging_pipeline, noise_variance)

    return distribution.log_likelihood(relion_stack.image_stack)


# @partial(eqx.filter_vmap, in_axes=(None, 0, None), out_axes=eqxi.if_mapped(axis=0))
# @partial(eqx.filter_vmap, in_axes=(0, None, None), out_axes=eqxi.if_mapped(axis=0))
# def compute_lklhood_matrix(atom_positions, relion_stack_vmap, args):
#     atom_identities, b_factors, noise_variance, relion_stack_novmap = args

#     relion_stack = eqx.combine(relion_stack_vmap, relion_stack_novmap)

#     potential = cxs.PengAtomicPotential(atom_positions, atom_identities, b_factors)
#     structural_ensemble = cxs.SingleStructureEnsemble(potential, relion_stack.pose)

#     scattering_theory = cxs.WeakPhaseScatteringTheory(
#         structural_ensemble=structural_ensemble,
#         potential_integrator=cxs.GaussianMixtureProjection(),
#         transfer_theory=relion_stack.transfer_theory,
#     )

#     imaging_pipeline = cxs.ContrastImagingPipeline(
#         relion_stack.instrument_config, scattering_theory
#     )
#     distribution = WhiteGaussianNoise(imaging_pipeline, noise_variance)

#     return distribution.log_likelihood(relion_stack.image_stack)


def compute_loss(atom_positions, model_weights, relion_stack_vmap, args):
    lklhood_matrix = compute_lklhood_matrix(atom_positions, relion_stack_vmap, args)
    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=model_weights[None, :], axis=1
    )

    return -jnp.mean(log_lklhood)


compute_loss_and_grads_positions = eqx.filter_jit(
    jax.value_and_grad(compute_loss, argnums=0)
)
compute_loss_and_grads_weights = eqx.filter_jit(
    jax.value_and_grad(compute_loss, argnums=1)
)
