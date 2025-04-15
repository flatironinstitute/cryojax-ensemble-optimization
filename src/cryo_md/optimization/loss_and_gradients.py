from functools import partial

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxopt import ProjectedGradient
from jaxopt.projection import projection_simplex


import cryojax.simulator as cxs

from ..simulator._distributions import VarianceMarginalizedWhiteGaussianNoise


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(None, eqx.if_array(0), None), out_axes=0)
@partial(eqx.filter_vmap, in_axes=(0, None, None), out_axes=0)
def compute_lklhood_matrix(atom_positions, relion_stack, args):
    atom_identities, b_factors, parameter_table, noise_variance = (
        args
    )

    potential = cxs.PengAtomicPotential(
        atom_positions,
        atom_identities,
        b_factors,
        scattering_factor_parameter_table=parameter_table,
    )
    structural_ensemble = cxs.SingleStructureEnsemble(
        potential, relion_stack.parameters.pose
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble=structural_ensemble,
        potential_integrator=cxs.GaussianMixtureProjection(),
        transfer_theory=relion_stack.parameters.transfer_theory,
    )

    imaging_pipeline = cxs.ContrastImageModel(
        relion_stack.parameters.instrument_config, scattering_theory
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

#     imaging_pipeline = cxs.ContrastImageModel(
#         relion_stack.instrument_config, scattering_theory
#     )
#     distribution = WhiteGaussianNoise(imaging_pipeline, noise_variance)

#     return distribution.log_likelihood(relion_stack.image_stack)


@eqx.filter_jit
def compute_loss(weights, lklhood_matrix):
    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=weights[None, :], axis=1
    )
    return -jnp.mean(log_lklhood)


@eqx.filter_jit
@partial(jax.value_and_grad, argnums=0, has_aux=True)
def compute_loss_weights_and_grads(atom_positions, weights, relion_stack_vmap, args):
    lklhood_matrix = compute_lklhood_matrix(atom_positions, relion_stack_vmap, args)

    pg = ProjectedGradient(fun=compute_loss, projection=projection_simplex)
    weights = pg.run(weights, lklhood_matrix=lklhood_matrix).params
    weights = jax.nn.softmax(weights)

    return compute_loss(weights, lklhood_matrix), weights
