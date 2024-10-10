from functools import partial

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp

import cryojax.simulator as cxs


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, 0, None), out_axes=eqxi.if_mapped(axis=0))
def _compute_noise_variance(key, relion_particle_stack_vmap, args):
    relion_particle_stack_novmap, potentials, potential_integrator, mask, config = args
    relion_particle_stack = eqx.combine(
        relion_particle_stack_vmap, relion_particle_stack_novmap
    )

    key, subkey = jax.random.split(key)

    snr = jax.random.uniform(
        subkey, (1,), minval=config["noise_snr"][0], maxval=config["noise_snr"][1]
    )
    key, subkey = jax.random.split(key)

    relion_particle_stack = eqx.tree_at(
        lambda d: (d.pose.offset_x_in_angstroms, d.pose.offset_y_in_angstroms),
        relion_particle_stack,
        replace_fn=lambda x: 0.0 * x,
    )

    structural_ensemble = cxs.DiscreteStructuralEnsemble(
        potentials, relion_particle_stack.pose, cxs.DiscreteConformationalVariable(0)
    )

    scattering_theory = cxs.WeakPhaseScatteringTheory(
        structural_ensemble,
        potential_integrator,
        relion_particle_stack.transfer_theory,
    )

    imaging_pipeline = cxs.ContrastImagingPipeline(
        relion_particle_stack.instrument_config, scattering_theory
    )

    signal = imaging_pipeline.render(get_real=True)
    signal /= jnp.linalg.norm(signal)
    # signal -= jnp.sum(signal * mask.array) / jnp.sum(mask.array)

    signal_variance = jnp.var(
        signal, where=jnp.where(mask.array == 1.0, True, False)
    )  # jnp.sum((signal * mask.array) ** 2) / (jnp.sum(mask.array) - 1)

    noise_variance = signal_variance / snr

    return noise_variance
