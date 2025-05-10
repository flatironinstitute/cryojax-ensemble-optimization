from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp

import cryojax.simulator as cxs


@eqx.filter_jit
@partial(eqx.filter_vmap, in_axes=(0, eqx.if_array(0), None), out_axes=0)
def _compute_noise_variance(key, relion_particle_stack, args):
    potentials, potential_integrator, mask, config = args

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

    image_model = cxs.ContrastImageModel(
        relion_particle_stack.instrument_config, scattering_theory
    )

    signal = image_model.render(outputs_real_space=True)
    signal /= jnp.linalg.norm(signal)
    # signal -= jnp.sum(signal * mask.array) / jnp.sum(mask.array)

    signal_variance = jnp.var(
        signal, where=jnp.where(mask.array == 1.0, True, False)
    )  # jnp.sum((signal * mask.array) ** 2) / (jnp.sum(mask.array) - 1)

    noise_variance = signal_variance / snr

    return noise_variance
