import numpy as np
import jax.numpy as jnp
import jax


def calc_dists(v1, v2):
    return jnp.linalg.norm(v1 - v2, axis=0)


def calc_angle(v1, v2):
    return jnp.arccos(jnp.dot(v1, v2) / (jnp.linalg.norm(v1) * jnp.linalg.norm(v2)))


def print_dist(model):
    model_roll = jnp.roll(model, shift=1, axis=1)
    model_diff_roll = jnp.roll(model - model_roll, shift=1, axis=1)

    dists = calc_dists(model, model_roll).round(1)
    angles = (calc_angles(model - model_roll, model_diff_roll) * 180 / np.pi).round(1)

    return (dists, angles)


calc_angles = jax.vmap(calc_angle, in_axes=(1, 1))


def calc_bias_force(model, ref_models):
    return jnp.linalg.norm(model - ref_models) ** 2


def calc_energy(model, eq_dist, k_dist, k_ang, ref_for_bias, k_bias):
    model_roll = jnp.roll(model, shift=1, axis=1)
    model_diff_roll = jnp.roll(model - model_roll, shift=1, axis=1)

    dists = calc_dists(model, model_roll)
    angles = calc_angles(model - model_roll, model_diff_roll)

    energy = (
        -0.5 * k_dist * jnp.sum((dists - eq_dist) ** 2)
        - 0.5 * k_ang * jnp.sum((angles - np.pi * 0.5) ** 2)
        - 0.5 * k_bias * calc_bias_force(model, ref_for_bias)
    )

    return energy


calc_energy_grad = jax.jit(jax.grad(calc_energy))

calc_energy_grad_all = jax.vmap(
    calc_energy_grad, in_axes=(0, None, None, None, 0, None)
)


def run_square_langevin(init_models, ref_models, n_steps, step_size):
    traj = np.zeros((n_steps + 1, *init_models.shape))
    traj[0] = init_models

    for i in range(n_steps):
        grad = calc_energy_grad_all(traj[i], 48.0, 0.0, 100.0, ref_models, 5.0)

        traj[i + 1] = (
            traj[i]
            + step_size * grad
            + np.sqrt(2 * step_size) * np.random.randn(*init_models.shape)
        )

    return traj
