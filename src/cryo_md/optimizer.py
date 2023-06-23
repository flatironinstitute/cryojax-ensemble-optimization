import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from cryo_md.md_engine import run_square_langevin
from cryo_md.likelihood.calc_lklhood import calc_lkl_and_grad, get_neigh_list


def run_optimizer(
    init_models, init_weights, image_stack, n_steps, step_size, gamma, batch_size=None, stride_neigh_list=1
):
    opt_models = init_models.copy()
    opt_weights = init_weights.copy()

    losses = np.zeros(n_steps)
    traj = np.zeros((n_steps + 1, *opt_models.shape))
    traj[0] = opt_models.copy()

    if batch_size is None:
        batch_size = image_stack.images.shape[0]

    for i in tqdm(range(n_steps)):

        # Run MD
        samples = run_square_langevin(
            opt_models + 0.0001, opt_models, n_steps=100, step_size=0.01
        )

        grad_prior = -np.mean(opt_models[None, :, :, :] - samples[1:], axis=0)
        grad_prior /= jnp.max(jnp.abs(grad_prior), axis=(1))[:, None, :]

        # Set up stochastic grad descent
        random_batch = np.arange(0, image_stack.images.shape[0], 1)
        np.random.shuffle(random_batch)
        random_batch = random_batch[:batch_size]

        # Update Neighbor List
        if i % stride_neigh_list == 0:
            neigh_list = get_neigh_list(opt_models, image_stack)

        # Optimize structure
        loss, grad = calc_lkl_and_grad(
            opt_models,
            opt_weights,
            image_stack.images[random_batch],
            image_stack.constant_params[0],
            image_stack.constant_params[1],
            image_stack.constant_params[2],
            image_stack.variable_params[random_batch],
            1.0,
            neigh_list[random_batch],
        )

        losses[i] = loss
        grad_str = grad[0]
        grad_wts = grad[1]

        grad_str /= jnp.max(jnp.abs(grad_str), axis=(1))[:, None, :]

        grad_total = gamma * grad_str + (1 - gamma) * grad_prior
        grad_total /= jnp.max(jnp.abs(grad_total), axis=(1))[:, None, :]

        opt_models = opt_models + step_size * grad_total

        # Find closest structure from prior sample to the opt. structure
        indices = np.argmin(
            np.linalg.norm(samples - opt_models[None, :, :, :], axis=(2, 3)), axis=0
        )
        for j in range(opt_models.shape[0]):
            opt_models = opt_models.at[j].set(samples[indices[j], j, :, :])

        opt_weights = opt_weights + 0.01 * grad_wts
        opt_weights /= jnp.sum(opt_weights)

        traj[i + 1] = opt_models.copy()

    return traj, opt_weights, losses
