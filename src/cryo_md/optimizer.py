import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

from cryo_md.md_engine import run_square_langevin
from cryo_md.likelihood.calc_lklhood import (
    calc_lkl_and_grad_wts,
    calc_lkl_and_grad_struct,
)


def optimize_weights(models, weights, steps, step_size, image_stack):
    losses = np.zeros(steps)
    for i in range(steps):
        loss, grad_wts = calc_lkl_and_grad_wts(
            models,
            weights,
            image_stack.images,
            image_stack.constant_params[0],
            image_stack.constant_params[1],
            image_stack.constant_params[2],
            image_stack.variable_params,
        )

        weights = weights + step_size * grad_wts
        weights /= jnp.sum(weights)

        losses[i] = loss

    return weights, losses


def run_optimizer(
    init_models,
    init_weights,
    image_stack,
    n_steps,
    step_size,
    gamma,
    batch_size=None,
):
    opt_models = init_models.copy()
    opt_weights = init_weights.copy()

    losses = np.zeros(n_steps)
    traj = np.zeros((n_steps, *opt_models.shape))
    
    traj_wts = np.zeros((n_steps, opt_weights.shape[0]))

    if batch_size is None:
        batch_size = image_stack.images.shape[0]

    for i in tqdm(range(n_steps)):

        opt_weights, _ = optimize_weights(opt_models, opt_weights, 10, 0.1, image_stack)

        # samples = run_square_langevin(
        #     opt_models + 0.0001, opt_models, n_steps=100, step_size=0.01
        # )

        # grad_prior = np.mean(samples[1:], axis=0) - opt_models
        # grad_prior /= jnp.max(jnp.abs(grad_prior), axis=(1))[:, None, :]

        random_batch = np.arange(0, image_stack.images.shape[0], 1)
        np.random.shuffle(random_batch)
        random_batch = random_batch[:batch_size]

        loss, grad_str = calc_lkl_and_grad_struct(
            opt_models,
            opt_weights,
            image_stack.images[random_batch],
            image_stack.constant_params[0],
            image_stack.constant_params[1],
            image_stack.constant_params[2],
            image_stack.variable_params[random_batch],
        )

        if loss == -np.inf:
            print("whoops")
            break

        losses[i] = loss

        grad_str /= jnp.max(jnp.abs(grad_str), axis=(1))[:, None, :]

        grad_total = gamma * grad_str #+ (1 - gamma) * grad_prior
        grad_total /= jnp.max(jnp.abs(grad_total), axis=(1))[:, None, :]

        opt_models = opt_models + step_size * grad_total

        # indices = np.argmin(
        #     np.linalg.norm(samples - opt_models[None, :, :, :], axis=(2, 3)), axis=0
        # )
        # for j in range(opt_models.shape[0]):
        #     opt_models = opt_models.at[j].set(samples[indices[j], j, :, :])

        traj[i] = opt_models.copy()
        traj_wts[i] = opt_weights.copy()

    return traj, traj_wts, losses
