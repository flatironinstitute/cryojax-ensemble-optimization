import jax.numpy as jnp
import jax
import numpy as np
from jax.typing import ArrayLike
from typing import Dict, Tuple


def calc_lklhood(
    models: np.ndarray, weights: np.ndarray, data: np.ndarray, sigma: float
) -> ArrayLike:
    """
    Calculates likelihood between a set of images and a set of structures as in Equation 20. for the Toy Model.
    """

    log_lklhood = (
        -0.5
        * jnp.sum((data[:, None, :] - models[None, :, :]) ** 2, axis=2)
        / sigma**2
    )

    log_lklhood = jax.scipy.special.logsumexp(
        a=log_lklhood, b=weights[None, :] / (sigma**2 * 2 * jnp.pi), axis=1
    )
    log_lklhood = jnp.sum(log_lklhood)

    return log_lklhood


def calc_lklhood_grad_strucs(
    models: np.ndarray, weights: np.ndarray, data: np.ndarray, sigma: float
) -> ArrayLike:
    log_lklhood = (
        -0.5
        * jnp.sum((data[:, None, :] - models[None, :, :]) ** 2, axis=2)
        / sigma**2
    )

    grad_lklhood_weights = jax.scipy.special.logsumexp(
        a=log_lklhood, b=weights[None, :], axis=1
    )

    data_model_diff = data[:, None, :] - models[None, :, :]

    mask_pos = data_model_diff > 0
    mask_neg = data_model_diff < 0

    log_gradient_pos = jax.scipy.special.logsumexp(
        a=(log_lklhood - grad_lklhood_weights[:, None])[:, :, None],
        b=data_model_diff * mask_pos.astype(jnp.float64),
        axis=0,
    )

    log_gradient_neg = jax.scipy.special.logsumexp(
        a=(log_lklhood - grad_lklhood_weights[:, None])[:, :, None],
        b=-data_model_diff * mask_neg.astype(jnp.float64),
        axis=0,
    )

    grad_lklhood_strucs = jnp.exp(log_gradient_pos) - jnp.exp(log_gradient_neg)
    grad_lklhood_strucs *= weights[:, None] / (sigma**2)

    return grad_lklhood_strucs


def calc_lklhood_grad_weights(
    models: np.ndarray, weights: np.ndarray, data: np.ndarray, sigma: float
) -> ArrayLike:
    log_lklhood = (
        -0.5
        * jnp.sum((data[:, None, :] - models[None, :, :]) ** 2, axis=2)
        / sigma**2
    )

    grad_lklhood_weights = jax.scipy.special.logsumexp(
        a=log_lklhood, b=weights[None, :], axis=1
    )

    grad_lklhood_weights = jax.scipy.special.logsumexp(
        a=(log_lklhood - grad_lklhood_weights[:, None]),
        axis=0,
    )

    grad_lklhood_weights = jnp.exp(grad_lklhood_weights)

    return grad_lklhood_weights


calc_grad_lklhood_jax = jax.jit(jax.grad(calc_lklhood, argnums=[0, 1]))


def calc_gradient(
    models: np.ndarray,
    samples: np.ndarray,
    weights: np.ndarray,
    data: np.ndarray,
    config: Dict,
) -> Tuple[ArrayLike, ArrayLike]:
    grad_prior = (
        config["gamma"]
        / config["delta_sigma"] ** 2
        * np.mean(samples - models[None, :, :], axis=0)
    )

    grad_lklhood = calc_grad_lklhood_jax(models, weights, data, config["sigma"])

    gradient_strucs = grad_lklhood[0] / data.shape[0] + grad_prior
    gradient_weights = grad_lklhood[1] / data.shape[0]

    return (gradient_strucs, gradient_weights)
