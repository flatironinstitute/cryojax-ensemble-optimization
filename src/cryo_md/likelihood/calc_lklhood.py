import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
from typing import Union, Tuple

from cryo_md.likelihood.simulator_for_lklhood import noiseless_simulator_
from cryo_md.image.image_stack import ImageStack


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def compare_coords_with_img_(
    coords: ArrayLike,
    image_ref: ArrayLike,
    box_size: int,
    pixel_size: float,
    sigma: float,
    var_imaging_args: ArrayLike,
) -> float:
    """
    Compare the coordinates with the image and return the log-likelihood. This function is used in the likelihood calculation.

    Parameters
    ----------
    coords : ArrayLike
        Coordinates of the model
    image_ref : ArrayLike
        Image to compare with
    box_size : int
        Size of the box
    pixel_size : float
        Pixel size
    sigma : float
        Standard deviation of the Gaussian
    var_imaging_args : ArrayLike
        Imaging parameters

    Returns
    -------
    float
        Log-likelihood
    """
    image_coords = noiseless_simulator_(
        coords, box_size, pixel_size, sigma, var_imaging_args
    )

    return (
        -0.5
        * jnp.linalg.norm(image_coords - image_ref) ** 2
        / var_imaging_args[10] ** 2
    )


batch_over_models_ = jax.jit(
    jax.vmap(compare_coords_with_img_, in_axes=(0, None, None, None, None, None)),
    static_argnums=(2, 3),
)

batch_over_images_ = jax.jit(
    jax.vmap(batch_over_models_, in_axes=(None, 0, None, None, None, 0)),
    static_argnums=(2, 3),
)


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def calc_lklhood_(
    models: ArrayLike,
    model_weights: ArrayLike,
    images: ArrayLike,
    box_size: int,
    pixel_size: float,
    sigma: float,
    variable_params: ArrayLike,
) -> float:
    lklhood_matrix = batch_over_images_(
        models, images, box_size, pixel_size, sigma, variable_params
    )

    model_weights = model_weights

    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=model_weights[None, :], axis=1
    )

    log_lklhood = jnp.sum(log_lklhood)

    return log_lklhood


def calc_likelihood(
    models: ArrayLike, model_weights: ArrayLike, image_stack: ImageStack
) -> float:
    """
    Calculate the log-likelihood.

    Parameters
    ----------
    models : ArrayLike
        Models to compare with
    model_weights : ArrayLike
        Weights of the models
    image_stack : ImageStack
        Image stack

    Returns
    -------
    float
        Log-likelihood
    """

    likelihood = calc_lklhood_(
        models,
        model_weights,
        image_stack.images,
        image_stack.constant_params[0],
        image_stack.constant_params[1],
        image_stack.constant_params[2],
        image_stack.variable_params,
    )

    return likelihood


calc_lkl_and_grad_struct_ = jax.jit(
    jax.value_and_grad(calc_lklhood_, argnums=0), static_argnums=(3, 4)
)

calc_lkl_and_grad_wts_ = jax.jit(
    jax.value_and_grad(calc_lklhood_, argnums=1), static_argnums=(3, 4)
)


def calc_lkl_and_grad_struct(
    models: ArrayLike, model_weights: ArrayLike, image_stack: ImageStack
) -> Tuple[float, ArrayLike]:
    """
    Calculate the log-likelihood and its gradient with respect to the structure.

    Parameters
    ----------
    models : ArrayLike
        Models to compare with
    model_weights : ArrayLike
        Weights of the models
    image_stack : ImageStack
        Image stack

    Returns
    -------
    float
        Log-likelihood
    ArrayLike
        Gradient with respect to the structure
    """

    log_lklhood, grad_str = calc_lkl_and_grad_struct_(
        models,
        model_weights,
        image_stack.images,
        image_stack.constant_params[0],
        image_stack.constant_params[1],
        image_stack.constant_params[2],
        image_stack.variable_params,
    )

    return log_lklhood, grad_str


def calc_lkl_and_grad_wts(
    models: ArrayLike, model_weights: ArrayLike, image_stack: ImageStack
) -> Tuple[float, ArrayLike]:
    """
    Calculate the log-likelihood and its gradient with respect to the model weights.

    Parameters
    ----------
    models : ArrayLike
        Models to compare with
    model_weights : ArrayLike
        Weights of the models
    image_stack : ImageStack
        Image stack

    Returns
    -------
    float
        Log-likelihood
    ArrayLike
        Gradient with respect to the model weights
    """

    log_lklhood, grad_wts = calc_lkl_and_grad_wts_(
        models,
        model_weights,
        image_stack.images,
        image_stack.constant_params[0],
        image_stack.constant_params[1],
        image_stack.constant_params[2],
        image_stack.variable_params,
    )

    return log_lklhood, grad_wts
