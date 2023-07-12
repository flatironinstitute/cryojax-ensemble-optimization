import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from functools import partial
from typing import Union

from cryo_md.likelihood.simulator_for_lklhood import noiseless_simulator_


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


batch_over_models = jax.jit(
    jax.vmap(compare_coords_with_img_, in_axes=(0, None, None, None, None, None)),
    static_argnums=(2, 3),
)

batch_over_images = jax.jit(
    jax.vmap(batch_over_models, in_axes=(None, 0, None, None, None, 0)),
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
    lklhood_matrix = batch_over_images(
        models, images, box_size, pixel_size, sigma, variable_params
    )

    model_weights = model_weights

    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=model_weights[None, :], axis=1
    )

    log_lklhood = jnp.sum(log_lklhood)

    return log_lklhood


calc_lkl_and_grad_struct = jax.jit(
    jax.value_and_grad(calc_lklhood_, argnums=0), static_argnums=(3, 4)
)

calc_lkl_and_grad_wts = jax.jit(
    jax.value_and_grad(calc_lklhood_, argnums=1), static_argnums=(3, 4)
)
