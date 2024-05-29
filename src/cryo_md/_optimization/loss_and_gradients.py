import jax
import jax.numpy as jnp
from jaxtyping import Array
from typing import Tuple
import logging

from .._simulator.simulator import simulator_
from .._data.particle_dataloader import NumpyLoader


def compare_coords_with_img_(
    coords: Array,
    image_ref: Array,
    struct_info: Array,
    grid: Array,
    grid_f: Array,
    pose_params: Array,
    ctf_params: Array,
    noise_var: Array,
) -> float:
    """
    Compare the coordinates with the image and return the log-likelihood. This function is used in the likelihood calculation.

    Parameters
    ----------
    coords : Array
        Coordinates of the model
    image_ref : Array
        Image to compare with
    struct_info : Array
        Structural information of the model.
    box_size : int
        Size of the box
    pixel_size : float
        Pixel size
    var_imaging_args : Array
        Imaging parameters

    Returns
    -------
    float
        Log-likelihood
    """
    image_coords = simulator_(
        coords, struct_info, grid, grid_f, pose_params, ctf_params
    )

    return -0.5 * jnp.linalg.norm(image_coords - image_ref) ** 2 / noise_var


batch_over_models_ = jax.jit(
    jax.vmap(
        compare_coords_with_img_,
        in_axes=(0, None, None, None, None, None, None, None),
    )
)

batch_over_images_ = jax.jit(
    jax.vmap(batch_over_models_, in_axes=(None, 0, None, None, None, 0, 0, 0))
)


def calc_lklhood_(
    models: Array,
    model_weights: Array,
    images: Array,
    struct_info: Array,
    grid: Array,
    grid_f: Array,
    pose_params: Array,
    ctf_params: Array,
    noise_var: Array,
) -> float:
    lklhood_matrix = batch_over_images_(
        models,
        images,
        struct_info,
        grid,
        grid_f,
        pose_params,
        ctf_params,
        noise_var,
    )

    model_weights = model_weights

    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=model_weights[None, :], axis=1
    )

    log_lklhood = jnp.mean(log_lklhood)  # / log_lklhood.shape[0]

    return log_lklhood


calc_lkl_and_grad_struct_ = jax.jit(jax.value_and_grad(calc_lklhood_, argnums=0))

calc_lkl_and_grad_wts_ = jax.jit(jax.value_and_grad(calc_lklhood_, argnums=1))


def calc_lkl_and_grad_struct(
    models: Array,
    model_weights: Array,
    image_stack: NumpyLoader,
    struct_info: Array,
    config: dict,
) -> Tuple[float, Array]:
    """
    Calculate the log-likelihood and its gradient with respect to the structure.

    Parameters
    ----------
    models : Array
        Models to compare with
    model_weights : Array
        Weights of the models
    image_stack : NumpyLoader
        Image stack
    struct_info : Array
        Structural information of the models.

    Returns
    -------
    float
        Log-likelihood
    Array
        Gradient with respect to the structure
    """

    for image_batch in image_stack:
        logging.debug(
            f"Calculating likelihood and gradient for batch {image_batch['idx']}"
        )
        log_lklhood, grad_str = calc_lkl_and_grad_struct_(
            models,
            model_weights,
            image_batch["proj"],
            struct_info,
            image_stack.dataset.proj_grid,
            image_stack.dataset.ctf_grid,
            image_batch["pose_params"],
            image_batch["ctf_params"],
            image_batch["noise_var"],
        )

        break

    return log_lklhood, grad_str


def calc_lkl_and_grad_wts(
    models: Array,
    model_weights: Array,
    image_stack: NumpyLoader,
    struct_info: Array,
    config: dict,
) -> Tuple[float, Array]:
    """
    Calculate the log-likelihood and its gradient with respect to the model weights.

    Parameters
    ----------
    models : Array
        Models to compare with
    model_weights : Array
        Weights of the models
    image_stack : NumpyLoader
        Image stack
    struct_info : Array
        Structural information of the models.

    Returns
    -------
    float
        Log-likelihood
    Array
        Gradient with respect to the model weights
    """

    log_lklhood = 0.0
    grad_wts = jnp.zeros_like(model_weights)

    for image_batch in image_stack:
        log_lklhood, grad_wts = calc_lkl_and_grad_wts_(
            models,
            model_weights,
            image_batch["proj"],
            struct_info,
            image_stack.dataset.proj_grid,
            image_stack.dataset.ctf_grid,
            image_batch["pose_params"],
            image_batch["ctf_params"],
            image_batch["noise_var"],
        )

        break

    return log_lklhood, grad_wts
