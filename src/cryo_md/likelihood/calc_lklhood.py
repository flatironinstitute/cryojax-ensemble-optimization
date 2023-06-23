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
    image_coords = noiseless_simulator_(
        coords, box_size, pixel_size, sigma, var_imaging_args
    )

    return -0.5 * jnp.linalg.norm(image_coords - image_ref)


def compare_dummy(
    coords: ArrayLike,
) -> ArrayLike:
    return jnp.array(0.0)


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def calc_loglkl_model_image(
    coords: ArrayLike,
    image_ref: ArrayLike,
    box_size: int,
    pixel_size: float,
    sigma: float,
    var_imaging_args: ArrayLike,
    is_neigh: Union[int, float, ArrayLike],
) -> ArrayLike:
    cond_comp = partial(
        compare_coords_with_img_,
        image_ref=image_ref,
        box_size=box_size,
        pixel_size=pixel_size,
        sigma=sigma,
        var_imaging_args=var_imaging_args,
    )

    log_lkl = jax.lax.cond(jnp.equal(is_neigh, 0.0), compare_dummy, cond_comp, coords)

    return log_lkl


batch_over_models = jax.jit(
    jax.vmap(calc_loglkl_model_image, in_axes=(0, None, None, None, None, None, 0)),
    static_argnums=(2, 3),
)

batch_over_images = jax.jit(
    jax.vmap(batch_over_models, in_axes=(None, 0, None, None, None, 0, 0)),
    static_argnums=(2, 3),
)


def get_neigh_list(models, data_stack):
    lklhood_matrix = batch_over_images(
        models,
        data_stack.images,
        data_stack.constant_params[0],
        data_stack.constant_params[1],
        data_stack.constant_params[2],
        data_stack.variable_params,
        jnp.ones((data_stack.images.shape[0], models.shape[0])),
    )

    return (lklhood_matrix == jnp.max(lklhood_matrix, axis=1)[:, None]).astype(float)


@partial(jax.jit, static_argnames=["pixel_size", "box_size"])
def calc_lklhood_(
    models: ArrayLike,
    model_weights: ArrayLike,
    images: ArrayLike,
    box_size: int,
    pixel_size: float,
    sigma: float,
    variable_params: ArrayLike,
    noise_var: float,
    neigh_list: ArrayLike,
) -> float:
    lklhood_matrix = (
        batch_over_images(
            models, images, box_size, pixel_size, sigma, variable_params, neigh_list
        )
        / noise_var
    )

    model_weights = model_weights * (noise_var * 2 * jnp.pi) ** (0.5)

    log_lklhood = jax.scipy.special.logsumexp(
        a=lklhood_matrix, b=model_weights[None, :] * neigh_list, axis=1
    )

    log_lklhood = jnp.sum(log_lklhood)

    return log_lklhood


calc_lkl_and_grad = jax.jit(
    jax.value_and_grad(calc_lklhood_, argnums=(0, 1)), static_argnums=(3, 4)
)
