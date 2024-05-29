import jax.numpy as jnp
import jax


def add_noise_(image, noise_grid, noise_radius_mask, noise_snr, random_key):
    radii_for_mask = noise_grid[None, :] ** 2 + noise_grid[:, None] ** 2
    mask = radii_for_mask < noise_radius_mask**2

    signal_power = jnp.sqrt(jnp.sum((image * mask) ** 2) / jnp.sum(mask))

    noise_power = signal_power / jnp.sqrt(noise_snr)
    image = image + jax.random.normal(random_key, shape=image.shape) * noise_power

    return image, noise_power**2


batch_add_noise_ = jax.vmap(add_noise_, in_axes=(0, None, None, 0, 0))
