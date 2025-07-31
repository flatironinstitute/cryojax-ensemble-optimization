import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as Rsp

from cryojax_ensemble_optimization.ensemble_optimization import (
    global_SO3_hier_search,
)


def gaussian_basis(grid, center, sigma=0.2):
    diff = grid - center
    dist_sq = jnp.sum(diff**2, axis=-1)
    return jnp.exp(-dist_sq / (2 * sigma**2))


def evaluate_gaussians_on_grid(centers, grid, sigma=0.2):
    vmapped = jax.vmap(
        jax.vmap(lambda c: gaussian_basis(grid, c, sigma), in_axes=0),  # over points
        in_axes=0,  # over batch
    )
    return vmapped(centers)


def batch_rotate_points(points, rotations):
    # points: (np, 3)
    # rotations: (nr, 3, 3)
    # Returns: (nr, np, 3)
    return jax.vmap(lambda R: points @ R)(rotations)


@pytest.mark.parametrize(
    "num_points, grid_size, n_quaternions, sigma",
    [
        (4, 20, 7, 0.2),
    ],
)
def test_global_SO3_hier_search(num_points, grid_size, n_quaternions, sigma):
    key = jax.random.PRNGKey(0)
    points = jax.random.uniform(key, (num_points, 3), minval=-1.0, maxval=1.0)

    x = jnp.linspace(-1.5, 1.5, grid_size)
    y = jnp.linspace(-1.5, 1.5, grid_size)
    z = jnp.linspace(-1.5, 1.5, grid_size)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid = jnp.stack([X, Y, Z], axis=-1)  # shape: (grid_size, grid_size, grid_size, 3)

    np.random.seed(42)
    quaternion_gt = Rsp.random().as_quat()  # shape: (4,)
    rotation_gt = R.from_quat(quaternion_gt).as_matrix()  # shape: (3, 3)
    points_gt = points @ rotation_gt
    values_gt = jnp.sum(
        jax.vmap(lambda c: gaussian_basis(grid, c, sigma))(points_gt), axis=0
    )

    def lossfn(quaternions):
        n_quaternions = len(quaternions)
        rotations = R.from_quat(quaternions).as_matrix()
        rotated_points = batch_rotate_points(points, rotations)
        values_est = jnp.sum(
            evaluate_gaussians_on_grid(rotated_points, grid, sigma=sigma), axis=1
        )
        return jnp.linalg.norm(
            (values_est - values_gt).reshape(n_quaternions, -1), axis=1
        )

    quaternions = Rsp.random(n_quaternions).as_quat()
    _ = lossfn(quaternions)  # Just to check it runs

    best_quats, _ = global_SO3_hier_search(
        lossfn, base_grid=1, n_rounds=5, N_candidates=40
    )

    assert np.allclose(best_quats, quaternion_gt, atol=0.1) or np.allclose(
        best_quats, -quaternion_gt, atol=0.1
    ), f"Expected {quaternion_gt} or {-quaternion_gt}, got {best_quats}"
