import numpy as np
import jax.numpy as jnp
import jax
from scipy.spatial.transform import Rotation


def gen_quat() -> np.ndarray:
    """
    Generate a random quaternion.

    Returns:
        quat (np.ndarray): Random quaternion

    """
    count = 0
    while count < 1:
        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quat /= norm
            count += 1

    return quat


@jax.jit
def rotate_struct(coords: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Use scipy's Rotation class to generate a rotation matrix from quaternions, and rotate a structure.

    Parameters:
    coords : np.ndarray
        Array with atomic coordinates, shape must be (3, number_of_atoms)

    quat: np.ndarray
        Array with quaternions defining a rotation, with the convention (x, y, z, w).

    Returns:
    rot_coords: np.ndarray
        Array with the rotated atomic coordinates
    """

    rot_coords = jnp.matmul(Rotation.from_quat(quat).as_matrix(), coords)

    return rot_coords
