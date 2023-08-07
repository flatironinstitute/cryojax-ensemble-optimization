"""
Rotation functions.

Functions
---------
gen_quat
    Generate a random quaternion.
calc_rot_matrix
    Calculate the rotation matrix from a quaternion
"""

import numpy as np
import jax.numpy as jnp
import jax
from scipy.spatial.transform import Rotation


def gen_quat(n_quats, dtype: float) -> np.ndarray:
    """
    Generate a random quaternion.

    Returns:
        quat (np.ndarray): Random quaternion

    """

    np.random.seed(1234)

    quats = np.empty((n_quats, 4), dtype=dtype)

    count = 0
    while count < n_quats:
        quat = np.random.uniform(
            -1, 1, 4
        )  # note this is a half-open interval, so 1 is not included but -1 is
        norm = np.sqrt(np.sum(quat**2))

        if 0.2 <= norm <= 1.0:
            quat /= norm
            quats[count] = quat
            count += 1

    return jnp.array(quats)


@jax.jit
def calc_rot_matrix(quat: jnp.array):
    """
    Calculate the rotation matrix from a quaternion.

    Parameters
    ----------
    quat : jnp.array
        Quaternion

    Returns
    -------
    rot_mat : jnp.array
        Rotation matrix
    """
    rot_mat = jnp.array(
        [
            [
                1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
                2.0 * (quat[0] * quat[1] - quat[2] * quat[3]),
                2.0 * (quat[0] * quat[2] + quat[1] * quat[3]),
            ],
            [
                2.0 * (quat[0] * quat[1] + quat[2] * quat[3]),
                1.0 - 2.0 * (quat[0] ** 2 + quat[2] ** 2),
                2.0 * (quat[1] * quat[2] - quat[0] * quat[3]),
            ],
            [
                2.0 * (quat[0] * quat[2] - quat[1] * quat[3]),
                2.0 * (quat[1] * quat[2] + quat[0] * quat[3]),
                1.0 - 2.0 * (quat[0] ** 2 + quat[1] ** 2),
            ],
        ]
    )

    return rot_mat
