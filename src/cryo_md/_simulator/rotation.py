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
from scipy.spatial.transform import Rotation


def gen_euler(n_quats, dtype: float) -> np.ndarray:
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

    quats = jnp.array(quats)
    euler_angs = Rotation.from_quat(quats).as_euler("ZYZ", degrees=False)

    return euler_angs


def calc_rot_matrix(angles):
    """
    Generates a rotation matrix from the Euler angles in the ZYZ convention.
    Equivalent to scipy.spatial.transform.Rotation.from_euler('ZYZ', angles).as_matrix()

    Parameters
    ----------
    angles : jnp.Tensor
        A tensor of shape (3,) containing the Euler angles in radians.

    Returns
    -------
    rot_matrix : jnp.Tensor
        A tensor of shape (3, 3) containing the rotation matrix.
    """
    alpha, beta, gamma = angles
    ca, cb, cg = jnp.cos(alpha), jnp.cos(beta), jnp.cos(gamma)
    sa, sb, sg = jnp.sin(alpha), jnp.sin(beta), jnp.sin(gamma)

    rot_matrix = jnp.array(
        [
            [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
            [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
            [-sb * cg, sb * sg, cb],
        ]
    )

    return rot_matrix.T
