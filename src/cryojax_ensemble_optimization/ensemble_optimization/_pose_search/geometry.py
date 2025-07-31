"""
Geometry related codes
The SO(3) grid part codes are adpated from deeprefine
(https://github.com/minhuanli/deeprefine/blob/master/deeprefine/geometry.py)
based on Hopf Fibration
All jax-ified functions have a `_jnp` suffix, other functions are numpy based
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Int

from .healjax import pix2ang


def construct_SO3_jnp(v1, v2):
    """
    Construct a continuous representation of SO(3) rotation with two 3D vectors
    https://arxiv.org/abs/1812.07035
    Parameters
    ----------
    v1, v2: 3D arrays
        Real-valued array in 3D space
    Returns
    -------
    R: A 3x3 SO(3) rotation matrix
    """
    e1 = v1 / jnp.linalg.norm(v1)
    u2 = v2 - e1 * jnp.tensordot(e1, v2, axes=1)
    e2 = u2 / jnp.linalg.norm(u2)
    e3 = jnp.cross(e1, e2)
    R = jnp.stack((e1, e2, e3)).T
    return R


def decompose_SO3(R, a=1, b=1, c=1):
    """
    Decompose the rotation matrix into the two vector representation
    This decomposition is not unique, so a, b, c can be set as
    arbitrary constants you like
    c != 0
    Parameters
    ----------
    R: 3x3 array
        Real-valued rotation matrix
    a, b, c: scalar constants
        Arbitrary constants for the decomposition (c must be nonzero)
    Returns
    -------
    v1, v2: Two real-valued 3D arrays, as the continuous
    representation of the rotation matrix
    """
    assert c != 0, "Give a nonzero c!"
    v1 = a * R[:, 0]
    v2 = b * R[:, 0] + c * R[:, 1]

    return v1, v2


# Quaternion representation of SO(3) and Hopf Fibration grid
def grid_s1(resol=1):
    Npix = 6 * 2**resol
    dt = 2 * np.pi / Npix
    grid = jnp.arange(Npix) * dt + dt / 2
    return grid


def grid_s2(resol=1):
    Nside = 2**resol
    Npix = 12 * Nside * Nside
    theta, phi = pix2ang(Nside, np.arange(Npix))
    return theta, phi


@partial(jax.vmap, in_axes=(0, 0, None))
@partial(jax.vmap, in_axes=(None, None, 0))
def _hopf_to_quat(theta, phi, psi):
    """
    Hopf coordinates to quaternions
    theta: [0,pi]
    phi: [0, 2pi)
    psi: [0, 2pi)
    already normalized
    """
    ct = jnp.cos(theta / 2)
    st = jnp.sin(theta / 2)
    quat = jnp.array(
        [
            ct * jnp.cos(psi / 2),
            ct * jnp.sin(psi / 2),
            st * jnp.cos(phi + psi / 2),
            st * jnp.sin(phi + psi / 2),
        ]
    )
    return quat.T.astype(np.float32)


def hopf_to_quat(theta, phi, psi):
    return _hopf_to_quat(theta, phi, psi).reshape(-1, 4)


def grid_SO3(resol):
    theta, phi = grid_s2(resol)
    psi = grid_s1(resol)
    quat = hopf_to_quat(theta, phi, psi).reshape(-1, 4)
    return quat


def quat_distance(q1, q2):
    """
    q1: [n1, 4]
    q2: [n2, 4]

    Return:
        [n1, n2]
    """
    q1 = q1 / np.linalg.norm(q1, ord=2, axis=-1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, ord=2, axis=-1, keepdims=True)
    args = np.abs(np.sum(q1[:, None, :] * q2[None, ...], axis=-1))
    return 2.0 * np.arccos(args)


def mat_distance(R1, R2):
    """
    R1: [a, 3, 3]
    R2: [b, 3, 3]

    Return:
        [a, b]
    """
    R = np.einsum("axy,bzy->abxz", R1, R2)
    args = (np.einsum("abii", R) - 1) / 2.0
    return np.arccos(args)


def quaternions_to_SO3_jnp(q):
    """
    Normalizes q and maps to group matrix.
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix
    https://danceswithcode.net/engineeringnotes/quaternions/quaternions.html

    Parameters
    ----------
    q: JAX array of shape (..., 4)
        Quaternion(s) in format [w, x, y, z] where w is real part

    Returns
    -------
    R: JAX array of shape (..., 3, 3)
        SO(3) rotation matrix/matrices
    """
    # Normalize quaternion
    q = q / jnp.linalg.norm(q, axis=-1, keepdims=True)
    r, i, j, k = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Stack rotation matrix elements
    rotation_elements = jnp.stack(
        [
            1 - 2 * j * j - 2 * k * k,
            2 * (i * j - r * k),
            2 * (i * k + r * j),
            2 * (i * j + r * k),
            1 - 2 * i * i - 2 * k * k,
            2 * (j * k - r * i),
            2 * (i * k - r * j),
            2 * (j * k + r * i),
            1 - 2 * i * i - 2 * j * j,
        ],
        axis=-1,
    )

    # Reshape to 3x3 matrices
    return rotation_elements.reshape(*q.shape[:-1], 3, 3)


def SO3_to_quaternions_jnp(mats, key=None):
    """
    Convert SO(3) rotation matrices to quaternions.

    Parameters
    ----------
    mats: JAX array of shape (n, 3, 3)
        Rotation matrices
    key: JAX random key (optional)
        Random key for reproducible random vector generation
        If None, uses a fixed seed

    Returns
    -------
    qs: JAX array of shape (n, 4)
        Quaternions in format [w, x, y, z]
    """
    if key is None:
        key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility

    # Compute eigenvalues and eigenvectors
    w, v = jnp.linalg.eig(mats)

    # Find indices where eigenvalue is close to 1 (rotation axis)
    # For real rotation matrices, one eigenvalue should be 1
    closest_to_one = jnp.argmin(jnp.abs(w - 1.0), axis=-1)
    batch_indices = jnp.arange(mats.shape[0])
    us = jnp.real(v[batch_indices, :, closest_to_one]).T

    # Generate random vector and compute orthogonal vector
    # take a random vector v orthogonal to u, determine sign of sin(theta/2)
    # https://math.stackexchange.com/questions/893984/conversion-of-rotation-matrix-to-quaternion
    v_rand = jax.random.normal(key, (3,))
    vs = jnp.cross(us, v_rand[None, :])
    vs = vs / jnp.linalg.norm(vs, axis=-1, keepdims=True)  # Normalize

    # Determine sign of sin(theta/2)
    cross_product = jnp.cross(vs, jnp.einsum("nij,nj->ni", mats, vs))
    signs = jnp.sign(jnp.einsum("ni,ni->n", cross_product, us))

    # Compute rotation angles
    traces = jnp.trace(mats, axis1=-2, axis2=-1)
    args = (traces - 1) / 2
    # Clamp to avoid numerical issues with arccos
    args = jnp.clip(args, -1.0, 1.0)
    thetas = jnp.arccos(args)

    # Compute quaternion components
    cos_halftheta = jnp.cos(thetas / 2.0)
    sin_halftheta = jnp.sin(thetas / 2.0) * signs

    # Construct quaternions [w, x, y, z]
    qs = jnp.concatenate([cos_halftheta[:, None], sin_halftheta[:, None] * us], axis=1)

    # Normalize quaternions
    qs = qs / jnp.linalg.norm(qs, axis=-1, keepdims=True)

    return qs


# Neighbors on the Hopf grid
def get_s1_neighbor(mini, curr_res):
    """
    Return the 2 nearest neighbors on S1 at the next resolution level
    """
    Npix = 6 * 2 ** (curr_res + 1)
    dt = 2 * np.pi / Npix
    # return np.array([2*mini, 2*mini+1])*dt + dt/2
    # the fiber bundle grid on SO3 is weird
    # the next resolution level's nearest neighbors in SO3 are not
    # necessarily the nearest neighbor grid points in S1
    # include the 13 neighbors for now... eventually learn/memorize the mapping
    ind = jnp.arange(0, 4, 1) + (2 * mini - 1)
    ind = jax.lax.cond(ind[0] < 0, lambda x: x.at[0].set(x[0] + Npix), lambda x: x, ind)

    return ind * dt + dt / 2, ind


def get_s2_neighbor(mini, curr_res):
    """
    Return the 4 nearest neighbors on S2 at the next resolution level
    """
    Nside = 2 ** (curr_res + 1)
    ind = jnp.arange(4) + 4 * mini
    return pix2ang(Nside, ind), ind


def get_base_ind(
    ind: Int[Array, " n_indices"], base_resol: Float = 1.0
) -> Tuple[Int[Array, " n_indices"], Int[Array, " n_indices"]]:
    """
    Return the corresponding S2 and S1 grid index for an index on the base SO3 grid
    """
    psii = ind % (6 * 2**base_resol)
    thetai = ind // (6 * 2**base_resol)
    return thetai, psii


@partial(jax.vmap, in_axes=(0, 0, 0, None))
def get_neighbor_SO3(quat, s2i, s1i, curr_res):
    """
    Return the 8 nearest neighbors on SO3 at the next resolution level
    """
    (theta, phi), s2_nexti = get_s2_neighbor(s2i, curr_res)
    psi, s1_nexti = get_s1_neighbor(s1i, curr_res)
    quat_n = hopf_to_quat(theta, phi, psi)
    ind = jnp.array(
        [
            jnp.repeat(s2_nexti, len(psi), total_repeat_length=16),
            jnp.tile(s1_nexti, len(theta)),
        ]
    )
    ind = ind.T
    # find the 8 nearest neighbors of 16 possible points
    # need to check distance from both +q and -q
    dists = jnp.minimum(
        jnp.sum((quat_n - quat) ** 2, axis=1), jnp.sum((quat_n + quat) ** 2, axis=1)
    )
    _, best_8_indices = jax.lax.top_k(-dists, k=8)

    return quat_n[best_8_indices], ind[best_8_indices]


# Loss based neighbor search
def getbestneighbors_base_SO3(loss, base_quats, N=10, base_resol=1):
    _, bestN_index = jax.lax.top_k(-loss, k=N)
    best_quats = base_quats[bestN_index]
    s2_index, s1_index = get_base_ind(bestN_index, base_resol)
    allnb_quats, allnb_s2s1 = get_neighbor_SO3(best_quats, s2_index, s1_index, base_resol)
    return allnb_quats.reshape(-1, 4), allnb_s2s1.reshape(-1, 2)


def getbestneighbors_next_SO3(loss, quats, s2s1_arr, curr_res=2, N=50):
    _, bestN_index = jax.lax.top_k(-loss, k=N)
    best_quats = quats[bestN_index]

    s2_index = s2s1_arr[bestN_index, 0].astype(int)
    s1_index = s2s1_arr[bestN_index, 1].astype(int)
    allnb_quats, allnb_s2s1 = get_neighbor_SO3(best_quats, s2_index, s1_index, curr_res)
    return allnb_quats.reshape(-1, 4), allnb_s2s1.reshape(-1, 2)
