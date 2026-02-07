import jax.numpy as jnp
from jaxtyping import Array, Float


def rigid_align_positions(
    target_pos: Float[Array, "n_atoms 3"], ref_pos: Float[Array, "n_atoms 3"]
) -> tuple[Float[Array, "n_atoms 3"], Float[Array, "3 3"], Float[Array, "1 3"]]:
    com_ref = jnp.mean(ref_pos, axis=0, keepdims=True)
    com_target = jnp.mean(target_pos, axis=0, keepdims=True)

    cross_cov_matrix = jnp.dot((ref_pos - com_ref).T, target_pos - com_target)

    U, _, Vh = jnp.linalg.svd(cross_cov_matrix)
    det = jnp.linalg.det(U) * jnp.linalg.det(Vh)
    rot_matrix = U @ jnp.diag(jnp.array([1.0, 1.0, det])) @ Vh

    displacement = com_ref - com_target @ rot_matrix.T

    aligned_pos = target_pos @ rot_matrix.T + displacement
    return aligned_pos, rot_matrix, displacement
