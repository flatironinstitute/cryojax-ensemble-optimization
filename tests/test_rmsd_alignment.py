import jax.numpy as jnp
import jax.random as jr
import pytest
from cryojax.rotations import SO3

from cryojax_eo.utils import rigid_align_positions


@pytest.mark.parametrize("n_atoms", [1000, 10000, 20000])
def test_rigid_align_positions(n_atoms):
    key = jr.key(0)
    key_rot, key_trans, key_points = jr.split(key, 3)

    # Rantom transformation
    random_rotation = SO3.sample_uniform(key_rot).as_matrix()
    random_translation = jr.normal(key_trans, (3,)) * 10.0

    random_points = jr.normal(key_points, (n_atoms, 3)) * 10.0
    random_points -= jnp.mean(random_points, axis=0, keepdims=True)  # Center the points

    transformed_points = (random_points + random_translation) @ random_rotation.T

    aligned_points, rot_matrix, displacement = rigid_align_positions(
        transformed_points, random_points
    )

    print(f"Test with {n_atoms} atoms:")
    print(random_translation, displacement)
    assert jnp.allclose(
        aligned_points, random_points, atol=1e-5
    ), "Aligned points do not match reference points"
    assert jnp.allclose(
        rot_matrix @ random_rotation, jnp.eye(3), atol=1e-5
    ), "Rotation matrices do not match"
    assert jnp.allclose(displacement, -random_translation), "Translations do not match"
