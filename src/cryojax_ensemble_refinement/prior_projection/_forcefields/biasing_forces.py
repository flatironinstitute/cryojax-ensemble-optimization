from jaxtyping import Array, Float
import jax.numpy as jnp

def compute_harmonic_bias_potential_energy(
    atom_positions: Float[Array, "n_atoms 3"],
    reference_atom_positions: Float[Array, "n_atoms 3"],
    force_constant: Float,
):
    return (
        -0.5
        * force_constant
        * jnp.sum((atom_positions - reference_atom_positions) ** 2)
    )
