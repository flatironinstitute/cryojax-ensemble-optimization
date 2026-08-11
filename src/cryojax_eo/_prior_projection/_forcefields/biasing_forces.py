from jaxtyping import Array, Float


def compute_harmonic_steering_force(
    positions: Float[Array, "n_atoms 3"],
    reference_positions: Float[Array, "n_atoms 3"],
    force_constant: Float,
):
    return -force_constant * (positions - reference_positions)
