from typing import TypedDict

import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from .base_forcefield import AbstractForceField


DEFAULT_POLYMER_PARAMS = dict(
    force_constant=1.0,
    equilibrium_distance=6.0,
)


class PolymerEnergyParams(TypedDict):
    force_constant: float
    equilibrium_distance: float


DEFAULT_SPRING_PARAMS = dict(
    force_constant=1.0,
    equilibrium_distance=11.08,
)


class SpringEnergyParams(TypedDict):
    force_constant: float
    equilibrium_distance: float


DEFAULT_SOFT_SPHERE_PARAMS = dict(
    soft_sphere_constant=1.0,
    particle_diameter=1.0,
    interaction_energy_scale=1.0,
    interaction_stiffness=2.0,
)


class SoftSphereEnergyParams(TypedDict):
    force_constant: float
    particle_diameter: float | Float[Array, "n m"]
    interaction_energy_scale: float | Float[Array, "n m"]
    interaction_stiffness: float | Float[Array, "n m"]


class RNAForceField(AbstractForceField):
    def __init__(
        self,
        bond_pair_indices: Int[Array, "n_pairs 2"],
        polymer_energy_params: PolymerEnergyParams = DEFAULT_POLYMER_PARAMS,
        spring_energy_params: SpringEnergyParams = DEFAULT_SPRING_PARAMS,
        soft_sphere_energy_params: SoftSphereEnergyParams = DEFAULT_SOFT_SPHERE_PARAMS,
    ):
        self.energy_fn = _compute_rna_energy

        spring_energy_params = DEFAULT_SPRING_PARAMS.copy()
        spring_energy_params.update(spring_energy_params)

        soft_sphere_energy_params = DEFAULT_SOFT_SPHERE_PARAMS.copy()
        soft_sphere_energy_params.update(soft_sphere_energy_params)

        polymer_energy_params = DEFAULT_POLYMER_PARAMS.copy()
        polymer_energy_params.update(polymer_energy_params)

        self.energy_fn_args = (
            bond_pair_indices,
            spring_energy_params,
            soft_sphere_energy_params,
            polymer_energy_params,
        )

    def __call__(self, coordinates: Float[Array, "n_atoms 3"]) -> float:
        return self.energy_fn(coordinates, *self.energy_fn_args)


def _compute_rna_energy(
    positions: Float[Array, "n_atoms 3"],
    bond_pair_indices: Int[Array, "n_pairs 2"],
    spring_energy_params: SpringEnergyParams,
    soft_sphere_energy_params: SoftSphereEnergyParams,
    polymer_energy_params: PolymerEnergyParams,
) -> float:
    distances = _compute_pairwise_distances(positions, bond_pair_indices)
    spring_energy = spring_energy_params["force_constant"] * _pairwise_distance_energy(
        distances, spring_energy_params["equilibrium_distance"]
    )
    soft_sphere_energy = soft_sphere_energy_params[
        "soft_sphere_constant"
    ] * _compute_soft_sphere_energy(
        distances,
        particle_diameter=soft_sphere_energy_params["particle_diameter"],
        interaction_energy_scale=soft_sphere_energy_params["interaction_energy_scale"],
        interaction_stiffness=soft_sphere_energy_params["interaction_stiffness"],
    )
    polymer_energy = polymer_energy_params["force_constant"] * _polymer_distance_energy(
        positions,
        equilibrium_distance=polymer_energy_params["equilibrium_distance"],
    )
    return spring_energy + soft_sphere_energy + polymer_energy


def _compute_pairwise_distances(
    positions: Float[Array, "n_atoms 3"], bond_pair_indices: Int[Array, "n_pairs 2"]
) -> Float[Array, " n_pairs"]:
    ri = positions[bond_pair_indices[:, 0]]
    rj = positions[bond_pair_indices[:, 1]]

    return jnp.linalg.norm(ri - rj, axis=1)


def _pairwise_distance_energy(
    distances: Float[Array, " n_pairs"],
    equilibrium_distance: float,
) -> float:
    """Calculate the spring energy based on pairwise distances.
    Args:
        bond_pair_indices (jnp.ndarray): shape=(n_pairs, 2) containing
            indices of atom bond_pair_indices.
        positions (jnp.ndarray): shape=(n_atoms, 3) containing the coordinates of atoms.
        equilibrium_distance (float): The equilibrium distance for the spring potential.
    Returns:
        float: The total spring energy.
    """
    return jnp.sum((distances - equilibrium_distance) ** 2)


def _compute_soft_sphere_energy(
    pairwise_distances: Float[Array, " n_pairs"],
    particle_diameter: float | Float[Array, " n_pairs"] = 1.0,
    interaction_energy_scale: float | Float[Array, " n_pairs"] = 1.0,
    interaction_stiffness: float | Float[Array, " n_pairs"] = 2.0,
) -> float:
    """Finite ranged repulsive interaction between soft spheres from
    https://jax-md.readthedocs.io/en/main/_modules/jax_md/energy.html#soft_sphere

    Args:
    pairwise_distances: Array of shape `[n_pairs]` of pairwise distances between particles
    particle_diameter: Particle diameter. Should either be a floating point scalar or an
        ndarray whose shape is `[n, m]`.
    interaction_energy_scale: Interaction energy scale.
        Should either be a floating point scalar or an ndarray whose shape is `[n, m]`.
    interaction_stiffness: Exponent specifying interaction stiffness.
        Should either be a floating point scalar or an ndarray whose shape is `[n, m]`.

    Returns:
    Matrix of energies whose shape is `[n, m]`.
    """
    pairwise_distances = pairwise_distances / particle_diameter

    return jnp.where(
        pairwise_distances < 1.0,
        interaction_energy_scale
        / interaction_stiffness
        * (1.0 - pairwise_distances) ** interaction_stiffness,
        0.0,
    ).sum()


def _polymer_distance_energy(
    positions: Float[Array, "n_atoms 3"],
    equilibrium_distance: float,
    force_constant: float = 1.0,
) -> float:
    distances = jnp.linalg.norm(positions[1:] - positions[:-1], axis=1)

    return force_constant * jnp.sum((distances - equilibrium_distance) ** 2)
