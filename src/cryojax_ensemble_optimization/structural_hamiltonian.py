from dataclasses import dataclass
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from cryojax.io import read_atoms_from_pdb
from jax import jit
from omegaconf import OmegaConf


Array = jnp.ndarray

repo_root = Path(
    __file__
).parent.parent  # Navigate to repo root (assuming this script is under /src)
config_path = repo_root / "config_files"  # Path to the configs folder
output_dir = repo_root / "outputs"  # Path to outputs directory


@dataclass
class OptimizationConfig:
    learnign_rate: float
    num_steps: int


@dataclass
class Spring:
    equilibrium_distance: float
    constant: float | int


@dataclass
class SoftSphere:
    constant: float
    sigma: float
    epsilon: float
    alpha: float


@dataclass
class StructuralHamiltonian:
    dot_bracket: str
    pdb_fname: str
    mdtraj_select: str
    spring: Spring
    soft_sphere: SoftSphere


def dot_bracket_to_base_pair_indices(dot_bracket: str) -> Array:
    """Convert dot-bracket notation to base pair indices.
    Args:
        dot_bracket (str): Dot-bracket notation string.
    Returns:
        jnp.ndarray: Array of shape (n_pairs, 2) containing indices of base pairs.
    """
    # Initialize an empty stack and a list to store pairs
    stack = []
    pairs = []

    # Traverse the dot-bracket notation
    for i, char in enumerate(dot_bracket):
        if char == "(":
            # Push the index of '(' onto the stack
            stack.append(i)
        elif char == ")":
            # Pop from the stack to get the matching '(' index
            start_index = stack.pop()
            # Store the pair (start_index, end_index)
            pairs.append((start_index, i))

    return jnp.array(pairs)


def pairwise_distance_energy(
    pairs: Array, coords: Array, equilibrium_distance: float
) -> Array:
    """Calculate the spring energy based on pairwise distances.
    Args:
        pairs (jnp.ndarray): shape=(n_pairs, 2) containing indices of atom pairs.
        coords (jnp.ndarray): shape=(n_atoms, 3) containing the coordinates of atoms.
        equilibrium_distance (float): The equilibrium distance for the spring potential.
    Returns:
        jnp.ndarray: The total spring energy.
    """
    ri = coords[pairs[:, 0]]
    rj = coords[pairs[:, 1]]

    distances = jnp.linalg.norm(ri - rj, axis=1)

    spring_energy = jnp.sum((distances - equilibrium_distance) ** 2)

    return spring_energy, distances


def upside_down_gaussian_energy(
    pairs: Array, coords: Array, eq_dist: float, sigma: float
) -> Array:
    """Calculate the upside-down Gaussian energy based on pairwise distances.
    Args:
        pairs (jnp.ndarray): shape=(n_pairs, 2) containing indices of atom pairs.
        coords (jnp.ndarray): shape=(n_atoms, 3) containing the coordinates of atoms.
        eq_dist (float): The equilibrium distance for the Gaussian potential.
        sigma (float): The width of the Gaussian.
    Returns:
        jnp.ndarray: The total Gaussian well energy.
    """
    # Calculate the pairwise distances
    ri = coords[pairs[:, 0]]
    rj = coords[pairs[:, 1]]
    distances = jnp.linalg.norm(ri - rj, axis=1)
    diff = distances - eq_dist
    gaussian_well = -jnp.exp(-0.5 * (diff / sigma) ** 2)
    return jnp.sum(gaussian_well)


def optimize_equilibrium_distance(pairs, coords, init_eq_dist, learning_rate, num_steps):
    """Optimize the equilibrium distance using gradient descent."""

    def loss_fn(eq_dist):
        spring_energy, _ = pairwise_distance_energy(pairs, coords, eq_dist)
        total_energy = spring_energy
        return total_energy

    # Initialize
    eq_dist = init_eq_dist

    # Gradient function
    grad_fn = jax.grad(loss_fn)

    # Optimization loop
    losses, vals = [], []
    for step in range(num_steps):
        loss_val = loss_fn(eq_dist)
        losses.append(loss_val)
        vals.append(eq_dist)

        # Compute gradient
        grad = grad_fn(eq_dist)

        # Update equilibrium distance
        eq_dist -= learning_rate * grad

        if step % 10 == 0:
            loss_val = loss_fn(eq_dist)
            print(f"Step {step}: eq_dist = {eq_dist:.4f}, loss = {loss_val:.4f}")

    return eq_dist, jnp.array(losses), jnp.array(vals)


def optimize_coords(
    pairs,
    init_coords,
    eq_dist,
    sigma,
    epsilon,
    alpha,
    spring_constant,
    soft_sphere_constant,
    learning_rate,
    num_steps,
):
    """Optimize the equilibrium distance using gradient descent."""

    def loss_fn(coords):
        spring_energy, distances = pairwise_distance_energy(pairs, coords, eq_dist)
        soft_sphere_energy = soft_sphere(distances, sigma, epsilon, alpha)
        total_energy = (
            spring_constant * spring_energy + soft_sphere_constant * soft_sphere_energy
        )
        return total_energy

    # Initialize
    coords = init_coords

    # Gradient function
    grad_fn = jax.grad(loss_fn)

    # Optimization loop
    losses, vals = [], []
    for step in range(num_steps):
        loss_val = loss_fn(coords)
        losses.append(loss_val)
        vals.append(eq_dist)

        # Compute gradient
        grad = grad_fn(coords)

        # Update equilibrium distance
        coords -= learning_rate * grad

        if step % 10 == 0:
            loss_val = loss_fn(coords)
            print(f"Step {step}: eq_dist = {eq_dist:.4f}, loss = {loss_val:.4f}")

    return coords, jnp.array(losses)


@partial(jit, static_argnums=(1,))
def safe_mask(mask, fn, operand, placeholder=0):
    masked = jnp.where(mask, operand, 0)
    return jnp.where(mask, fn(masked), placeholder)


def soft_sphere(
    dr: Array,
    sigma: Array = 1,
    epsilon: Array = 1,
    alpha: Array = 2,
) -> Array:
    """Finite ranged repulsive interaction between soft spheres from https://jax-md.readthedocs.io/en/main/_modules/jax_md/energy.html#soft_sphere

    Args:
    dr: An ndarray of shape `[n, m]` of pairwise distances between particles.
    sigma: Particle diameter. Should either be a floating point scalar or an
        ndarray whose shape is `[n, m]`.
    epsilon: Interaction energy scale. Should either be a floating point scalar
        or an ndarray whose shape is `[n, m]`.
    alpha: Exponent specifying interaction stiffness. Should either be a float
        point scalar or an ndarray whose shape is `[n, m]`.
    Returns:
    Matrix of energies whose shape is `[n, m]`.
    """
    f32 = jnp.float32
    dr = dr / sigma
    fn = lambda dr: epsilon / alpha * (f32(1.0) - dr) ** alpha

    if (
        isinstance(alpha, float)
        or isinstance(alpha, int)
        or issubclass(type(alpha.dtype), jnp.integer)
    ):
        return jnp.where(dr < 1.0, fn(dr), f32(0.0)).sum()

    return safe_mask(dr < 1.0, fn, dr, f32(0.0)).sum()


@hydra.main(config_path=str(config_path), config_name="config_structural_hamiltonian")
def main(cfg: StructuralHamiltonian):
    """Main function to run the structural Hamiltonian calculations.

    Read in dot-bracket notation and PDB file, calculate the spring energy based on
    pairwise distances, and optimize the equilibrium distance using gradient descent.
    """
    config = OmegaConf.to_container(cfg, resolve=True)
    base_pair_indices = dot_bracket_to_base_pair_indices(config["dot_bracket"])
    print(config["mdtraj_select"])
    atom_positions, _, _ = read_atoms_from_pdb(
        config["pdb_fname"],
        center=True,
        loads_b_factors=True,
        select=config["mdtraj_select"],
    )
    print(f"Atom positions: {atom_positions[:10]}")
    spring_energy, distances = pairwise_distance_energy(
        base_pair_indices, atom_positions, config["spring"]["equilibrium_distance"]
    )
    print(f"Spring energy: {spring_energy:.4f}")
    clash_energy = soft_sphere(
        distances,
        sigma=config["soft_sphere"]["sigma"],
        epsilon=config["soft_sphere"]["epsilon"],
        alpha=config["soft_sphere"]["alpha"],
    )
    print(f"Clash energy: {clash_energy:.4f}")

    final_eq_dist, _, _ = optimize_equilibrium_distance(
        base_pair_indices,
        atom_positions,
        init_eq_dist=3.0,
        learning_rate=config["optimization"]["learning_rate"],
        num_steps=config["optimization"]["num_steps"],
    )
    print(f"Final equilibrium distance: {final_eq_dist:.4f}")

    final_atom_positions, _ = optimize_coords(
        base_pair_indices,
        atom_positions,
        eq_dist=config["spring"]["equilibrium_distance"],
        sigma=config["soft_sphere"]["sigma"],
        epsilon=config["soft_sphere"]["epsilon"],
        alpha=config["soft_sphere"]["alpha"],
        spring_constant=config["spring"]["constant"],
        soft_sphere_constant=config["soft_sphere"]["constant"],
        learning_rate=config["optimization"]["learning_rate"],
        num_steps=config["optimization"]["num_steps"],
    )

    print(f"Final equilibrium distance: {final_atom_positions[:10]}")
    rmsd = jnp.sqrt(jnp.mean((atom_positions - final_atom_positions) ** 2))
    print(f"RMSD: {rmsd:.4f}")


if __name__ == "__main__":
    main()
