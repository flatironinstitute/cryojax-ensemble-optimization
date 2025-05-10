from typing import TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .._forcefields.base_forcefield import (
    AbstractForceField,
)
from .._forcefields.biasing_forces import compute_harmonic_bias_potential_energy

EnergyFuncArgs = TypeVar("EnergyFuncArgs")


class AbstractLangevinSampler(eqx.Module, strict=True):
    """
    Abstract class for a sampler.
    """

    n_steps: eqx.AbstractVar[Int]
    step_size: eqx.AbstractVar[Float]
    forcefield: AbstractForceField


class SteeredLangevinSampler(AbstractLangevinSampler):
    """
    Langevin sampler for Langevin dynamics.
    """

    n_steps: Int
    step_size: Float
    forcefield: AbstractForceField
    biasing_force_constant: Float

    def __init__(
        self,
        n_steps: Int,
        step_size: Float,
        forcefield: AbstractForceField,
        biasing_force_constant: Float,
    ):
        """
        **Arguments:**
        - `n_steps`: Number of Langevin steps to take.
        - `step_size`: Step size for Langevin dynamics.
        - `forcefield`: Force field to use for Langevin dynamics.
        - `biasing_force_constant`: Force constant for the biasing potential.
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.forcefield = forcefield
        self.biasing_force_constant = biasing_force_constant

    def __call__(
        self,
        key: PRNGKeyArray,
        initial_atom_positions: Float[Array, "n_atoms 3"],
        ref_atom_positions: Float[Array, "n_atoms 3"],
    ) -> Float[Array, "n_steps n_atoms 3"]:
        """
        Sample a trajectory from the initial atom positions.

        **Arguments:**
        key: JAX random key for generating random numbers.
        initial_atom_positions: Initial positions of the atoms.
        energy_fn_args: Arguments for the energy function.

        **Returns:**
        trajectory: Langevin Dynamics Trajectory.
        """
        return _run_steered_langevin(
            key,
            initial_atom_positions,
            ref_atom_positions,
            self.n_steps,
            self.step_size,
            self.forcefield,
            self.biasing_force_constant,
        )


class ParallelSteeredLangevinSampler(AbstractLangevinSampler):
    """
    Parallel Langevin sampler for Langevin dynamics.
    """

    n_steps: Int
    step_size: Float
    forcefield: AbstractForceField
    biasing_force_constant: Float

    def __init__(
        self,
        n_steps: Int,
        step_size: Float,
        forcefield: AbstractForceField,
        biasing_force_constant: Float,
    ):
        """
        **Arguments:**
        - `n_steps`: Number of Langevin steps to take.
        - `step_size`: Step size for Langevin dynamics.
        - `forcefield`: Force field to use for Langevin dynamics.
        - `biasing_force_constant`: Force constant for the biasing potential.
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.forcefield = forcefield
        self.biasing_force_constant = biasing_force_constant

    def __call__(
        self,
        key: PRNGKeyArray,
        initial_atom_positions: Float[Array, "n_walkers n_atoms 3"],
        ref_atom_positions: Float[Array, "n_walkers n_atoms 3"],
    ) -> Float[Array, "n_walkers n_steps n_atoms 3"]:
        """
        Sample a trajectory from the initial atom positions.

        **Arguments:**
        key: JAX random key for generating random numbers.
        initial_atom_positions: Initial positions of the atoms.
        energy_fn_args: Arguments for the energy function.

        **Returns:**
        trajectory: Langevin Dynamics Trajectory for multiple walkers.
        """
        return _run_steered_langevin_parallel(
            key,
            initial_atom_positions,
            ref_atom_positions,
            self.n_steps,
            self.step_size,
            self.forcefield,
            self.biasing_force_constant,
        )


@eqx.filter_jit
def _take_steered_langevin_step(
    key: PRNGKeyArray,
    atom_positions: Float[Array, "n_atoms 3"],
    ref_atom_positions: Float[Array, "n_atoms 3"],
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
    step_size: Float,
) -> Float[Array, "n_atoms 3"]:
    energy_gradient = jax.grad(lambda x: forcefield(x))(atom_positions)

    biasing_gradient = jax.grad(compute_harmonic_bias_potential_energy)(
        atom_positions, ref_atom_positions, biasing_force_constant
    )

    gradient = energy_gradient + biasing_gradient
    # jax.debug.print("{gradient}", gradient=gradient)
    new_positions = (
        atom_positions
        + step_size * gradient
        + jnp.sqrt(2 * step_size) * jax.random.normal(key, shape=atom_positions.shape)
    )
    return new_positions


@eqx.filter_jit
def _run_steered_langevin(
    key: PRNGKeyArray,
    initial_walker: Float[Array, "n_atoms 3"],
    reference_walker: Float[Array, "n_atoms 3"],
    n_steps: int,
    step_size: Float,
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
):
    def _step_for_scan(carry, x):
        key, old_positions = carry
        key, subkey = jax.random.split(key)
        new_positions = _take_steered_langevin_step(
            subkey,
            old_positions,
            reference_walker,
            forcefield,
            biasing_force_constant,
            step_size,
        )
        return (key, new_positions), new_positions

    _, trajectory = jax.lax.scan(
        f=_step_for_scan,
        init=(key, initial_walker),
        length=n_steps,
    )
    return trajectory


@eqx.filter_jit
def _take_steered_langevin_step_parallel(
    key: PRNGKeyArray,
    atom_positions: Float[Array, "n_walkers n_atoms 3"],
    ref_atom_positions: Float[Array, "n_walkers n_atoms 3"],
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
    step_size: Float,
) -> Float[Array, "n_walkers n_atoms 3"]:
    energy_gradient = eqx.filter_vmap(jax.grad(lambda x: forcefield(x)))(atom_positions)

    biasing_gradient = eqx.filter_vmap(
        jax.grad(compute_harmonic_bias_potential_energy), in_axes=(0, 0, None)
    )(atom_positions, ref_atom_positions, biasing_force_constant)

    gradient = energy_gradient + biasing_gradient

    new_positions = (
        atom_positions
        + step_size * gradient
        + jnp.sqrt(2 * step_size) * jax.random.normal(key, shape=atom_positions.shape)
    )

    return new_positions


@eqx.filter_jit
def _run_steered_langevin_parallel(
    key: PRNGKeyArray,
    initial_walkers: Float[Array, "n_walkers n_atoms 3"],
    reference_walkers: Float[Array, "n_walkers n_atoms 3"],
    n_steps: int,
    step_size: Float,
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
):
    def _step_for_scan(carry, x):
        key, old_positions = carry
        key, subkey = jax.random.split(key)
        new_positions = _take_steered_langevin_step_parallel(
            subkey,
            old_positions,
            reference_walkers,
            forcefield,
            biasing_force_constant,
            step_size,
        )
        return (key, new_positions), new_positions

    _, trajectory = jax.lax.scan(
        f=_step_for_scan,
        init=(key, initial_walkers),
        length=n_steps,
    )
    return trajectory
