from typing import Tuple, TypeVar
from typing_extensions import override

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray

from .._forcefields.base_forcefield import AbstractForceField
from .._forcefields.biasing_forces import compute_harmonic_bias_potential_energy
from ..base_prior_projector import AbstractPriorProjector


EnergyFuncArgs = TypeVar("EnergyFuncArgs")


class SteeredOverdampedLangevinSampler(AbstractPriorProjector):
    """
    Overdamped Langevin sampler for Overdamped Langevin dynamics.
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
        - `n_steps`: Number of steps to take.
        - `step_size`: Step size for Overdamped Langevin dynamics.
        - `forcefield`: Force field to use for Overdamped Langevin dynamics.
        - `biasing_force_constant`: Force constant for the biasing potential.
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.forcefield = forcefield
        self.biasing_force_constant = biasing_force_constant

    @override
    def initialize(
        self, init_state: Tuple[PRNGKeyArray, Float[Array, "n_atoms 3"]]
    ) -> Tuple[PRNGKeyArray, Float[Array, "n_atoms 3"]]:
        """
        Initialize the sampler with the initial state. For this sampler, the initial state
        is the indentity, and its purpsoe is simply to validate the type of the input.

        **Arguments:**
        init_state: Initial positions of the atoms or walkers, and initial key.

        **Returns:**
        Initial state for the sampler.
        """
        key, initial_walker = init_state

        assert (
            initial_walker.ndim == 2 and initial_walker.shape[1] == 3
        ), "initial_walker must be a 2D array with shape (n_atoms, 3)"
        assert jnp.isdtype(
            initial_walker, "real floating"
        ), "initial_walker must be a real floating point array"

        return init_state

    def __call__(
        self,
        ref_walkers: Float[Array, "n_atoms 3"],
        state: Tuple[PRNGKeyArray, Float[Array, "n_atoms 3"]],
    ) -> Tuple[
        Float[Array, "n_atoms 3"],
        Tuple[PRNGKeyArray, Float[Array, "n_atoms 3"]],
    ]:
        """
        Sample a trajectory from the initial atom positions.

        **Arguments:**
        key: JAX random key for generating random numbers.
        initial_walkers: Initial positions of the atoms.
        energy_fn_args: Arguments for the energy function.

        **Returns:**
        trajectory: Overdamped Langevin Dynamics Trajectory.
        """

        key, initial_walkers = state
        key, walkers = _run_steered_overdamped_langevin(
            key,
            initial_walkers,
            ref_walkers,
            self.n_steps,
            self.step_size,
            self.forcefield,
            self.biasing_force_constant,
        )
        return walkers, (key, walkers)


class ParallelSteeredOverdampedLangevinSampler(AbstractPriorProjector):
    """
    Parallel Overdamped Langevin sampler for Overdamped Langevin dynamics.
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
        - `n_steps`: Number of steps to take.
        - `step_size`: Step size for Overdamped Langevin dynamics.
        - `forcefield`: Force field to use for Overdamped Langevin dynamics.
        - `biasing_force_constant`: Force constant for the biasing potential.
        """
        self.n_steps = n_steps
        self.step_size = step_size
        self.forcefield = forcefield
        self.biasing_force_constant = biasing_force_constant

    @override
    def initialize(
        self, init_state: Tuple[PRNGKeyArray, Float[Array, "n_walkers n_atoms 3"]]
    ) -> Tuple[PRNGKeyArray, Float[Array, "n_atoms 3"]]:
        """
        Initialize the sampler with the initial state. For this sampler, the initial state
        is the indentity, and its purpsoe is simply to validate the type of the input.

        **Arguments:**
        init_state: Initial positions of the atoms or walkers, and initial key.

        **Returns:**
        Initial state for the sampler.
        """
        key, initial_walker = init_state

        assert (
            initial_walker.ndim == 3 and initial_walker.shape[2] == 3
        ), "initial_walker must be a 2D array with shape (n_atoms, 3)"
        assert jnp.isdtype(
            initial_walker, "real floating"
        ), "initial_walker must be a real floating point array"

        return init_state

    def __call__(
        self,
        ref_walkers: Float[Array, "n_walkers n_atoms 3"],
        state: Tuple[PRNGKeyArray, Float[Array, "n_walkers n_atoms 3"]],
    ) -> Tuple[
        Float[Array, "n_walkers n_atoms 3"],
        Tuple[PRNGKeyArray, Float[Array, "n_walkers n_atoms 3"]],
    ]:
        """
        Sample a trajectory from the initial atom positions.

        **Arguments:**
        key: JAX random key for generating random numbers.
        initial_walkers: Initial positions of the atoms.
        energy_fn_args: Arguments for the energy function.

        **Returns:**
        trajectory: Overdamped Langevin Dynamics Trajectory.
        """

        key, initial_walkers = state
        key, walkers = _run_steered_overdamped_langevin_parallel(
            key,
            initial_walkers,
            ref_walkers,
            self.n_steps,
            self.step_size,
            self.forcefield,
            self.biasing_force_constant,
        )
        return walkers, (key, walkers)


@eqx.filter_jit
def _take_steered_overdamped_langevin_step(
    key: PRNGKeyArray,
    atom_positions: Float[Array, "n_atoms 3"],
    ref_walkers: Float[Array, "n_atoms 3"],
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
    step_size: Float,
) -> Float[Array, "n_atoms 3"]:
    energy_gradient = jax.grad(lambda x: forcefield(x))(atom_positions)

    biasing_gradient = jax.grad(compute_harmonic_bias_potential_energy)(
        atom_positions, ref_walkers, biasing_force_constant
    )

    gradient = energy_gradient + biasing_gradient
    # jax.debug.print("{gradient}", gradient=gradient)
    new_positions = (
        atom_positions
        - step_size * gradient
        + jnp.sqrt(2 * step_size) * jax.random.normal(key, shape=atom_positions.shape)
    )
    return new_positions


@eqx.filter_jit
def _run_steered_overdamped_langevin(
    key: PRNGKeyArray,
    initial_walker: Float[Array, "n_atoms 3"],
    reference_walker: Float[Array, "n_atoms 3"],
    n_steps: int,
    step_size: Float,
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
):
    def _step_func(i, val):
        key, old_positions = val
        key, subkey = jax.random.split(key)
        new_positions = _take_steered_overdamped_langevin_step(
            subkey,
            old_positions,
            reference_walker,
            forcefield,
            biasing_force_constant,
            step_size,
        )
        return (key, new_positions)

    return jax.lax.fori_loop(
        lower=0,
        upper=n_steps,
        body_fun=_step_func,
        init_val=(key, initial_walker),
    )


@eqx.filter_jit
def _take_steered_overdamped_langevin_step_parallel(
    key: PRNGKeyArray,
    atom_positions: Float[Array, "n_walkers n_atoms 3"],
    ref_walkers: Float[Array, "n_walkers n_atoms 3"],
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
    step_size: Float,
) -> Float[Array, "n_walkers n_atoms 3"]:
    energy_gradient = eqx.filter_vmap(jax.grad(lambda x: forcefield(x)))(atom_positions)

    biasing_gradient = eqx.filter_vmap(
        jax.grad(compute_harmonic_bias_potential_energy), in_axes=(0, 0, None)
    )(atom_positions, ref_walkers, biasing_force_constant)

    gradient = energy_gradient + biasing_gradient

    new_positions = (
        atom_positions
        - step_size * gradient
        + jnp.sqrt(2 * step_size) * jax.random.normal(key, shape=atom_positions.shape)
    )

    return new_positions


@eqx.filter_jit
def _run_steered_overdamped_langevin_parallel(
    key: PRNGKeyArray,
    initial_walkers: Float[Array, "n_walkers n_atoms 3"],
    reference_walkers: Float[Array, "n_walkers n_atoms 3"],
    n_steps: int,
    step_size: Float,
    forcefield: AbstractForceField,
    biasing_force_constant: Float,
):
    def _step_func(i, val):
        key, old_positions = val
        key, subkey = jax.random.split(key)
        new_positions = _take_steered_overdamped_langevin_step_parallel(
            subkey,
            old_positions,
            reference_walkers,
            forcefield,
            biasing_force_constant,
            step_size,
        )
        return (key, new_positions)

    return jax.lax.fori_loop(
        lower=0,
        upper=n_steps,
        body_fun=_step_func,
        init_val=(key, initial_walkers),
    )
