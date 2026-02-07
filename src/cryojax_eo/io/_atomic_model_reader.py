import jax.numpy as jnp
import mdtraj
import numpy as np
from cryojax.constants import (
    PengScatteringFactorParameters,
    b_factor_to_variance,
)
from cryojax.io import read_atoms_from_pdb
from jaxtyping import Array, Float, Int

from cryojax_eo.utils import rigid_align_positions


def read_walkers_from_pdbs(
    filenames: list[str],
    *,
    loads_b_factors: bool = False,
) -> tuple[
    Float[Array, "n_walkers n_atoms 3"],
    Float[Array, "n_atoms n_gaussians"],
    Float[Array, "n_atoms n_gaussians"],
]:
    """
    Reads atomic models from PDB files and extracts positions, scattering amplitudes, and
    variances for each model. The function can also load Debye-Waller b-factors
    if specified. Due to the constraints of this library, the variances and
    amplitudes must be consistent across all models. This means that the atomic models
    must have the same atomic composition, but the positions can differ.

    **Arguments:**
    - `filenames`:
        List of paths to PDB files containing atomic models.
    - `loads_b_factors`:
        If `True`, the function will read the B-factors from the PDB files
        and use them to compute the variances. If `False`, default variances
        based on atomic types will be used.

    **Returns:**
    A tuple containing:
    - `walkers`:
        A JAX array of shape `(n_walkers, n_atoms, 3)`
        with the atomic positions for each model.
    - `amplitudes`:
        A JAX array of shape `(n_atoms, n_gaussians)` with the scattering amplitudes.
    - `variances`:
        A JAX array of shape `(n_atoms, n_gaussians)` with the scattering variances.
    """

    walkers_and_params = {}
    for i in range(len(filenames)):
        atom_positions, atomic_numbers, atom_properties = read_atoms_from_pdb(
            filenames[i], center=True, loads_properties=True
        )
        variances, amplitudes = _compute_scattering_params(
            atomic_numbers,
            atom_properties["b_factors"] if loads_b_factors else None,
        )
        walkers_and_params[i] = {
            "positions": atom_positions,
            "amplitudes": amplitudes,
            "variances": variances,
        }

    _assert_consistent_scattering_params(walkers_and_params)

    variances = walkers_and_params[0]["variances"]
    amplitudes = walkers_and_params[0]["amplitudes"]
    walkers = np.array(
        [walkers_and_params[i]["positions"] for i in range(len(filenames))]
    )
    topology = mdtraj.load(filenames[0]).topology
    walkers = _align_walkers_to_reference(walkers, topology)

    return walkers, amplitudes, variances


def _align_walkers_to_reference(
    walkers: Float[Array, "n_walkers n_atoms 3"],
    topology: mdtraj.Topology,
):
    aligned_walkers = np.zeros_like(walkers)
    atom_indices = topology.select("name CA")
    for i in range(walkers.shape[0]):
        _, rot_matrix, displacement = rigid_align_positions(
            walkers[i, atom_indices], walkers[0, atom_indices]
        )
        aligned_walkers[i] = walkers[i] @ rot_matrix.T + displacement

    return aligned_walkers


def _assert_consistent_scattering_params(walkers_and_params: dict):
    """
    Asserts that the scattering amplitudes and variances are consistent across
    all models. This is necessary because the current implementation of the optimization
    process assumes that all models have the same atomic composition,
    and thus the same scattering parameters.
    """
    reference_amplitudes = walkers_and_params[0]["amplitudes"]
    reference_variances = walkers_and_params[0]["variances"]

    for i, (model_id, params) in enumerate(walkers_and_params.items()):
        if i == 0:
            continue  # Skip the reference model

        _var_shapes_match = params["amplitudes"].shape == reference_amplitudes.shape
        _amp_shapes_match = params["variances"].shape == reference_variances.shape
        if not _var_shapes_match or not _amp_shapes_match:
            raise ValueError(
                f"Scattering parameters for model {model_id} "
                "have inconsistent shapes with the reference model."
            )
        else:
            if not jnp.allclose(params["amplitudes"], reference_amplitudes):
                raise ValueError(
                    f"Scattering amplitudes for model {model_id} "
                    "are inconsistent with the reference model."
                )
            if not jnp.allclose(params["variances"], reference_variances):
                raise ValueError(
                    f"Scattering variances for model {model_id} "
                    "are inconsistent with the reference model."
                )


def _compute_scattering_params(
    atomic_numbers: Int[Array, " n_atoms"],
    b_factors: Float[Array, " n_atoms"] | None,
):
    scattering_factors = PengScatteringFactorParameters(atomic_numbers)
    amplitudes = scattering_factors.a

    if b_factors is None:
        variances = scattering_factors.b
    else:
        variances = b_factor_to_variance(scattering_factors.b + b_factors[:, None])

    return amplitudes, variances
