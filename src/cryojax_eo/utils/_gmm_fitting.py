from pathlib import Path
from typing import Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from cryojax.io import read_atoms_from_pdb
from cryojax.jax_util import make_filter_spec
from jaxtyping import Array, Float, Int


class Gaussian3D(eqx.Module, strict=True):  # todo: strict equal true
    bead_positions: Float[Array, "n_beads 3"]
    log_amplitude: Float
    log_variance: Float
    shape: Tuple[Int, Int, Int]
    voxel_size: Int
    n_gaussians_per_bead: Int

    def as_volume(self):
        ones = jnp.ones((self.bead_positions.shape[0], self.n_gaussians_per_bead))
        gmm_potential = cxs.GaussianMixtureVolume(
            self.bead_positions,
            amplitudes=jnp.exp(self.log_amplitude) * ones,
            variances=jnp.exp(self.log_variance) * ones,
        )

        return gmm_potential.to_real_voxel_grid(
            self.shape,
            self.voxel_size,
        )

    def save(self, filename: str | Path, overwrite: bool = False):
        ones = jnp.ones((self.bead_positions.shape[0], self.n_gaussians_per_bead))

        if Path(filename).exists() and not overwrite:
            raise FileExistsError(f"Filename {filename} exists, but overwrite is False")
        else:
            jnp.savez(
                filename,
                bead_positions=self.bead_positions,
                amplitudes=jnp.exp(self.log_amplitude) * ones,
                variances=jnp.exp(self.log_variance) * ones,
            )
        return


def make_gmm_model_from_atomic_model(
    pdb_file,
    box_size,
    voxel_size,
    *,
    fit_selection_string='name "C2"',
    init_log_amp=6.0,
    init_log_var=1.0,
    n_gaussians_per_bead=1,
    atol=1e-3,
    rtol=1e-3,
    max_steps=500,
):
    target_volume = _generate_target_volume(pdb_file, box_size, voxel_size)

    init_gmm_model = _generate_initial_model(
        pdb_file,
        box_size,
        voxel_size,
        selection_string=fit_selection_string,
        init_log_amp=init_log_amp,
        init_log_var=init_log_var,
        n_gaussians_per_bead=n_gaussians_per_bead,
    )

    return fit_gmm_model_to_volume(
        init_gmm_model, target_volume, atol=atol, rtol=rtol, max_steps=max_steps
    )


def fit_gmm_model_to_volume(
    init_gmm_model,
    target_volume,
    *,
    atol=1e-3,
    rtol=1e-3,
    max_steps=500,
):
    filter_spec = make_filter_spec(init_gmm_model, _where_gaussian3D_opt)

    gmm_model_opt, gmm_model_noopt = eqx.partition(init_gmm_model, filter_spec)
    y0, pytreedef = jax.tree.flatten(gmm_model_opt)

    sol = optx.least_squares(
        _compute_residues,
        y0=y0,
        args=(pytreedef, gmm_model_noopt, target_volume),
        solver=optx.GaussNewton(atol=atol, rtol=rtol),
        max_steps=max_steps,
        throw=True,
    )

    final_y = sol.value
    return eqx.combine(jax.tree.unflatten(pytreedef, final_y), gmm_model_noopt)


def _where_gaussian3D_opt(gaussian3d: Gaussian3D):
    return (
        gaussian3d.log_amplitude,
        gaussian3d.log_variance,
    )


def _generate_target_volume(reference_pdb_file, box_size, voxel_size):
    # read in atoms
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        reference_pdb_file, center=True, loads_b_factors=True
    )
    scattering_factor_parameters = cxs.PengScatteringFactorParameters(atom_types)

    # make target via peng volume
    atomic_potential = cxs.PengAtomicVolume.from_tabulated_parameters(
        atom_positions, scattering_factor_parameters, b_factors
    )

    return atomic_potential.to_real_voxel_grid(
        (box_size, box_size, box_size),
        voxel_size,
    )


def _generate_initial_model(
    reference_pdb_file,
    box_size,
    voxel_size,
    *,
    selection_string='name "C2"',
    init_log_amp=40.0,
    init_log_var=1.0,
    n_gaussians_per_bead=1,
):
    atom_positions, _ = read_atoms_from_pdb(
        reference_pdb_file, center=True, selection_string=selection_string
    )

    return Gaussian3D(
        bead_positions=jnp.array(atom_positions),
        log_amplitude=init_log_amp,
        log_variance=init_log_var,
        shape=(box_size, box_size, box_size),
        voxel_size=voxel_size,
        n_gaussians_per_bead=n_gaussians_per_bead,
    )


@eqx.filter_jit
def _compute_residues(y, args):
    pytreedef, gmm_model_noopt, target_volume = args
    gmm_volume = eqx.combine(
        jax.tree.unflatten(pytreedef, y), gmm_model_noopt
    ).as_volume()
    return gmm_volume - target_volume
