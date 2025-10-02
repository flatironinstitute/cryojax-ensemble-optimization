from pathlib import Path
from typing import Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from cryojax.constants import (
    get_tabulated_scattering_factor_parameters,
)
from cryojax.io import read_atoms_from_pdb
from cryojax.jax_util import get_filter_spec
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
        gmm_potential = cxs.GaussianMixtureAtomicPotential(
            self.bead_positions,
            gaussian_amplitudes=jnp.exp(self.log_amplitude) * ones,
            gaussian_variances=jnp.exp(self.log_variance) * ones,
        )

        return gmm_potential.as_real_voxel_grid(
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
                gaussian_amplitudes=jnp.exp(self.log_amplitude) * ones,
                gaussian_variances=jnp.exp(self.log_variance) * ones,
            )
        return


def generate_gmm_model_from_atomic_model(
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
    filter_spec = get_filter_spec(init_gmm_model, _where_gaussian3D_opt)

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
    atom_positions, atom_identities, _ = read_atoms_from_pdb(
        reference_pdb_file, center=True, loads_b_factors=True
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities
    )

    # make target via peng potential
    atomic_potential = cxs.PengAtomicPotential(  # use gmm
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
    )

    return atomic_potential.as_real_voxel_grid(
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
        bead_positions=atom_positions,
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


"""
import hydra
from omegaconf import OmegaConf

repo_root = Path(
    __file__
).parent.parent  # Navigate to repo root (assuming this script is under /src)
config_path = repo_root / "config_files"  # Path to the configs folder
output_dir = repo_root / "outputs"  # Path to outputs directory


class Gaussian3D(eqx.Module):  # todo: strict equal true
    log_var: jnp.array
    log_weight: jnp.array

    def __call__(self, atom_positions, n_pix, voxel_size, n_gaussians_per_bead):
        ones = jnp.ones((atom_positions.shape[0], n_gaussians_per_bead))
        coasegrained_potential = GaussianMixtureAtomicPotential(
            atom_positions,
            gaussian_amplitudes=jnp.exp(self.log_weight) * ones,
            gaussian_variances=jnp.exp(self.log_var) * ones,
        )
        n_voxels_per_side = (n_pix, n_pix, n_pix)

        cgpotential_as_real_voxel_grid = coasegrained_potential.as_real_voxel_grid(
            n_voxels_per_side,
            voxel_size,
        )
        return cgpotential_as_real_voxel_grid


@dataclass
class CoarseGrainInitialization:
    variance: float
    weight: float


@dataclass
class CoarseGrainOptimization:
    max_steps: int
    atol: float
    rtol: float
    variance: float
    initial_point: CoarseGrainInitialization


@dataclass
class CoarseGrain:
    pdb_fname: str
    fname_out: str
    mdtraj_select: str
    n_pix: int
    voxel_size: float  # positive float
    optimization: CoarseGrainOptimization


@hydra.main(config_path="config_files", config_name="config_coarse_grain")
def param_gaussian_3d(cfg: CoarseGrain):
    args = OmegaConf.to_container(cfg, resolve=True)

    # read in atoms
    fname = args["pdb_fname"]
    atom_positions, atom_identities, _ = read_atoms_from_pdb(
        fname, center=True, loads_b_factors=True
    )
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )

    # make target via peng potential
    atomic_potential = PengAtomicPotential(  # use gmm
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
    )
    n_pix = args["n_pix"]
    n_voxels_per_side = (n_pix, n_pix, n_pix)
    voxel_size = args["voxel_size"]
    target = potential_as_real_voxel_grid = atomic_potential.as_real_voxel_grid(
        n_voxels_per_side,
        voxel_size,
    )

    # select centering atom for coarse grained model
    atom_positions, atom_identities, _ = read_atoms_from_pdb(
        fname, center=True, loads_b_factors=True, selection_string=args["mdtraj_select"]
    )

    # make model for iterative least squares optimization based inference (optimistix)
    model = Gaussian3D(
        log_var=jnp.log(args["optimization"]["initial_point"]["variance"]),
        log_weight=jnp.log(args["optimization"]["initial_point"]["weight"]),
    )

    def residual_fn(model: Gaussian3D, args):
        atom_positions, n_pix, voxel_size, n_gaussians_per_bead, target = args
        return model(atom_positions, n_pix, voxel_size, n_gaussians_per_bead) - target

    target = potential_as_real_voxel_grid
    n_pix = args["n_pix"]
    voxel_size = args["voxel_size"]
    n_gaussians_per_bead = 1  # TODO: consider generalizing to 3-5 gaussians
    sol = optx.least_squares(
        residual_fn,
        y0=model,
        args=(atom_positions, n_pix, voxel_size, n_gaussians_per_bead, target),
        solver=optx.GaussNewton(
            atol=args["optimization"]["atol"], rtol=args["optimization"]["rtol"]
        ),
        max_steps=args["optimization"]["max_steps"],
        throw=True,
    )

    fitted = sol.value
    fitted_var = jnp.exp(fitted.log_var)
    fitted_weight = jnp.exp(fitted.log_weight)
    print("Fitted variance:", fitted_var)  # TODO: logger
    print("Fitted weight:", fitted_weight)

    # write out fitted parameters
    jnp.savez(
        args["fname_out"],
        var=jnp.exp(fitted.log_var),
        weight=jnp.exp(fitted.log_weight),
    )

    # test gmm projection (unit convention):
    # GaussianMixtureAtomicPotential.GaussianMixtureProjection
    # and GaussianMixtureProjection.as_real_voxel_grid.sum(0)
    n_beads = len(atom_positions)
    n_gaussians_per_bead = 1  # generalize?
    gaussian_mixture_projection = projection_from_params(
        atom_positions,
        gaussian_amplitudes=jnp.ones((n_beads, n_gaussians_per_bead)) * fitted_weight,
        gaussian_variances=jnp.ones((n_beads, n_gaussians_per_bead)) * fitted_var,
        shape=n_voxels_per_side[:2],
        voxel_size=voxel_size,
    )
    fitted_projection = fitted(
        atom_positions, n_pix, voxel_size, n_gaussians_per_bead
    ).sum(0)
    assert jnp.allclose(gaussian_mixture_projection, fitted_projection, atol=1e-7)


def projection_from_params(
    atom_positions, gaussian_amplitudes, gaussian_variances, shape, voxel_size
):
    fit_potential = GaussianMixtureAtomicPotential(
        atom_positions,
        gaussian_amplitudes=gaussian_amplitudes,
        gaussian_variances=gaussian_variances,
    )

    integrator = cxs.GaussianMixtureProjection(use_error_functions=True)

    instrument_config = InstrumentConfig(
        shape=shape,
        voxel_size=voxel_size,
        voltage_in_kilovolts=300.0,
    )
    gaussian_mixture_projection = (
        integrator.compute_integrated_potential(
            fit_potential, instrument_config, outputs_real_space=True
        )
        / voxel_size
    )  # divide by pixel size to get agreement
    return gaussian_mixture_projection


if __name__ == "__main__":
    param_gaussian_3d()
"""
