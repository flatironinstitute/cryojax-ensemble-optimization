import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from dataclasses import dataclass
import hydra
from pathlib import Path
from omegaconf import OmegaConf

from cryojax.simulator import GaussianMixtureAtomicPotential
from cryojax.simulator import PengAtomicPotential
import cryojax.simulator as cxs
from cryojax.io import read_atoms_from_pdb
from cryojax.constants import (
    get_tabulated_scattering_factor_parameters,
    read_peng_element_scattering_factor_parameter_table,
)
from cryojax.simulator import InstrumentConfig


repo_root = Path(__file__).parent.parent  # Navigate to repo root (assuming this script is under /src)
config_path = repo_root / "config_files"  # Path to the configs folder
output_dir = repo_root / "outputs"  # Path to outputs directory


class Gaussian3D(eqx.Module): # todo: strict equal true
    log_var: jnp.array
    log_weight: jnp.array

    def __call__(self, atom_positions, n_pix, voxel_size, n_gaussians_per_bead):
        ones = jnp.ones((atom_positions.shape[0], n_gaussians_per_bead))
        coasegrained_potential = GaussianMixtureAtomicPotential(
            atom_positions,
            gaussian_amplitudes=jnp.exp(self.log_weight)*ones,
            gaussian_variances=jnp.exp(self.log_var)*ones,
        )
        n_voxels_per_side = (n_pix, n_pix, n_pix)

        cgpotential_as_real_voxel_grid = coasegrained_potential.as_real_voxel_grid(
            n_voxels_per_side, voxel_size, 
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
    voxel_size: float # positive float
    optimization: CoarseGrainOptimization


@hydra.main(config_path='config_files', config_name="config_coarse_grain")
def param_gaussian_3d(cfg: CoarseGrain):
    args = OmegaConf.to_container(cfg, resolve=True)


    # read in atoms
    fname = args['pdb_fname'] 
    atom_positions, atom_identities, _ = read_atoms_from_pdb(fname, center=True, loads_b_factors=True)
    scattering_factor_parameters = get_tabulated_scattering_factor_parameters(
        atom_identities, read_peng_element_scattering_factor_parameter_table()
    )

    # make target via peng potential
    atomic_potential = PengAtomicPotential( # use gmm
        atom_positions,
        scattering_factor_a=scattering_factor_parameters["a"],
        scattering_factor_b=scattering_factor_parameters["b"],
    )
    n_pix = args['n_pix']
    n_voxels_per_side = (n_pix, n_pix, n_pix)
    voxel_size = args['voxel_size'] 
    target = potential_as_real_voxel_grid = atomic_potential.as_real_voxel_grid(
        n_voxels_per_side, voxel_size, 
    )

    # select centering atom for coarse grained model
    select = args['mdtraj_select']
    atom_positions, atom_identities, _ = read_atoms_from_pdb(fname, center=True, loads_b_factors=True, select=select)

    # make model for iterative least squares optimization based inference (optimistix)
    model = Gaussian3D(
        log_var=jnp.log(args['optimization']['initial_point']['variance']),
        log_weight=jnp.log(args['optimization']['initial_point']['weight']),
    )

    def residual_fn(model: Gaussian3D, args):
        atom_positions, n_pix, voxel_size, n_gaussians_per_bead, target = args
        return model(atom_positions, n_pix, voxel_size, n_gaussians_per_bead) - target

    target = potential_as_real_voxel_grid
    n_pix = args['n_pix']
    voxel_size = args['voxel_size']
    n_gaussians_per_bead = 1 #TODO: consider generalizing to 3-5 gaussians
    sol = optx.least_squares(
        residual_fn,
        y0=model,
        args=(atom_positions, n_pix, voxel_size, n_gaussians_per_bead, target),
        solver=optx.GaussNewton(atol=args['optimization']['atol'], rtol=args['optimization']['rtol']), 
        max_steps=args['optimization']['max_steps'],
        throw=True
    )

    fitted = sol.value
    fitted_var = jnp.exp(fitted.log_var)
    fitted_weight = jnp.exp(fitted.log_weight)
    print("Fitted variance:", fitted_var) #TODO: logger
    print("Fitted weight:", fitted_weight)

    # write out fitted parameters
    jnp.savez(
        args['fname_out'],
        var=jnp.exp(fitted.log_var),
        weight=jnp.exp(fitted.log_weight),
    )

    # test gmm projection (unit convention): GaussianMixtureAtomicPotential.GaussianMixtureProjection and GaussianMixtureProjection.as_real_voxel_grid.sum(0)
    n_beads = len(atom_positions)
    n_gaussians_per_bead = 1 # generalize?
    gaussian_mixture_projection = projection_from_params(atom_positions, 
                               gaussian_amplitudes=jnp.ones((n_beads, n_gaussians_per_bead))*fitted_weight, 
                               gaussian_variances=jnp.ones((n_beads, n_gaussians_per_bead))*fitted_var,
                               shape=n_voxels_per_side[:2],
                               pixel_size=voxel_size,
    )
    fitted_projection = fitted(atom_positions, n_pix, voxel_size, n_gaussians_per_bead).sum(0)
    assert jnp.allclose(gaussian_mixture_projection, fitted_projection, atol=1e-7)


def projection_from_params(atom_positions, gaussian_amplitudes, gaussian_variances, shape, pixel_size):
    
    fit_potential = GaussianMixtureAtomicPotential(
        atom_positions,
        gaussian_amplitudes=gaussian_amplitudes,
        gaussian_variances=gaussian_variances,
        )

    integrator = cxs.GaussianMixtureProjection(use_error_functions=True) 

    instrument_config = InstrumentConfig(
        shape=shape,
        pixel_size=pixel_size,
        voltage_in_kilovolts=300.0,
    )
    gaussian_mixture_projection = integrator.compute_integrated_potential(fit_potential, instrument_config, outputs_real_space=True) / pixel_size # divide by pixel size to get agreement
    return gaussian_mixture_projection


if __name__ == "__main__":

    param_gaussian_3d()