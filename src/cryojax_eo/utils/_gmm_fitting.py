from pathlib import Path
from typing import List, Tuple

import cryojax.simulator as cxs
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
from cryojax.io import read_atoms_from_pdb
from cryojax.jax_util import make_filter_spec
from jaxtyping import Array, Float, Int, PyTree


class Gaussian3D(eqx.Module, strict=True):
    positions: Float[Array, "n_beads 3"]
    amplitude: Float
    variance: Float
    shape: Tuple[Int, Int, Int]
    voxel_size: Int
    n_gaussians_per_bead: Int

    def to_real_voxel_grid(
        self,
    ) -> Float[Array, "{self.shape[0]} {self.shape[1]} {self.shape[2]}"]:
        gmm_volume = self.to_gmm_volume()

        return gmm_volume.to_real_voxel_grid(
            self.shape,
            self.voxel_size,
        )

    def to_gmm_volume(self) -> cxs.GaussianMixtureVolume:
        ones = jnp.ones((self.positions.shape[0], self.n_gaussians_per_bead))
        gmm_volume = cxs.GaussianMixtureVolume(
            self.positions,
            amplitudes=self.amplitude * ones,
            variances=self.variance * ones,
        )
        return gmm_volume


def make_gmm_model_from_atomic_model(
    pdb_file: str | Path,
    box_size: int,
    voxel_size: float,
    *,
    fit_selection_string: str = 'name "C2"',
    init_amp: float = 6.0,
    init_var: float = 1.0,
    n_gaussians_per_bead: int = 1,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    max_steps: int = 500,
) -> cxs.GaussianMixtureVolume:
    target_volume = _make_target_voxel_grid(pdb_file, box_size, voxel_size)

    init_gmm_model = _make_initial_gmm(
        pdb_file,
        box_size,
        voxel_size,
        selection_string=fit_selection_string,
        init_amp=init_amp,
        init_var=init_var,
        n_gaussians_per_bead=n_gaussians_per_bead,
    )

    return fit_gmm_model_to_voxel_grid(
        init_gmm_model, target_volume, atol=atol, rtol=rtol, max_steps=max_steps
    )


def fit_gmm_model_to_voxel_grid(
    init_gmm_model: Gaussian3D,
    target_volume: Float[Array, "z y x"],
    *,
    atol=1e-3,
    rtol=1e-3,
    max_steps=500,
) -> cxs.GaussianMixtureVolume:
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
    return eqx.combine(
        jax.tree.unflatten(pytreedef, final_y), gmm_model_noopt
    ).to_gmm_volume()


def _where_gaussian3D_opt(gaussian3d: Gaussian3D):
    return (
        gaussian3d.amplitude,
        gaussian3d.variance,
    )


def _make_target_voxel_grid(
    reference_pdb_file: str | Path, box_size: int, voxel_size: float
) -> Float[Array, "{box_size} {box_size} {box_size}"]:
    # read in atoms
    atom_positions, atom_types, b_factors = read_atoms_from_pdb(
        reference_pdb_file,
        center=True,
        loads_b_factors=True,
        selection_string="not element H",
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


def _make_initial_gmm(
    reference_pdb_file: str | Path,
    box_size: int,
    voxel_size: float,
    *,
    selection_string: str = 'name "C2"',
    init_amp: float = 40.0,
    init_var: float = 1.0,
    n_gaussians_per_bead: int = 1,
) -> Gaussian3D:
    atom_positions, _ = read_atoms_from_pdb(
        reference_pdb_file, center=True, selection_string=selection_string
    )

    return Gaussian3D(
        positions=jnp.array(atom_positions),
        amplitude=init_amp,
        variance=init_var,
        shape=(box_size, box_size, box_size),
        voxel_size=voxel_size,
        n_gaussians_per_bead=n_gaussians_per_bead,
    )


@eqx.filter_jit
def _compute_residues(
    y: List[float], args: Tuple[PyTree, Gaussian3D, Float[Array, "z y x"]]
) -> Float[Array, "z y x"]:
    pytreedef, gmm_model_noopt, target_volume = args
    gmm_volume = eqx.combine(
        jax.tree.unflatten(pytreedef, y), gmm_model_noopt
    ).to_real_voxel_grid()
    return gmm_volume - target_volume
