# Cryo-EM Flexible Fitting

Flexible fitting refines a single atomic model against a consensus cryo-EM density map using steered MD. Unlike ensemble optimization, there are no particle images or ensemble weights — the loss is a real-space cross-correlation between the projected model density and the reference volume.

You can run flexible fitting through the command line:

```bash
run_flexible_fitting --config config_flexible_fitting.yaml
```

## Input Format

Flexible fitting uses a custom YAML input format. An example config file for running on a GPU is given below:

```yaml
# Parameters marked with (*) are optional

path_to_atomic_model: /path/to/atomic_model.pdb         # initial structure (.pdb or .cif)
path_to_prealigned_atomic_model: /path/to/prealigned_atomic_model.pdb  # reference structure aligned to the map
path_to_output: ./flexible_fitting_output/

atom_selection: "not element H"  # (*) Default: "all". mdtraj selection string, or a .txt/.npy file of atom indices
loads_b_factors: false           # (*) Default: false. Load Debye-Waller b-factors from the PDB

reference_volume_params:
  path_to_reference_volume: /path/to/reference_volume.mrc  # consensus map (.mrc)
  flexible_fitting_box_size: 128  # (*) Default: 128. Box size (in pixels) used for the fitting loss
  rigid_alignment_box_size: 32    # (*) Default: 32.  Box size (in pixels) used for rigid-body alignment
  reference_volume_voxel_size:    # (*) Default: null. Overrides the voxel size in the MRC header
  path_to_volumetric_mask:        # (*) Default: null. Path to a mask (.mrc) applied to the cross-correlation loss
  path_to_weights:                # (*) Default: null. Fourier weights (.mrc); switches the loss to weighted MSE

projector_params:
  n_steps: 1000                         # number of MD steps per iteration
  bias_constant_in_kjpermol: 3.0e5      # biasing force constant (kJ/mol); or [start, end] for a linear schedule
  platform: CUDA                        # CPU, CUDA, or OpenCL
  platform_properties:                  # (*) platform-specific OpenMM settings
    DeviceIndex: '0'
# platform: CPU                         # CPU version
# platform_properties:
#   Threads: '32'                       # number of CPU threads (OpenMM does not always respect this)
  path_to_initial_state:                # (*) Default: null. Path to an OpenMM state XML for warm-starting
  md_params:                            # (*) Override default OpenMM simulation parameters. All fields are optional.
    forcefield: amber14-all.xml           # (*) Default: amber14-all.xml
    water_model: amber14/tip3p.xml        # (*) Default: amber14/tip3p.xml
    nonbonded_method: CutoffNonPeriodic   # (*) Options: PME, CutoffNonPeriodic, NoCutoff, CutoffPeriodic, Ewald, LJPME
    nonbonded_cutoff_nm: 1.0              # (*) Default: 1.0 nm
    constraints: HBonds                   # (*) Options: HBonds, AllBonds, HAngles, null (no constraints)
    temperature_K: 300.0                  # (*) Default: 300 K
    friction_per_ps: 1.0                  # (*) Default: 1.0 ps^-1
    timestep_ps: 0.002                    # (*) Default: 0.002 ps (2 fs)

walker_optimizer_params:
  type: steepest_desc   # (*) Default: steepest_desc. Options: steepest_desc, adam
  n_steps: 10           # (*) Default: 1. Number of gradient steps per iteration
  step_size: 2.0        # step size in Angstroms
  n_batches_of_atoms: 1       # (*) Default: 1. Atom batches for memory-friendly volume evaluation
  volume_render_backend:      # (*) How walkers are rendered onto the voxel grid. Every field is optional
    spread_mode: local        # (*) Default: local. Options: local, exact
    spread_width_in_stds: 6.0 # (*) Default: 6.0. Ignored when spread_mode is 'exact'
    enable_pallas: false      # (*) Default: false. Pallas/Triton GPU kernels. Requires a CUDA GPU and spread_mode: local

early_stopping:         # (*) Default: null (never stop early). Omit the whole block to disable
  patience: 10          # required when the block is present
  rtol: 1.0e-4          # (*) Default: 1.0e-4
  atol: 1.0e-4          # (*) Default: 1.0e-4

n_steps: 100  # total number of flexible fitting iterations
rng_seed: 0   # (*) Default: 0. Seeds the OpenMM thermostat; 0 draws a fresh seed each run
```

## Comments on the input parameters

**Alignment.** The `path_to_prealigned_atomic_model` must be aligned to the frame of reference of the consensus map. This reference structure is used only for rigid-body re-alignment at each iteration; the structure being optimized is `path_to_atomic_model`. Providing a `path_to_reference_volume` enables this alignment and is strongly recommended. The `rigid_alignment_box_size` controls the resolution of the alignment step — a value of 32 typically adds about 1 second per iteration.

**Flexible fitting box size.** The `flexible_fitting_box_size` sets the resolution at which the cross-correlation loss is computed. Larger boxes capture more detail but require more GPU memory. For memory-constrained settings, increase `n_batches_of_atoms` to spread atom contributions across sequential evaluations.

**Volumetric mask.** The optional `path_to_volumetric_mask` accepts a `.mrc` mask file (e.g., the dilated solvent mask from a homogeneous refinement job). It is Fourier-cropped to `flexible_fitting_box_size` and applied to the cross-correlation loss, focusing the fitting on a specific region of the map and suppressing noise outside the mask. If omitted, the loss is computed over the full volume.

**Loss function.** By default the fit maximizes the real-space cross-correlation between the rendered model density and the reference volume. Supplying `path_to_weights` switches the loss to a Fourier-weighted MSE instead. The weights are read from an `.mrc` file whose box size must be an integer multiple of `flexible_fitting_box_size`, and can be obtained by running `relion_reconstruct` with `--external_reconstruct`. The progress bar reports `C.C` for the cross-correlation loss and `MSE` for the weighted one.

**Bias constant.** The `bias_constant_in_kjpermol` controls how strongly the MD simulation is steered toward the gradient direction. A single float applies a constant force; a two-element list `[start, end]` applies a linear schedule from `start` to `end` over all `n_steps` iterations. This parameter typically requires tuning for each system. Values that are too large cause the simulation to explode; values that are too small produce no structural change.

**Warm-starting.** The `path_to_initial_state` accepts an OpenMM state XML file (e.g., saved from a previous run or a pre-equilibrated simulation) and can improve convergence by starting from a well-equilibrated configuration.

**MD simulation parameters.** The optional `md_params` block lets you override any of the underlying OpenMM simulation parameters directly from the YAML config. All fields are optional and fall back to built-in defaults when omitted. The `nonbonded_method` field accepts string aliases: `CutoffNonPeriodic` (default, for implicit/vacuum simulations), `PME` or `Ewald` (explicit solvent with a periodic box), `CutoffPeriodic` (periodic box with a simple cutoff), `LJPME` (PME for both electrostatics and Lennard-Jones), and `NoCutoff` (no cutoff; very small systems only). The `constraints` field accepts `HBonds` (default, compatible with a 2 fs timestep), `AllBonds`, `HAngles` (allows ~4 fs timesteps), or `null` to disable constraints. Numeric fields use explicit unit suffixes: `_nm` for nanometers, `_K` for Kelvin, `_per_ps` for ps⁻¹, and `_ps` for picoseconds. The `platform` and `platform_properties` fields are set at the top level of `projector_params` and do not belong inside `md_params`.

**Walker optimizer.** The `type` field inside `walker_optimizer_params` selects how the walker is moved along the loss gradient between MD projections. `steepest_desc` (the default) normalizes the per-atom gradient before stepping, so `step_size` is literally how far each atom moves, in Angstroms. `adam` instead uses an Adam update with `step_size` as its learning rate, in which case the displacement is adaptive and `step_size` is only an upper bound on it. Each iteration takes `n_steps` such steps before handing the structure back to the MD projector.

**Early stopping.** The optional `early_stopping` block halts the run once the loss stops improving. `patience` is the number of iterations without improvement to tolerate and is required whenever the block is present; `rtol` and `atol` set the relative and absolute tolerances used to decide what counts as an improvement, and both default to `1e-4`. Omit the block entirely to run all `n_steps` iterations — note that writing `early_stopping:` with nothing under it is the same as omitting it.

**Reproducibility.** The top-level `rng_seed` seeds the OpenMM Langevin thermostat so a run can be reproduced. The default of `0` is special: it keeps OpenMM's own behavior of drawing a fresh seed for every run, which makes runs non-reproducible. Set any non-zero integer to fix the thermostat's random stream.

### Volume render backend

Every loss evaluation renders the walker — a mixture of gaussians, one per atom — onto a voxel grid of size `flexible_fitting_box_size` so it can be compared against the reference volume. The optional `volume_render_backend` block inside `walker_optimizer_params` controls how that rendering is computed. It affects only speed, memory, and numerical accuracy, not the model being optimized, and all of its fields fall back to built-in defaults when omitted. It does not affect the rigid-body alignment step or the MD projector.

- `spread_mode` (default `local`): `local` spreads each gaussian onto only the voxels near its center, with the truncation width set by `spread_width_in_stds`. `exact` instead evaluates dense gaussian integrals over the whole grid. Since gaussians are short-ranged compared to typical box sizes, `local` is much faster for the same result, and is the recommended choice; use `exact` when you want a reference calculation with no truncation. In `local` mode the number of voxels per gaussian is derived automatically from the atom variances and the voxel size. Unlike the 2D projection used in ensemble optimization, the rendered grid is never computed oversized and cropped back — its shape is always exactly `flexible_fitting_box_size`, since it is compared voxel-by-voxel against the reference map.
- `spread_width_in_stds` (default `6.0`): how far each gaussian is spread, in standard deviations, when `spread_mode: local`. Lowering it makes the rendering cheaper in both time and memory, at the cost of truncating the tails more aggressively. It is ignored when `spread_mode: exact`.
- `enable_pallas` (default `false`): whether to use [Pallas](https://docs.jax.dev/en/latest/pallas/index.html)/Triton GPU kernels instead of the pure-JAX implementation. This requires a CUDA GPU — it raises an error if requested without one — and only applies when `spread_mode: local`; it is ignored for `exact`.

The main reason to set `enable_pallas: true` is memory rather than speed. The pure-JAX forward pass is usually the faster of the two, but the Pallas backward pass is a pure gather with a flat memory profile, which matters because rendering a 3D grid is far more memory-hungry than projecting to a 2D plane. It is worth enabling when a large `flexible_fitting_box_size` exhausts GPU memory; `n_batches_of_atoms` is the other knob for the same problem, and the two can be combined. If you are not memory-limited, leave it `false`. Note that this field is always passed through explicitly, so it takes precedence over the `CRYOJAX_ENABLE_PALLAS` environment variable.

## Outputs

- `final_walker.npy`: final refined atomic positions as a NumPy array.
- `final_walker.pdb`: the same final structure as a PDB file.
- `curr_walker.pdb`: the most recent structure, overwritten at each iteration. Useful for watching the fit progress while it runs.
- `traj_walker.xtc`: the trajectory followed by the walker over the course of the fit.
- `states_proj/state_*`: OpenMM state files saved at each iteration. Useful for restarting or inspecting the MD trajectory.
- A `log` file named by the date the run was started.
- A copy of the input config YAML written to the output directory.
