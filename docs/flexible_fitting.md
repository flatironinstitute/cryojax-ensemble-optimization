# Cryo-EM Flexible Fitting Input

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

walker_optimizer_params:
  n_steps: 10           # (*) Default: 1. Number of gradient steps per iteration
  step_size: 2.0        # step size in Angstroms
  batch_size_for_z_planes: 1  # (*) Default: 1. Z-planes evaluated in parallel with jax.vmap
  n_batches_of_atoms: 1       # (*) Default: 1. Atom batches for memory-friendly volume evaluation

n_steps: 100  # total number of flexible fitting iterations
```

## Comments on the input parameters

**Alignment.** The `path_to_prealigned_atomic_model` must be aligned to the frame of reference of the consensus map. This reference structure is used only for rigid-body re-alignment at each iteration; the structure being optimized is `path_to_atomic_model`. Providing a `path_to_reference_volume` enables this alignment and is strongly recommended. The `rigid_alignment_box_size` controls the resolution of the alignment step — a value of 32 typically adds about 1 second per iteration.

**Flexible fitting box size.** The `flexible_fitting_box_size` sets the resolution at which the cross-correlation loss is computed. Larger boxes capture more detail but require more GPU memory. For memory-constrained settings, increase `n_batches_of_atoms` to spread atom contributions across sequential evaluations, and increase `batch_size_for_z_planes` to parallelize over z-planes with `jax.vmap`.

**Bias constant.** The `bias_constant_in_kjpermol` controls how strongly the MD simulation is steered toward the gradient direction. A single float applies a constant force; a two-element list `[start, end]` applies a linear schedule from `start` to `end` over all `n_steps` iterations. This parameter typically requires tuning for each system. Values that are too large cause the simulation to explode; values that are too small produce no structural change.

**Warm-starting.** The `path_to_initial_state` accepts an OpenMM state XML file (e.g., saved from a previous run or a pre-equilibrated simulation) and can improve convergence by starting from a well-equilibrated configuration.

## Outputs

- `final_walker.npy`: final refined atomic positions as a NumPy array.
- `states_proj/state_*`: OpenMM state files saved at each iteration. Useful for restarting or inspecting the MD trajectory.
- A `log` file named by the date the run was started.
- A copy of the input config YAML written to the output directory.
