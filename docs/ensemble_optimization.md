# Cryo-EM Ensemble Optimization

You can run the ensemble optimization through our API, as shown in our [Tutorial](tutorial/ensemble_optimization/running_the_optimization.ipynb), or through the command line:

```bash
run_ensemble_optimization --config config_optimization.yaml
```

## Input Format

Cryo-EM Ensemble Optimization uses a custom YAML input format. A sample [config file](./example_configs/config_optimization.yaml) for running the ensemble optimization on a GPU with three walkers is shown below:

```yaml
# Parameters marked with (*) are optional

data_params:
  path_to_relion_project: /path/to/relion/project/
  path_to_starfile: /path/to/starfile.star
  loads_envelope: True # (*) Default: false, loads envelope function parameters
  data_sign: "dark-on-light" # Relion convention, dark particles on a light background
  path_to_volumetric_mask: /path/to/volumetric_mask.mrc # (*) e.g., the mask from a refinement procedure

alignment_params:
  path_to_prealigned_atomic_model: /path/to/prealigned_atomic_model.pdb # an atomic model aligned to the consensus volume
  path_to_reference_volume: path_to_reference_volume.mrc # (*) Reference volume (for local alignment)
  downsample_box_size: 32 # (*) Downsample box size for reference volume

projector_params:
  bias_constant_in_kjpermol: 3.0e5 # Bias constant for the Steered MD projection (might require tuning)
  n_steps: 1000
  path_to_initial_states: # (*) Initial states for each replica, useful for continuing optimizations. Optional.
    - ./initial_state_0.xml # MUST BE DIFFERENT FOR EACH REPLICA if provided
    - ./initial_state_1.xml
    - ./initial_state_2.xml
  platform: CUDA
  platform_properties:
    DeviceIndex: '0'
  # platform: CPU # CPU version for defining the platform
  # platform_properties:
  #   Threads: "32" # Can be unreliable, OpenMM does not always use all available threads
  md_params: # (*) Override default OpenMM simulation parameters. All fields are optional.
    forcefield: amber14-all.xml       # (*) Default: amber14-all.xml
    water_model: amber14/tip3p.xml    # (*) Default: amber14/tip3p.xml
    nonbonded_method: CutoffNonPeriodic # (*) Options: PME, CutoffNonPeriodic, NoCutoff, CutoffPeriodic, Ewald, LJPME
    nonbonded_cutoff_nm: 1.0          # (*) Default: 1.0 nm
    constraints: HBonds               # (*) Options: HBonds, AllBonds, HAngles, null (no constraints)
    temperature_K: 300.0              # (*) Default: 300 K
    friction_per_ps: 1.0              # (*) Default: 1.0 ps^-1
    timestep_ps: 0.002                # (*) Default: 0.002 ps (2 fs)

likelihood_optimizer_params:
  batch_size: 50 # Batch size used for computing the log-likelihood in parallel
  initial_weights: # (*) Initial weights, one for each replica. Default is 1/M, M = number of walkers
  - 0.33
  - 0.33
  - 0.33
  n_steps: 10
  step_size: 2.0
  n_batches_per_step: 5 # allows for memory-friendly computation of gradients

atom_selection: ... # path to a txt/npy file, or a mdtraj-compatible selection string, e.g., "not element H"
loads_b_factors: true # Load Debye-Waller b-factors from provided PDBs
n_steps: 100 # Number of ensemble optimization steps

path_to_atomic_models: # Can be the same for each replica
- path/to/atomic_models/initial_model_0.pdb
- path/to/atomic_models/initial_model_1.pdb
- path/to/atomic_models/initial_model_2.pdb

path_to_output: ./optimization_output/round3/
rng_seed: 0 # seed for all RNG operations
```

## Comments on the input parameters

Alignment is crucial. If the structure is not aligned to the frame of reference of the cryo-EM particles, the likelihood will not be computed correctly and the optimization will diverge as the structure is driven toward noise. When possible, include a reference volume for alignment — this is particularly important for heterogeneous systems. Alignment typically adds about 1 second per iteration for volumes with a 32-pixel box size (controllable via `downsample_box_size` in `alignment_params`). A dilated volumetric mask can also help by reducing background noise in the images.

The `path_to_initial_states` argument is useful for restarting simulations or warm-starting from a previously equilibrated MD state. We recommend providing a unique state file for each walker to avoid numerical issues that arise when walkers are indistinguishable.

The optional `md_params` block inside `projector_params` lets you override any of the underlying OpenMM simulation parameters without modifying the code. All fields are optional and fall back to built-in defaults when omitted. The `nonbonded_method` field accepts string aliases: `CutoffNonPeriodic` (default, for implicit/vacuum simulations), `PME` or `Ewald` (for explicit solvent with a periodic box), `CutoffPeriodic` (periodic box with a simple cutoff), `LJPME` (PME for both electrostatics and Lennard-Jones), and `NoCutoff` (no cutoff; very small systems only). Note that periodic methods require a simulation box defined in the PDB. The `constraints` field accepts `HBonds` (default, constrains bonds to hydrogen — compatible with a 2 fs timestep), `AllBonds`, `HAngles` (allows ~4 fs timesteps), or `null` to disable constraints entirely. Numeric fields use explicit unit suffixes: `_nm` for nanometers, `_K` for Kelvin, `_per_ps` for ps⁻¹, and `_ps` for picoseconds. The `platform` and `platform_properties` fields are set separately at the top level of `projector_params`.

## Outputs

- `final_ensemble.npz`: final walker positions and weights.
- `final_walker_*.pdb`: a PDB file for each final walker.
- `traj_walker_*.xtc`: the trajectory followed by each walker during the optimization.
- `states_proj_*/`: a directory containing the OpenMM states at each iteration. Useful for restarting.
- A `log` file.

# Known Issues

## Crashing OpenMM Simulations

This issue occurs rarely and is difficult to reproduce. Occasionally, the steered MD simulation will explode during the optimization. In some cases this is caused by water molecules in the input PDB — make sure to exclude them using the `atom_selection` parameter. In other cases, changing the nonbonded interaction method can resolve the issue; try different values for `nonbonded_method` in the `md_params` block of the config file.

## Corrupted CryoSPARC Datasets

If your dataset contains particles picked in CryoSPARC, we have found that MRC files are sometimes padded with zeros, producing zero-value particles that cause numerical issues (division by zero when computing the likelihood). This can be fixed by preprocessing the dataset; we are working on providing a simple command-line script to do this.
