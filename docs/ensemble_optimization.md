# Cryo-EM Ensemble Optimization Input

You can run the ensemble optimization through our API, as shown in our [Tutorial](../tutorial/ensemble_optimization/running_the_optimization.ipynb), or through the command line:

```bash
run_ensemble_optimization --config config_optimization.yaml
```

## Input Format

Cryo-EM Ensemble Optimization uses a custom YAML input format. An example for a [config file](./example_configs/config_optimization.yaml) for running the ensemble optimization using CUDA for three walkers is given below:

```yaml
# Parameters marked with (*) are optional

dataset_params:
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
  bias_proportion: 0.01 # proportion between the norm of the bias force and MD force, i.e., proportion = ||F_bias|| / ||F_MD||. Recommended to try 0.01 first, and increase to no more than 0.1
  bias_constant_in_kjmol: ... # (*) overrides bias_proportion, useful to save time when repeating computations.
  n_steps: 1000
  path_to_initial_states: # (*) Initial states for each replica, useful for continuing optimizations. Optional.
    - ./initial_state_0.xml # MUST BE DIFFERENT FOR EACH REPLICA if provided
    - ./initial_state_1.xml
    - ./initial_state_2.xml
  platform: CUDA
  platform_properties:
    DeviceIndex: '0'
# platform: CPU # CPU Version for defining the platform
# Threads: "32" # Can be unreliable, OpenMM does not always use all the available threads

likelihood_optimizer_params:
  batch_size: 50 # Batch size used for computing the log-likelihood in parallel
  initial_weights: # (*) Initial weights, one for each replica. Default is 1/M, M = number of walkers
  - 0.33
  - 0.33
  - 0.33
  n_steps: 10
  step_size: 2.0
  n_batches_per_step: 5 # Allows for a memory-friendly way to compute more batches per step

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

First, alignment is crucial. Suppose during the optimization process, the structure is not aligned to the frame of reference of the cryo-EM particles. In that case, the likelihood won't be computed correctly, and the optimization will explode as the structure gets optimized towards noise. When possible, include a reference volume for alignment. This is particularly important for heterogeneous systems. This usually results in a `1s` delay to each iteration for volumes with a 32-pixel box size (you can downsample the volume through the `downsample_box_size` in `alignment params`). A dilated volumetric mask can also significantly help the optimization by helping reduce the overall noise in the images.

The `path_to_initial_states` argument helps restart simulations or start from a previously equilibrated MD simulation. We recommend that each walker have a unique state file to avoid numerical issues with indistinguishable walkers.

## Outputs

- `final_ensemble.npz`: final walker positions and weights.
- `final_walker_*.pdb`, a PDB file for each final walker.
- `traj_walker_*.xtc`, the trajectory followed by each walker during the optimization.
- `states_proj_*/`, a directory with the OpenMM states at each iteration of the optimization. Useful for restarting.
- A `log` file.



# Known Issues


## Crashing OpenMM Simulations

This is still an issue we are investigating, as it happens very rarely, and it is difficult to reproduce. Basically, sometimes the steered MD simulation with OpenMM will explore. Weirdly, this is solved by applying a random rotation and translation to the initial PDB. We suspect this is caused by the `RMSD 'energy term exploding when the structures are too close (division by zero during the gradient computation). We're currently discussing this with one of the OpenMM developers, and hopefully the fix will be simple.


## Corrupted CryoSPARC datasets

If your dataset contains picked Particles from CryoSPARC, we have found that sometimes the MRCs are padded with zeros, effectively containing zero-value particles that can cause numerical issues (division by zero when computing the likelihood). This can be easily fixed by preprocessing the dataset; we are working on providing a simple command-line script to do this.
