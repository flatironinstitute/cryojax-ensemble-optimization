# Cryo-EM Ensemble Reweighting

!!! note
    This pipeline does not require OpenMM.

If you already have a set of structures (atomic models or volumes), you can run ensemble reweighting through the command line:

```bash
run_ensemble_reweighting --config config_reweighting.yaml
```

Once likelihoods have been computed, you can re-optimize weights with different `max_iter` or `tol` values without recomputing likelihoods:

```bash
# use the log_likelihood_matrix.npz saved in the output directory from the config
run_ensemble_reweighting --config config_reweighting.yaml --from-likelihoods

# or point to a specific npz file
run_ensemble_reweighting --config config_reweighting.yaml --from-likelihoods /path/to/log_likelihood_matrix.npz
```

This pipeline is more flexible than ensemble optimization: input PDBs do not need to share the same topology, and volumetric files in `.mrc` format are also supported. The main requirement is that all structures are already aligned. Alignment can be done manually in ChimeraX for PDB files, or using the volume alignment tools in CryoSPARC.

During the pipeline, PDB and CIF files are converted to volumes using `cryojax`. Because these volumes are high-resolution, we provide an option to apply a low-pass filter to all input structures. This is recommended when mixing `.pdb`/`.cif` and `.mrc` files.

## Input Format

Cryo-EM Ensemble Reweighting uses a custom YAML input format. A sample [config file](./example_configs/config_reweighting.yaml) for running ensemble reweighting with three structural files is shown below:

```yaml
# Parameters marked with (*) are optional

dataset_params:
  path_to_relion_project: /path/to/relion/project/
  path_to_starfile: /path/to/starfile.star
  loads_envelope: True # (*) Default: false, loads envelope function parameters
  data_sign: "dark-on-light" # Relion convention, dark particles on a light background
  path_to_volumetric_mask: /path/to/volumetric_mask.mrc # (*) e.g., the mask from a refinement procedure

# How many images are used to compute likelihoods in parallel
# This can usually be a large value since Fourier Slice Extraction is memory efficient
n_images_in_parallel: 5000

# Maximum number of iterations for the reweighting algorithm
max_iter: 500

# Convergence tolerance for the multiplicative gradient weight optimization
tol: 1e-4

# ab initio version, ignores starfile poses
# default is false
estimates_poses: true

# A collection of structural files
# can be any combination of pdb, cif, mrc files
path_to_structural_files:
- path/to/structural_files/pdb_structural_file_0.pdb
- path/to/structural_files/cif_structural_file_1.cif
- path/to/structural_files/mrc_structural_file_2.mrc

path_to_output_dir: ./reweighting_results/

########### Atomic model exclusive parameters ###########
# These will be ignored if no structural files have
# pdb or cif format.

# atom selection string in mdtraj convention
# Examples: "all", "not element H", "name CA"
atom_selection: "all"

# Load Debye-Waller b-factors from provided PDBs
loads_b_factors: true
#########################################################

```

## Outputs

- `optimized_weights.yaml`: final weight for each structural file, keyed by filename stem (e.g. `model_0` for `model_0.pdb`).
- `log_likelihood_matrix.npz`: log-likelihood computed for each image–structure pair, keyed by filename stem. Can be reloaded with `--from-likelihoods` to re-optimize weights without rerunning the full pipeline.
- A `log` file.
- A copy of the input config file.
