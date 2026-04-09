# Cryo-EM Ensemble Reweighting Input

Note: This does NOT require OpenMM to be installed.

If you already have a set of structures (atomic models or volumes), you can run ensemble reweighting through the command line:

```bash
run_ensemble_reweighting --config config_reweighting.yaml
```

This pipeline is relatively more flexible, as PDBs do not need to have the same topology, and volumes in .mrc format are also supported. The main limitation, however, is that we assume these models are already aligned. This can be easily done manually in Chimera for PDBs, or through the volume alignment tools in CryoSPARC.

During the pipeline PDB and CIF files will be converte through volumes using `cryojax`. Since these volumes will be high resolution, we provide an option to apply a low-pass filter to all input structures. This is recommended when performing reweighting between .pdb/.cif and .mrc files.

## Input Format

Cryo-EM Ensemble Reweighting uses a custom YAML input format. An example for a [config file](./example_configs/config_reweighting.yaml) for running the ensemble optimization using CUDA for three structural files is given below:

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

# Maximum number of iterations for
# reweighting algorithm
max_iter: 500

# Random seed for splitting data into
# train and validation set
random_seed: 0

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

- `optimized_weights`: final weights for each structural file
- A `log` file.
- A copy of the input config file
- log-likelihood matrix: log-likelihood computed for each pair of image, structural file
