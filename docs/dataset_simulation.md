# Cryo-EM Heterogeneous Dataset Simulation

You can simulate a heterogeneous dataset from multiple PDBs through our API, as shown in our [Tutorial](../tutorial/data_simulation/simulate_and_visualize_data.ipynb), or through the command line:

```bash
simulate_data --config config_simulation.yaml
```

## Input Format

The `simulate_data` command uses a custom YAML input format. A sample [config file](./example_configs/config_simulation.yaml) is shown below:

```yaml
# Parameters marked with (*) are optional
# All parameters that can be defined in a range ([min_value, max_value]) can also be
# defined as a single value (e.g., offset_x_in_angstroms: 0.0)

number_of_images: 20000
batch_size_for_generation: 1000 # images generated in parallel
images_per_file: 5000

data_sign: dark-on-light # Relion convention, dark particles on a light background
box_size: 128
pixel_size: 2.0

offset_x_in_angstroms:
  - min_value
  - max_value
offset_y_in_angstroms:
  - min_value
  - max_value

amplitude_contrast_ratio: 0.1
phase_shift: 0.0
spherical_aberration_in_mm: 2.7
voltage_in_kilovolts: 300.0

ctf_scale_factor: 1.0
astigmatism_angle_in_degrees:
  - min_value
  - max_value
astigmatism_in_angstroms: [min, max]
defocus_in_angstroms:
  - min_value
  - max_value
envelope_b_factor:
  - min_value
  - max_value

# Noise is added by normalizing the variance of the signal.
# The following sets the region in the image used to compute the variance.
# We recommend half the box size (may vary with pixel size and offset range).
mask_radius: 64
mask_rolloff_width: 1.0 # smooths the edge of the mask
noise_snr:
  - min_value
  - max_value

atomic_models_params:
  atom_selection: not element H
  atomic_models_probabilities:
  - 0.7
  - 0.3
  loads_b_factors: true
  path_to_atomic_models: # use of wildcard is permitted, e.g., `path/to/atomic_models/initial_model_*.pdb`
    - path/to/atomic_models/initial_model_0.pdb
    - path/to/atomic_models/initial_model_1.pdb
    - path/to/atomic_models/initial_model_2.pdb
    - ...
overwrite: true

path_to_relion_project: output/directory/for/mrcs/files/
path_to_starfile: path/to/starfile.star

rng_seed: 0 # seed for noise and parameter generation
```

## Comments on the input parameters

When multiple models are provided, all structures are aligned to the first model in the list (or the first in alphabetical order if a wildcard is used). This ensures that no spurious heterogeneity is introduced by misaligned models when the true poses are used for reconstruction. All models must share the same topology. Parameters specified as a range are sampled uniformly over that range.

## Outputs

- `*.mrcs`: particle images in `.mrcs` format.
- A STAR file containing pose and CTF information.
- A copy of the config file, written to the `path_to_relion_project` directory.

## Loading the data

The simulated data can be visualized using the [cryoJAX](https://michael-0brien.github.io/cryojax/) API:

```python
from cryospax import (
    RelionParticleParameterFile,
    RelionParticleStackDataset,
)


relion_dataset = RelionParticleStackDataset(
    RelionParticleParameterFile(path_to_starfile=...),
    path_to_relion_project=...,
)

relion_dataset[0:100]

>>> dict(
    "parameters": dict("pose": ..., "transfer_theory"=..., "image_config"=...),
    "images": ... # Array with images
)

```

The inputs for `RelionParticleStackDataset` should match the corresponding fields in the config file. For more details, see the cryoJAX tutorial on [Loading Cryo-EM Images](https://michael-0brien.github.io/cryojax/examples/read-dataset/).
