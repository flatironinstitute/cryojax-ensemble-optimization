<h1 align='center'>Cryo-electron microscopy ensemble optimization using individual particles and physical constraints</h1>


## Summary

CryoJAX ensemble optimization is a module of the [cryoJAX](https://github.com/michael-0brien/cryojax) library, a [JAX](https://github.com/jax-ml/jax) and [Equinox](https://docs.kidger.site/equinox/)-based library for differentiable cryo-EM forward models. The purpose of this library is to provide a framework for optimizing structural ensembles, defined as a weighted discrete set of atomic structures, given a set of cryo-EM images. To do this, we implement an algorithm inspired by projected gradient descent, where the optimization step is performed by comparing the ensemble to the cryo-EM dataset, and the projection step is done through Steered Molecular Dynamics using the popular [OpenMM](https://openmm.org/) library. Details and results are available in our [JCTC paper](https://pubs.acs.org/doi/10.1021/acs.jctc.6c00053). Instructions for reproducing the paper results are provided below.


Full documentation is available at [flatironinstitute.github.io/cryojax-ensemble-optimization](https://flatironinstitute.github.io/cryojax-ensemble-optimization/).

## Installation

Our library has been tested on the latest Ubuntu version. Availability for other platforms is dependent on the availability of OpenMM and JAX.

### CPU Installation

Our library can be installed to be used with a CPU via pip.

```bash
pip install cryojax_eo
```
We recommend using a freshly created virtual environment to install our library. A CPU installation is only recommended for dataset simulation, as OpenMM is built for GPU, and simulations will take a long time if run on CPU.

### GPU Installation

We recommend installing our library using conda (or one of its variants), as matching JAX's and OpenMM's CUDA versions can be difficult otherwise. Here we show an example of how to install our library with [mamba](https://github.com/mamba-org/mamba):

```bash
mamba create -n cryojax_eo_env python==3.11
mamba activate cryojax_eo_env
mamba install -c conda-forge openmm cuda-version==12.4 # Insert your cuda version!
pip install --upgrade "jax[cuda12]"
pip install cryojax_eo
```
To find your CUDA version, you can run `nvidia-smi` in a terminal. The CUDA version will appear in the top right corner of the output.

Note: OpenMM is not required for all our utilities. Data simulation and Ensemble Reweighting only require JAX!

### Apptainer/Singularity Installation (HPC clusters)

For HPC environments where conda is unavailable or installation is restricted, we provide an [Apptainer](https://apptainer.org/) definition file at `container/cryojax_eo.def`. Pre-built images are hosted on [Sylabs Cloud](https://cloud.sylabs.io/library/davidsilva27/default/cryojax_eo).

> **Note:** On most HPC clusters, Apptainer or Singularity must be loaded as an environment module before use: `module load apptainer` (or `module load singularity`). If neither works, check availability with `module spider apptainer` or contact your system administrator.

**Option 1 — Pull a pre-built image:**
```bash
apptainer pull --library https://library.sylabs.io library://davidsilva27/default/cryojax_eo:latest
```

> **Note:** The `--library` flag is required because newer versions of Apptainer/Singularity no longer include Sylabs Cloud as the default endpoint. Replace `apptainer` with `singularity` if that is what your cluster provides.

**Option 2 — Build from the definition file:**
```bash
# Use sudo if available
sudo apptainer build container/cryojax_eo.sif container/cryojax_eo.def

# On HPC clusters where sudo is not available, use --fakeroot instead
apptainer build --fakeroot container/cryojax_eo.sif container/cryojax_eo.def
```

**Running commands with the container:**
```bash
apptainer exec --nv --bind /path/to/data:/path/to/data path/to/container/cryojax_eo.sif \
    run_ensemble_optimization --config config.yaml
```

The `--nv` flag exposes the host GPU to the container. The `--bind` flag mounts a directory from the host into the container, which is required if your data lives outside your home directory (e.g., on a scratch or Ceph filesystem). Replace `/path/to/data` with the relevant path on your system.

To avoid typing `--bind` every time, you can set the following environment variable in your `~/.bashrc`:
```bash
export APPTAINER_BIND=/path/to/data:/path/to/data
```

## Cryo-EM Ensemble Optimization Input

See the [input documentation](docs/ensemble_optimization.md)

## Cryo-EM Ensemble Reweighting Input

See the [input documentation](docs/ensemble_reweighting.md)


## Cryo-EM heterogeneous dataset simulation

See the [input documentation](docs/dataset_simulation.md)


## Reproducing Paper Results

All the necessary data, atomic models, config files, and instructions to reproduce our results are available in [Zenodo](https://doi.org/10.5281/zenodo.19224943) and [Dryad](https://datadryad.org/dataset/doi:10.5061/dryad.j6q573nwc). These results were obtained with version `0.2.3` of `cryojax-eo`.

## Contact

Please submit any bug reports, feature requests, or general feedback as a GitHub issue or discussion.


## Contributing

If you are contributing to this project, please install the package with the following command:

```bash
pip install -e ".[dev]"
```


This will install the required dependencies for development, the most important being `Ruff` and `pre-commit`. After installation, activate your environment and install the `pre-commit` hooks by running

`pre-commit install`

Make sure that your code is formatted according to our guidelines by running:

```bash
pre-commit run --all-files
```

This will make sure the code is formatted correctly, fix whatever can be automatically fixed, and tell you if something else needs to be fixed.

## Acknowledgements
We thank Michael O'Brien, Miro Astore, Lars Dingeldein, Wai Shing Tang, Aaditya Rangan, and Sonya Hanson for helpful discussions.
