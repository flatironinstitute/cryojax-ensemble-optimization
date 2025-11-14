<h1 align='center'>Cryo-electron microscopy ensemble optimization using individual particles and physical constraints</h1>


## Summary

CryoJAX ensemble optimization is a module of the [cryoJAX](https://github.com/michael-0brien/cryojax) library, a [JAX](https://github.com/jax-ml/jax) and [Equinox](https://docs.kidger.site/equinox/)-based library for differentiable cryo-EM forward models. The purpose of this library is to provide a framework for optimizing structural ensembles, defined as a weighted discrete set of atomic structures, given a set of cryo-EM images. To do this, we implement an algorithm inspired by projected gradient descent, where the optimization step is performed by comparing the ensemble to the cryo-EM dataset, and the projection step is done through Steered Molecular Dynamics using the popular [OpenMM](https://openmm.org/) library. Details and results are available in our preprint: TODO. Instructions for reproducing the paper results are provided below.


## Installation

Our library has been tested on the latest Ubuntu version. Availability for other platforms is dependent on the availability of OpenMM and JAX.

### CPU Installation

Our library can be installed to be used with a CPU via pip.

```bash
pip install git+git@github.com:flatironinstitute/cryojax-ensemble-optimization.git
```
We recommend using a freshly created virtual environment to install our library. A CPU installation is only recommended for dataset simulation, as OpenMM is built for GPU, and simulations will take a long time if run on CPU.

### GPU Installation

We recommend installing our library using conda (or one of its variants), as matching JAX's and OpenMM's CUDA versions can be difficult otherwise. Here we show an example of how to install our library with [mamba](https://github.com/mamba-org/mamba):

```bash
mamba create -n cryojax_eo_env python==3.11
mamba activate cryojax_eo_env
mamba install -c conda-forge openmm cuda-version==12.4 # Insert your cuda version!
pip install --upgrade "jax[cuda12]"
pip install git+https://github.com/flatironinstitute/cryojax-ensemble-optimization.git
```
To find your CUDA version, you can run `nvidia-smi` in a terminal, and the CUDA version will appear in the top right corner of the output.


## Cryo-EM Ensemble Optimization Input

See the [input documentation](docs/ensemble_optimization.md)


## Cryo-EM heterogeneous dataset simulation

See the [input documentation](docs/dataset_simulation.md)


## Reproducing Paper Results (TODO)

All the necessary data, atomic models, config files, and instructions to reproduce our results are available in Zenodo.

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
