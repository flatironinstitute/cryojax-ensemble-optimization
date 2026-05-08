# CryoJAX Ensemble Optimization

**Cryo-electron microscopy ensemble optimization using individual particles and physical constraints**

`cryojax_eo` is a module of the [cryoJAX](https://github.com/michael-0brien/cryojax) library, a [JAX](https://github.com/jax-ml/jax) and [Equinox](https://docs.kidger.site/equinox/)-based library for differentiable cryo-EM forward models. The purpose of this library is to provide a framework for optimizing structural ensembles, defined as a weighted discrete set of atomic structures, given a set of cryo-EM images.

To do this, we implement an algorithm inspired by projected gradient descent, where the optimization step is performed by comparing the ensemble to the cryo-EM dataset, and the projection step is done through Steered Molecular Dynamics using the popular [OpenMM](https://openmm.org/) library.

Details and results are available in our [preprint](https://www.biorxiv.org/content/10.64898/2025.12.02.691891v1).

## Capabilities

- **Ensemble Optimization** — optimize a weighted ensemble of atomic structures against cryo-EM particle images using Steered Molecular Dynamics
- **Ensemble Reweighting** — compute optimal weights for an existing set of structures or volumes against cryo-EM images (no OpenMM required)
- **Dataset Simulation** — generate synthetic heterogeneous cryo-EM datasets from multiple atomic models
- **Flexible Fitting** — fit a single atomic model to a consensus density map using steered MD

## Getting started

See the [Installation](installation.md) page, then pick the workflow that matches your use case from the **Usage** section.

## Reproducing paper results

All data, atomic models, config files, and instructions are available on [Zenodo](https://doi.org/10.5281/zenodo.19224943).

## Contact

Please submit bug reports, feature requests, or general feedback as a [GitHub issue](https://github.com/flatironinstitute/cryojax-ensemble-optimization/issues).
