<h1 align='center'>Physics-Constrained Cryo-Electron Microscopy Ensemble Optimizaiton</h1>


## Description

TODO


## Installation

Before installing, use your prefered virtual environment manager to initialize a virtual env. We recommend `uv`, although any manager should work.

```bash
uv venv my-venv-name --python 3.11
```
Don't forget to activate your environment!

Most dependencies are installed automatically when you install the package. However, although OpenMM is a required dependency, it is not installed automatically as its installation might require specific steps. If you are not worried about the OpenMM installation, you can install it directly from PyPI using pip:
```bash
pip install openmm
```
OpenMM can also be installed using conda, or from source. For more information, see the [OpenMM installation instructions](http://docs.openmm.org/development/userguide/application/01_getting_started.html#installing-openmm).

Lastly, install cryoJAX Ensemble Optimization package by cloning this repo:
```bash
git clone git@github.com:DSilva27/cryojax-ensemble-optimization.git
cd cryojax_ensemble_optimization
pip install .
```

If you intent to use a GPU for JAX operations you might need to install a cuda supported version of JAX manually. We recommend following the official [install JAX](https://github.com/google/jax#installation) instructions.

## Contributing

If you are contributing to this project please install the package with the following command

```bash
pip install -e ".[dev]"
```

This will install the required dependencies for development, the most important being `Ruff` and `pre-commit`. After installation activate your environment and install the `pre-commit` hooks by running

`pre-commit install`

Make sure that your code is formatted according to our guidelines by running:

```bash
pre-commit run --all-times
```

This will make sure the code is formatted correcly, fix whatever can be automatically fixed, and tell you if something else needs to be fixed.

## Acknowledgements
TODO
