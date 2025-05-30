<h1 align='center'>Physics-Constrained Cryo-Electron Microscopy Ensemble Optimizaiton</h1>


## Description

TODO


## Installation

Before installing, use your prefered virtual environment manager to initialize a virtual env. We recommend `uv`, although any manager should work.

```bash
uv venv my-venv-name --python 3.11
```
Don't forget to activate your environment!

Most dependencies are installed automatically when you install the package. However, the current version of the package used the `dev` branch of `cryoJAX`, and for this reason it needs to be installed manually. To do this, clone the cryoJAX repo, switch branches and install

```bash
git clone git@github.com:mjo22/cryojax.git
cd cryojax
git checkout dev
pip install .
```

In addition, although OpenMM is a required dependency, it is not installed automatically as its installation might require specific steps. If you are not worried about the OpenMM installation, you can install it directly from PyPI using pip:
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


## Acknowledgements
TODO
