===========
Cryo-MD
===========


Description
===========
Cryo-MD is a python package for TODO


Requirements
===========
Cryo-MD requires the following packages:
- Jax
- Numpy
- MDAnalysis
- OpenMM


Setting-up Cluster Environment
==============================

First load the required modules
::
    module load python
    module load modules/2.1-20230203  openmpi/4.0.7
    module load openmm/7.7.0

Then create a virtual environment using venv
::

    python3 -m venv --system-site-packages /path/to/venvs/cryo_md_env

Then activate the environment
::

    source /path/to/venvs/cryo_md_env/bin/activate

Finally install the required packages
::

    pip install jax jaxlib numpy mdanalysis

If you will use jupyter notebooks, then you must also create a kernel for the environment
::
    module load jupyter-kernels
    python -m make-custom-kernel cryo_md_kernel

Note: you must do this AFTER loading modules and activating the environment, otherwise the kernel will not be able to find the packages (including OpenMM).


Installation
============
To install, first clone this repository and switch to the branch develop
::

    git clone git@github.com:DSilva27/cryo_MD.git
    cd cryo_MD
    git checkout develop

Then install using pip (use -e for editable mode)
::

    pip install -e .


Running the Tests
=================
Tests can be run in your current environment using pytest, (TODO)

::

    pytest tests/test.py