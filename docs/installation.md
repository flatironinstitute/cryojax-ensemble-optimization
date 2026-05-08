# Installation

`cryojax_eo` has been tested on the latest Ubuntu version. Availability for other platforms is dependent on the availability of OpenMM and JAX.

## CPU Installation

For dataset simulation and ensemble reweighting (no OpenMM required), install via pip:

```bash
pip install cryojax_eo
```

We recommend using a freshly created virtual environment. A CPU installation is only recommended for dataset simulation and ensemble reweighting — OpenMM simulations will be very slow on CPU.

## GPU Installation

We recommend installing with conda (or one of its variants), as matching JAX's and OpenMM's CUDA versions can be difficult otherwise. Here we show an example using [mamba](https://github.com/mamba-org/mamba):

```bash
mamba create -n cryojax_eo_env python==3.11
mamba activate cryojax_eo_env
mamba install -c conda-forge openmm cuda-version==12.4 # Insert your CUDA version!
pip install --upgrade "jax[cuda12]"
pip install cryojax_eo
```

To find your CUDA version, run `nvidia-smi` — the version appears in the top-right corner of the output.

!!! note
    OpenMM is not required for all workflows. Dataset simulation and ensemble reweighting only require JAX.

## Apptainer/Singularity Installation (HPC clusters)

For HPC environments where conda is unavailable or installation is restricted, we provide an [Apptainer](https://apptainer.org/) definition file at `container/cryojax_eo.def`. Pre-built images are available as assets on the [GitHub Releases](https://github.com/flatironinstitute/cryojax-ensemble-optimization/releases) page.

**Option 1 — Download a pre-built image:**

```bash
wget https://github.com/flatironinstitute/cryojax-ensemble-optimization/releases/download/vX.X/cryojax_eo.sif
```

**Option 2 — Build from the definition file:**

```bash
apptainer build --fakeroot container/cryojax_eo.sif container/cryojax_eo.def
```

**Running commands with the container:**

```bash
apptainer exec --nv --bind /path/to/data:/path/to/data container/cryojax_eo.sif \
    run_ensemble_optimization --config config.yaml
```

The `--nv` flag exposes the host GPU to the container. The `--bind` flag mounts a directory from the host into the container, required when your data lives outside your home directory (e.g., on a scratch or Ceph filesystem).

To avoid typing `--bind` on every invocation, set this in your `~/.bashrc`:

```bash
export APPTAINER_BIND=/path/to/data:/path/to/data
```
