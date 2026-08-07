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

For HPC environments where conda is unavailable or installation is restricted, we provide an [Apptainer](https://apptainer.org/) definition file at `container/cryojax_eo.def`. We do not distribute pre-built images — build your own from the definition file, which also lets you match the CUDA version to your cluster's driver.

!!! note "Loading Apptainer/Singularity on HPC clusters"
    On most HPC clusters, Apptainer or Singularity is available as an environment module and must be loaded before use:
    ```bash
    module load apptainer   # or: module load singularity
    ```
    If neither works, check what is available with `module spider apptainer` or `module spider singularity`, or contact your system administrator.

**Step 1 — Check your CUDA version.**

Run `nvidia-smi` — the driver's CUDA version appears in the top-right corner of the output. The definition file pins `cuda-version==12.6`. If your driver is older, edit this line in `container/cryojax_eo.def` before building:

```
$MAMBA install -n cryojax_eo -c conda-forge openmm "cuda-version==12.6" -y
```

**Step 2 — Build the image.**

```bash
# Use sudo if available
sudo apptainer build container/cryojax_eo.sif container/cryojax_eo.def

# On HPC clusters where sudo is not available, use --fakeroot instead
apptainer build --fakeroot container/cryojax_eo.sif container/cryojax_eo.def
```

The build pulls `cryojax_eo` from the `main` branch on GitHub. To pin a specific release, change the `pip install` line in the definition file to reference a tag, for example:

```
"cryojax_eo @ git+https://github.com/flatironinstitute/cryojax-ensemble-optimization.git@v0.1.0"
```

!!! note
    If your cluster provides Singularity rather than Apptainer, replace `apptainer` with `singularity` in all commands.

    Building requires network access and a few GB of scratch space. If your compute nodes are offline, build on a login node or set `APPTAINER_TMPDIR` to a filesystem with enough room.

**Step 3 — Run commands with the container:**

```bash
apptainer exec --nv --bind /path/to/data:/path/to/data container/cryojax_eo.sif \
    run_ensemble_optimization --config config.yaml
```

The `--nv` flag exposes the host GPU to the container. The `--bind` flag mounts a directory from the host into the container, required when your data lives outside your home directory (e.g., on a scratch or Ceph filesystem).

To avoid typing `--bind` on every invocation, set this in your `~/.bashrc`:

```bash
export APPTAINER_BIND=/path/to/data:/path/to/data
```
