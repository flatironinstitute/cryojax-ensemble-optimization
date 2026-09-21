"""Shared helpers for the command-line entry point tests.

The tests in this package are smoke tests: they build a config, write it to a
YAML file, call the command's `main(argparse.Namespace(...))`, and assert that
the expected output was produced. The point is to check that every documented
config option still parses and still wires through, not to check numerics.

Options whose machinery may be missing on a given machine (the Pallas/Triton
kernels need a CUDA GPU, OpenMM's CUDA and OpenCL platforms need their plugins)
are wrapped in `skip_if_unavailable`, which turns "not available here" into a
skip instead of a failure.
"""

import argparse
import contextlib
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# Optional machinery
# --------------------------------------------------------------------------- #


# (exception type, message pattern) pairs. A failure is downgraded to a skip only
# if both its type and its message match. Anything else propagates, so a genuine
# regression still fails the suite instead of disappearing into a skip.
def _unavailable_signatures():
    try:
        from openmm import OpenMMException
    except ImportError:  # openmm is an optional dependency
        OpenMMException = None

    signatures = [
        # cryojax.ndimage._spreading.pallas_spread.resolve_enable_pallas, raised
        # at trace time when enable_pallas is on and there is no GPU backend.
        (RuntimeError, re.compile(r"`enable_pallas` requires a CUDA GPU")),
        # openmm.app.ForceField, for an XML this build does not ship.
        (ValueError, re.compile(r'Could not locate file "')),
    ]
    if OpenMMException is not None:
        signatures += [
            # Platform plugin never registered.
            (
                OpenMMException,
                re.compile(r'There is no registered Platform called "[^"]+"'),
            ),
            # Platform registered, but no usable device or driver.
            (
                OpenMMException,
                re.compile(r"Error (initializing|loading|creating) (CUDA|OpenCL|HIP)"),
            ),
            (
                OpenMMException,
                re.compile(r"No compatible (CUDA|OpenCL) device is available"),
            ),
            (
                OpenMMException,
                re.compile(r"CUDA-capable device|CUDA_ERROR_|libcuda|libOpenCL"),
            ),
            # A periodic nonbonded method on a topology with no unit cell.
            (OpenMMException, re.compile(r"periodic box|PeriodicBoxVectors")),
        ]
    return tuple(signatures)


def _exception_chain(exc):
    """Yield `exc` and everything it was raised from or during, once each."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


@contextlib.contextmanager
def skip_if_unavailable(what: str):
    """Run a block that needs machinery this machine may not have.

    Turns a recognized "this backend is not available here" failure into a
    `pytest.skip`, and re-raises anything else untouched.

    **Arguments:**

    - `what`:
        Short description used in the skip message, e.g. `"the Pallas backend"`.
    """
    try:
        yield
    except Exception as exc:
        for err in _exception_chain(exc):
            for exc_type, pattern in _unavailable_signatures():
                if isinstance(err, exc_type) and pattern.search(str(err)):
                    pytest.skip(
                        f"{what} is not available here: {type(err).__name__}: {err}"
                    )
        raise


def openmm_platform_available(name: str) -> bool:
    try:
        from openmm import Platform

        Platform.getPlatformByName(name)
    except Exception:
        return False
    return True


# --------------------------------------------------------------------------- #
# Config plumbing
# --------------------------------------------------------------------------- #

#: Sentinel for `deep_update`: drop the key from the config entirely. Needed for
#: options whose "off" state is the key being absent rather than being null.
REMOVE = object()


class Replace:
    """Wrap a dict in a `deep_update` override to replace it, not merge into it.

    Needed for free-form dicts such as `platform_properties`, where merging the
    override into the base would leave the base's keys behind and produce a
    combination neither the base nor the override asked for.
    """

    def __init__(self, value):
        self.value = value


def deep_update(base: dict, overrides: dict) -> dict:
    """Recursively merge `overrides` into a copy of `base`.

    A value of `REMOVE` deletes the key; a value wrapped in `Replace` overwrites
    it instead of being merged into it.
    """
    merged = dict(base)
    for key, value in overrides.items():
        if value is REMOVE:
            merged.pop(key, None)
        elif isinstance(value, Replace):
            merged[key] = value.value
        elif isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def write_config(tmp_path, config: dict, name: str = "config.yaml") -> str:
    path = str(tmp_path / name)
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


def run_command(main_fn, tmp_path, config: dict, *, name="config.yaml", **namespace):
    """Write `config` to YAML and invoke a command's `main`."""
    namespace.setdefault("config", write_config(tmp_path, config, name))
    return main_fn(argparse.Namespace(**namespace))


# --------------------------------------------------------------------------- #
# Generated artifacts
#
# Session scoped and built from `DATA_DIR` directly rather than from the
# function-scoped path fixtures in `tests/conftest.py`, so that a session-scoped
# fixture is not asked to depend on a function-scoped one.
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def openmm_state_xml(tmp_path_factory):
    """An OpenMM state file, as `SteeredMDSimulator.initialize` would write it.

    Used for `projector_params.path_to_initial_state(s)`. Built directly rather
    than through `SteeredMDSimulator.initialize`, which also runs 1000 MD steps.
    """
    openmm = pytest.importorskip("openmm")
    openmm_app = pytest.importorskip("openmm.app")
    openmm_unit = pytest.importorskip("openmm.unit")

    out_dir = tmp_path_factory.mktemp("openmm_state")
    path = str(out_dir / "initial_state.xml")

    with skip_if_unavailable("OpenMM"):
        pdb = openmm_app.PDBFile(str(DATA_DIR / "ala_model_0.pdb"))
        forcefield = openmm_app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=openmm_app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * openmm_unit.nanometer,
            constraints=openmm_app.HBonds,
        )
        integrator = openmm.LangevinIntegrator(
            300.0 * openmm_unit.kelvin,
            1.0 / openmm_unit.picosecond,
            0.002 * openmm_unit.picoseconds,
        )
        simulation = openmm_app.Simulation(
            pdb.topology,
            system,
            integrator,
            openmm.Platform.getPlatformByName("CPU"),
            {"Threads": "1"},
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy(maxIterations=10)
        simulation.saveState(path)

    return path


@pytest.fixture(scope="session")
def _heavy_atom_indices():
    import mdtraj

    topology = mdtraj.load(str(DATA_DIR / "ala_model_0.pdb")).topology
    return np.asarray(topology.select("not element H"), dtype=int)


@pytest.fixture(scope="session")
def atom_indices_txt(tmp_path_factory, _heavy_atom_indices):
    path = str(tmp_path_factory.mktemp("atom_indices") / "indices.txt")
    np.savetxt(path, _heavy_atom_indices, fmt="%d")
    return path


@pytest.fixture(scope="session")
def atom_indices_npy(tmp_path_factory, _heavy_atom_indices):
    path = str(tmp_path_factory.mktemp("atom_indices_npy") / "indices.npy")
    np.save(path, _heavy_atom_indices)
    return path


@pytest.fixture(scope="session")
def pdb_glob_pattern(tmp_path_factory):
    """A glob string matching two atomic models, for `path_to_atomic_models`."""
    out_dir = tmp_path_factory.mktemp("pdb_glob")
    for name in ("ala_model_0.pdb", "ala_model_1.pdb"):
        shutil.copyfile(DATA_DIR / name, out_dir / name)
    return os.path.join(str(out_dir), "*.pdb")


@pytest.fixture(scope="session")
def distinct_pdb_copies(tmp_path_factory, map_frame_pdb):
    """Three copies of the same model under distinct names.

    `run_ensemble_reweighting` keys its results by `Path(file).stem`, so passing
    the same path several times silently collapses them into one entry. The
    copies are in map coordinates because that command, like
    `run_ensemble_optimization`, shifts its structures by `-L/2`; see
    `map_frame_pdb`.
    """
    out_dir = tmp_path_factory.mktemp("distinct_pdbs")
    paths = []
    for name in ("model_a.pdb", "model_b.pdb", "model_c.pdb"):
        dest = out_dir / name
        shutil.copyfile(map_frame_pdb, dest)
        paths.append(str(dest))
    return paths


@pytest.fixture(scope="session")
def mse_weights_mrc(tmp_path_factory):
    """Weights for `ModelToVolumeWeightedMSELossFn`.

    That loss multiplies the weights by an rfft-shaped array, so they are
    half-Fourier shaped `(dim, dim, dim // 2 + 1)` rather than a full cube, and
    must be non-negative because the product is square-rooted.
    """
    import mrcfile

    path = str(tmp_path_factory.mktemp("mse_weights") / "weights.mrc")
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.ones((32, 32, 17), dtype=np.float32))
    return path


@pytest.fixture(scope="session")
def c2_beads_pdb(tmp_path_factory):
    """`ala_model_0.pdb` with its alpha carbons renamed to `C2`.

    `GMMFitConfig.fit_selection_string` defaults to `name "C2"`, which selects
    nothing in either of the bundled protein models, so the branch of
    `fit_gmm_to_atoms` that runs without a config file needs its own input. The
    other atoms are kept: the target density is built from every heavy atom
    while only the `C2` beads are fitted, which is the coarse-graining the
    command is for. Fitting beads against a target built from those same beads
    makes the Gauss-Newton solve singular.
    """
    import mdtraj

    structure = mdtraj.load(str(DATA_DIR / "ala_model_0.pdb"))
    for atom in structure.topology.atoms:
        if atom.name == "CA":
            atom.name = "C2"

    path = str(tmp_path_factory.mktemp("c2_beads") / "beads.pdb")
    structure.save_pdb(path)
    return path


@pytest.fixture(scope="session")
def simulated_particle_stack(tmp_path_factory):
    """A small particle stack with a realistic pixel size, simulated once.

    The checked-in stack in `tests/data/particle_stack` has a 0.2 angstrom pixel
    size. At that sampling the spread width that `spread_mode='local'` derives
    from the scattering variances exceeds the integrator grid, so the *default*
    volume integrator backend cannot run against it. Simulating a stack at 2.0
    angstroms keeps the default config usable.

    **Returns:** a dict with `path_to_starfile`, `path_to_relion_project`,
    `box_size`, `pixel_size` and `number_of_images`.
    """
    from cryojax_eo.commands._simulate_data import main as simulate_main

    out_dir = tmp_path_factory.mktemp("simulated_stack")
    project = str(out_dir / "relion_project")
    starfile = os.path.join(project, "particles.star")
    config = {
        "number_of_images": 5,
        "data_sign": "light-on-dark",
        "pixel_size": 2.0,
        "box_size": 32,
        "noise_snr": 0.1,
        "rng_seed": 0,
        "atomic_models_params": {
            "path_to_atomic_models": [
                str(DATA_DIR / "ala_model_0.pdb"),
                str(DATA_DIR / "ala_model_1.pdb"),
            ],
            "atomic_models_probabilities": [0.5, 0.5],
            "loads_b_factors": True,
            "atom_selection": "not element H",
        },
        "path_to_relion_project": project,
        "path_to_starfile": starfile,
        "images_per_file": 5,
        "batch_size_for_generation": 5,
        "overwrite": True,
    }
    run_command(simulate_main, out_dir, config, name="simulate_stack.yaml")
    return {
        "path_to_starfile": starfile,
        "path_to_relion_project": project,
        "box_size": 32,
        "pixel_size": 2.0,
        "number_of_images": 5,
    }


@pytest.fixture(scope="session")
def map_frame_pdb(tmp_path_factory, simulated_particle_stack):
    """`ala_model_0.pdb` expressed in map coordinates.

    `run_ensemble_optimization` shifts the prealigned model by `-L/2` to move it
    from a map's frame (box corner at the origin, as cryoSPARC and RELION write
    it) into cryojax's box-centered frame. The bundled PDBs are already centered
    on the origin, so applying that shift to one of them directly would put the
    model at the corner of the box, outside the imaged region. Placing it at
    `+L/2` first is what an actual map-aligned model looks like.
    """
    import mdtraj

    box_length_in_angstroms = (
        simulated_particle_stack["box_size"] * simulated_particle_stack["pixel_size"]
    )
    shift_in_nanometers = (box_length_in_angstroms / 2) / 10.0

    structure = mdtraj.load(str(DATA_DIR / "ala_model_0.pdb")).center_coordinates()
    structure.xyz = structure.xyz + shift_in_nanometers

    path = str(tmp_path_factory.mktemp("map_frame_pdb") / "prealigned_model.pdb")
    structure.save_pdb(path)
    return path
