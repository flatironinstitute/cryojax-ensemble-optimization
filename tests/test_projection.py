import os
import shutil

import mdtraj
import numpy as np

from cryojax_eo.ensemble_optimization import (
    EnsembleSteeredMDSimulator,
    SteeredMDSimulator,
)


def make_steered_md_simulator(path_to_pdb):
    model = mdtraj.load(path_to_pdb)
    atom_list = model.topology.select("not element H")

    return SteeredMDSimulator(
        path_to_pdb,
        n_steps=10,
        restrain_atom_list=atom_list,
        parameters_for_md={"platform": "CPU", "properties": {"Threads": "4"}},
        base_state_file_path=os.path.join(
            os.path.dirname(__file__), "outputs/md_states", "state_it"
        ),
    )


def test_steered_md_simulator(sample_path_to_pdb1):
    model = mdtraj.load(sample_path_to_pdb1)
    ref_positions = model.xyz[0] * 10.0

    simulator = make_steered_md_simulator(sample_path_to_pdb1)
    state = simulator.initialize()
    new_positions, state = simulator(ref_positions, state, 10.0)

    assert ref_positions.shape == new_positions.shape, (
        "Positions after projection have the wrong shape"
    )
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "outputs/"))
    return


def test_ensemble_steered_md_simulator(sample_path_to_pdb1, sample_path_to_pdb2):
    model1 = mdtraj.load(sample_path_to_pdb1)
    model2 = mdtraj.load(sample_path_to_pdb2)
    ref_positions = np.stack([model1.xyz[0], model2.xyz[0]], axis=0) * 10.0

    simulator = EnsembleSteeredMDSimulator(
        [
            make_steered_md_simulator(sample_path_to_pdb1),
            make_steered_md_simulator(sample_path_to_pdb2),
        ]
    )
    state = simulator.initialize()
    new_positions, state = simulator(ref_positions, state, 10.0)

    assert ref_positions.shape == new_positions.shape, (
        "Positions after projection have the wrong shape"
    )
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "outputs/"))
    return
