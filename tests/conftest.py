import os

import jax
import pytest


jax.config.update("jax_enable_x64", True)


@pytest.fixture
def sample_path_to_pdb1():
    return os.path.join(os.path.dirname(__file__), "data", "ala_model_0.pdb")


@pytest.fixture
def sample_path_to_pdb2():
    return os.path.join(os.path.dirname(__file__), "data", "ala_model_1.pdb")


@pytest.fixture
def sample_path_gmm_model():
    return os.path.join(os.path.dirname(__file__), "data", "test_groel_gmm_model.npz")


@pytest.fixture
def sample_path_mrc_file():
    return os.path.join(os.path.dirname(__file__), "data", "test_groel_volume.mrc")


@pytest.fixture
def sample_path_groel_pdb():
    return os.path.join(os.path.dirname(__file__), "data", "apo_groel.pdb")
