import pytest
import os
import jax
jax.config.update("jax_enable_x64", True)

@pytest.fixture
def sample_path_to_pdb1():
    return os.path.join(os.path.dirname(__file__), "data", "ala_model_0.pdb")

@pytest.fixture
def sample_path_to_pdb2():
    return os.path.join(os.path.dirname(__file__), "data", "ala_model_1.pdb")