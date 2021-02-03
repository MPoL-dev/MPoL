import pytest
import numpy as np

# We need a fixture which provides mock visibilities of the sort we'd
# expect from visread, but *without* the CASA dependency.

# fixture to provide tuple of uu, vv, weight, data_re, and data_im values
@pytest.fixture(scope="session")
def mock_visibility_data():

    # download the npz file to the temporary directory
    return np.load("test/logo_cube.npz")

