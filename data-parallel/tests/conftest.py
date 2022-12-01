import pytest
import numpy as np
from processing.data_management import Dataset

np.random.seed(0)


@pytest.fixture()
def sample_input_data():
    """Returns sample data for tests"""
    sample_data = {
        "features": np.random.rand(12, 2),
        "targets": np.random.randint(2, size=12),
    }

    return Dataset(data=sample_data)
