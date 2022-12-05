import os
import subprocess
from pathlib import Path

import torch

from processing import framework
import tests


def test_save_load():
    """Tests the correctness of save and load of the model"""
    test_model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    test_loss = torch.nn.CrossEntropyLoss()

    test_framework = framework.DLFramework(model=test_model, loss=test_loss, lr=1)

    # Get model directory
    model_dir = (
        Path(tests.__file__).resolve().parent.parent / "models" / "test_model.pt"
    )
    test_framework.save(model_dir)
    assert model_dir.is_file(), "The model was not correctly saved!"
    # Delete created model
    subprocess.run(f"rm {model_dir}".split(), check=True)
