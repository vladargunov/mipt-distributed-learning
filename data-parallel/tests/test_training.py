import torch

from processing import framework


def test_training(sample_input_data):
    """Tests the shape of prediction"""
    test_model = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())
    test_loss = torch.nn.CrossEntropyLoss()

    test_framework = framework.DLFramework(model=test_model, loss=test_loss, lr=1)

    test_framework.prepare_data(dataset=sample_input_data)
    test_framework.train(epochs=3, batch_size=4, validation_frequency=1)

    prediction_shape = test_framework.predict(sample_input_data).shape[0]
    assert prediction_shape == len(sample_input_data), "Prediction shape is incorrect!"
