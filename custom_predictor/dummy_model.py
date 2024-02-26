import torch.nn as nn
import torch
import numpy as np


class DummyModel(nn.Module):
    """
    Dummy model for testing purposes. Returns a tensor of the same shape as the
    input.
    """

    def __init__(self):
        super().__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_shape = input_tensor.shape
        return torch.from_numpy(
            np.random.rand(input_shape[0], input_shape[1], input_shape[2]).astype(
                np.float32
            )
        )
