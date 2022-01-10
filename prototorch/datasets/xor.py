"""Exclusive-or (XOR) dataset for binary classification."""

import torch


def make_xor(num_samples=500):
    x = torch.rand(num_samples, 2)
    y = torch.zeros(num_samples)
    y[torch.logical_and(x[:, 0] > 0.5, x[:, 1] < 0.5)] = 1
    y[torch.logical_and(x[:, 1] > 0.5, x[:, 0] < 0.5)] = 1
    return x, y


class XOR(torch.utils.data.TensorDataset):
    """Exclusive-or (XOR) dataset for binary classification."""

    def __init__(self, num_samples: int = 500):
        x, y = make_xor(num_samples)
        super().__init__(x, y)
