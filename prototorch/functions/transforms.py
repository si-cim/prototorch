import torch


def gaussian(distance, variance):
    return torch.exp(-(distance * distance) / (2 * variance))
