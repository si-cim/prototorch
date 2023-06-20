"""ProtoTorch similarities."""

import torch

from .distances import euclidean_distance


def gaussian(x, variance=1.0):
    return torch.exp(-(x * x) / (2 * variance))


def euclidean_similarity(x, y, variance=1.0):
    distances = euclidean_distance(x, y)
    similarities = gaussian(distances, variance)
    return similarities


def cosine_similarity(x, y):
    """Compute the cosine similarity between :math:`x` and :math:`y`.

    Expected dimension of x is 2.
    Expected dimension of y is 2.
    """
    x, y = (arr.view(arr.size(0), -1) for arr in (x, y))
    norm_x = x.pow(2).sum(1).sqrt()
    norm_y = y.pow(2).sum(1).sqrt()
    norm_mat = norm_x.unsqueeze(-1) @ norm_y.unsqueeze(-1).T
    epsilon = torch.finfo(norm_mat.dtype).eps
    norm_mat.clamp_(min=epsilon)
    similarities = (x @ y.T) / norm_mat
    return similarities
