"""ProtoTorch loss functions."""

import torch


def glvq_loss(distances, target_labels, prototype_labels):
    """GLVQ loss function with support for one-hot labels."""
    matcher = torch.eq(target_labels.unsqueeze(dim=1), prototype_labels)
    if prototype_labels.ndim == 2:
        # if the labels are one-hot vectors
        nclasses = target_labels.size()[1]
        matcher = torch.eq(torch.sum(matcher, dim=-1), nclasses)
    not_matcher = torch.bitwise_not(matcher)

    inf = torch.full_like(distances, fill_value=float('inf'))
    distances_to_wpluses = torch.where(matcher, distances, inf)
    distances_to_wminuses = torch.where(not_matcher, distances, inf)
    dpluses = torch.min(distances_to_wpluses, dim=1, keepdim=True).values
    dminuses = torch.min(distances_to_wminuses, dim=1, keepdim=True).values

    mu = (dpluses - dminuses) / (dpluses + dminuses)
    return mu
