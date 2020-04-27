"""ProtoTorch loss functions."""

import torch


def _get_dp_dm(distances, targets, plabels):
    matcher = torch.eq(targets.unsqueeze(dim=1), plabels)
    if plabels.ndim == 2:
        # if the labels are one-hot vectors
        nclasses = targets.size()[1]
        matcher = torch.eq(torch.sum(matcher, dim=-1), nclasses)
    not_matcher = torch.bitwise_not(matcher)

    inf = torch.full_like(distances, fill_value=float('inf'))
    d_matching = torch.where(matcher, distances, inf)
    d_unmatching = torch.where(not_matcher, distances, inf)
    dp = torch.min(d_matching, dim=1, keepdim=True).values
    dm = torch.min(d_unmatching, dim=1, keepdim=True).values
    return dp, dm


def glvq_loss(distances, target_labels, prototype_labels):
    """GLVQ loss function with support for one-hot labels."""
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = (dp - dm) / (dp + dm)
    return mu
