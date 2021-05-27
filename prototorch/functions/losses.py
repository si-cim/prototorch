"""ProtoTorch loss functions."""

import torch


def _get_matcher(targets, labels):
    """Returns a boolean tensor."""
    matcher = torch.eq(targets.unsqueeze(dim=1), labels)
    if labels.ndim == 2:
        # if the labels are one-hot vectors
        num_classes = targets.size()[1]
        matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
    return matcher


def _get_dp_dm(distances, targets, plabels, with_indices=False):
    """Returns the d+ and d- values for a batch of distances."""
    matcher = _get_matcher(targets, plabels)
    not_matcher = torch.bitwise_not(matcher)

    inf = torch.full_like(distances, fill_value=float("inf"))
    d_matching = torch.where(matcher, distances, inf)
    d_unmatching = torch.where(not_matcher, distances, inf)
    dp = torch.min(d_matching, dim=-1, keepdim=True)
    dm = torch.min(d_unmatching, dim=-1, keepdim=True)
    if with_indices:
        return dp, dm
    return dp.values, dm.values


def glvq_loss(distances, target_labels, prototype_labels):
    """GLVQ loss function with support for one-hot labels."""
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = (dp - dm) / (dp + dm)
    return mu


def lvq1_loss(distances, target_labels, prototype_labels):
    """LVQ1 loss function with support for one-hot labels.

    See Section 4 [Sado&Yamada]
    https://papers.nips.cc/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = dp
    mu[dp > dm] = -dm[dp > dm]
    return mu


def lvq21_loss(distances, target_labels, prototype_labels):
    """LVQ2.1 loss function with support for one-hot labels.

    See Section 4 [Sado&Yamada]
    https://papers.nips.cc/paper/1995/file/9c3b1830513cc3b8fc4b76635d32e692-Paper.pdf
    """
    dp, dm = _get_dp_dm(distances, target_labels, prototype_labels)
    mu = dp - dm

    return mu
