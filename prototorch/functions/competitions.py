"""ProtoTorch competition functions."""

import torch


def stratified_sum(
        value: torch.Tensor,
        labels: torch.LongTensor) -> (torch.Tensor, torch.LongTensor):
    """Group-wise sum"""
    uniques = labels.unique(sorted=True).tolist()
    labels = labels.tolist()

    key_val = {key: val for key, val in zip(uniques, range(len(uniques)))}
    labels = torch.LongTensor(list(map(key_val.get, labels)))

    labels = labels.view(labels.size(0), 1).expand(-1, value.size(1))

    unique_labels = labels.unique(dim=0)
    result = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
        0, labels, value)
    return result.T


def stratified_min(distances, labels):
    """Group-wise minimum"""
    clabels = torch.unique(labels, dim=0)
    num_classes = clabels.size()[0]
    if distances.size()[1] == num_classes:
        # skip if only one prototype per class
        return distances
    batch_size = distances.size()[0]
    winning_distances = torch.zeros(num_classes, batch_size)
    inf = torch.full_like(distances.T, fill_value=float("inf"))
    # distances_to_wpluses = torch.where(matcher, distances, inf)
    for i, cl in enumerate(clabels):
        # cdists = distances.T[labels == cl]
        matcher = torch.eq(labels.unsqueeze(dim=1), cl)
        if labels.ndim == 2:
            # if the labels are one-hot vectors
            matcher = torch.eq(torch.sum(matcher, dim=-1), num_classes)
        cdists = torch.where(matcher, distances.T, inf).T
        winning_distances[i] = torch.min(cdists, dim=1,
                                         keepdim=True).values.squeeze()
    if labels.ndim == 2:
        # Transpose to return with `batch_size` first and
        # reverse the columns to fix the ordering of the classes
        return torch.flip(winning_distances.T, dims=(1, ))

    return winning_distances.T  # return with `batch_size` first


def wtac(distances, labels):
    winning_indices = torch.min(distances, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    return winning_labels


def knnc(distances, labels, k=1):
    winning_indices = torch.topk(-distances, k=k, dim=1).indices
    # winning_labels = torch.mode(labels[winning_indices].squeeze(),
    #                             dim=1).values
    winning_labels = torch.mode(labels[winning_indices], dim=1).values
    return winning_labels
