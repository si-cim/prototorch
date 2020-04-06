"""ProtoTorch competition functions."""

import torch


def wtac(distances, labels):
    winning_indices = torch.min(distances, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    return winning_labels


def knnc(distances, labels, k):
    winning_indices = torch.topk(-distances, k=k, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    return winning_labels
