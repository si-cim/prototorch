"""ProtoTorch competition functions."""

import torch


# @torch.jit.script
def wtac(distances, labels):
    winning_indices = torch.min(distances, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    return winning_labels


# @torch.jit.script
def knnc(distances, labels, k):
    winning_indices = torch.topk(-distances, k=k.item(), dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    return winning_labels
