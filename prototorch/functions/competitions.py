"""ProtoTorch competition functions."""

import torch


def wtac(distances: torch.Tensor,
         labels: torch.LongTensor) -> (torch.LongTensor):
    """Winner-Takes-All-Competition.

    Returns the labels corresponding to the winners.

    """
    winning_indices = torch.min(distances, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    return winning_labels


def knnc(distances: torch.Tensor,
         labels: torch.LongTensor,
         k: int = 1) -> (torch.LongTensor):
    """K-Nearest-Neighbors-Competition.

    Returns the labels corresponding to the winners.

    """
    winning_indices = torch.topk(-distances, k=k, dim=1).indices
    winning_labels = torch.mode(labels[winning_indices], dim=1).values
    return winning_labels
