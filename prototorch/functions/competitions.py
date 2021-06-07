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


def wtac_thresh(distances: torch.Tensor,
                labels: torch.LongTensor,
                theta_boundary: torch.Tensor) -> (torch.LongTensor):
    """ Used for OneClassClassifier.
    Calculates if distance is in between the Voronoi-cell of prototype or not. Voronoi-cell is defined by >theta_boundary<. (like a radius) """
    #in_boundary = (theta_boundary - distances)
    #winning_indices = torch.min(in_boundary, dim=1).indices
    winning_indices = torch.min(distances, dim=1).indices
    winning_labels = labels[winning_indices].squeeze()
    in_boundary = (theta_boundary - distances)
    in_boundary = in_boundary.gather(1, winning_indices.unsqueeze(1)).squeeze()
    winning_labels = torch.where(in_boundary > 0., winning_labels, torch.max(labels)+1) # '-1' -> 'garbage class'
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
