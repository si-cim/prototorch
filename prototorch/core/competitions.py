"""ProtoTorch competitions"""

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


class WTAC(torch.nn.Module):
    """Winner-Takes-All-Competition Layer.

    Thin wrapper over the `wtac` function.

    """
    def forward(self, distances, labels):
        return wtac(distances, labels)


class LTAC(torch.nn.Module):
    """Loser-Takes-All-Competition Layer.

    Thin wrapper over the `wtac` function.

    """
    def forward(self, probs, labels):
        return wtac(-1.0 * probs, labels)


class KNNC(torch.nn.Module):
    """K-Nearest-Neighbors-Competition.

    Thin wrapper over the `knnc` function.

    """
    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def forward(self, distances, labels):
        return knnc(distances, labels, k=self.k)

    def extra_repr(self):
        return f"k: {self.k}"
