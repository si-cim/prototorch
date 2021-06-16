"""ProtoTorch Competition Modules."""

import torch
from prototorch.functions.competitions import knnc, wtac


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
