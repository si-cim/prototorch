"""ProtoTorch losses."""

import torch

from prototorch.functions.activations import get_activation
from prototorch.functions.losses import glvq_loss


class GLVQLoss(torch.nn.Module):
    def __init__(self, margin=0.0, squashing="identity", beta=10, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.squashing = get_activation(squashing)
        self.beta = torch.tensor(beta)

    def forward(self, outputs, targets):
        distances, plabels = outputs
        mu = glvq_loss(distances, targets, prototype_labels=plabels)
        batch_loss = self.squashing(mu + self.margin, beta=self.beta)
        return torch.sum(batch_loss, dim=0)


class NeuralGasEnergy(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, d):
        order = torch.argsort(d, dim=1)
        ranks = torch.argsort(order, dim=1)
        cost = torch.sum(self._nghood_fn(ranks, self.lm) * d)

        return cost, order

    def extra_repr(self):
        return f"lambda: {self.lm}"

    @staticmethod
    def _nghood_fn(rankings, lm):
        return torch.exp(-rankings / lm)
