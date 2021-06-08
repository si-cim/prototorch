import torch


# Functions
def gaussian(distances, variance):
    return torch.exp(-(distances * distances) / (2 * variance))


def rank_scaled_gaussian(distances, lambd):
    order = torch.argsort(distances, dim=1)
    ranks = torch.argsort(order, dim=1)

    return torch.exp(-torch.exp(-ranks / lambd) * distances)


# Modules
class GaussianPrior(torch.nn.Module):
    def __init__(self, variance):
        super().__init__()
        self.variance = variance

    def forward(self, distances):
        return gaussian(distances, self.variance)


class RankScaledGaussianPrior(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, distances):
        return rank_scaled_gaussian(distances, self.lambd)
