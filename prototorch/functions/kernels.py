"""
Experimental Kernels
"""

import torch


class ExplicitKernel:
    def __init__(self, projection=torch.nn.Identity()):
        self.projection = projection

    def __call__(self, x, y):
        return self.projection(x) @ self.projection(y).T


class RadialBasisFunctionKernel:
    def __init__(self, sigma) -> None:
        self.s2 = sigma * sigma

    def __call__(self, x, y):
        return torch.exp(-torch.sum((x - y)**2) / (2 * self.s2))
