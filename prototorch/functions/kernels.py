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
        remove_dim = False
        if len(x.shape) > 1:
            x = x.unsqueeze(1)
            remove_dim = True
        output = torch.exp(-torch.sum((x - y)**2, dim=-1) / (2 * self.s2))
        if remove_dim:
            output = output.squeeze(1)
        return output
