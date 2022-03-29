"""ProtoTorch transforms"""

import torch
from torch.nn.parameter import Parameter

from .initializers import (
    AbstractLinearTransformInitializer,
    EyeTransformInitializer,
)


class LinearTransform(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        initializer:
        AbstractLinearTransformInitializer = EyeTransformInitializer()):
        super().__init__()
        self.set_weights(in_dim, out_dim, initializer)

    @property
    def weights(self):
        return self._weights.detach().cpu()

    def _register_weights(self, weights):
        self.register_parameter("_weights", Parameter(weights))

    def set_weights(
        self,
        in_dim: int,
        out_dim: int,
        initializer:
        AbstractLinearTransformInitializer = EyeTransformInitializer()):
        weights = initializer.generate(in_dim, out_dim)
        self._register_weights(weights)

    def forward(self, x):
        return x @ self._weights


# Aliases
Omega = LinearTransform
