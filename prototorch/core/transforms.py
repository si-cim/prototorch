"""ProtoTorch transforms"""

import torch
from torch.nn.parameter import Parameter

from .initializers import (
    AbstractLinearTransformInitializer,
    EyeLinearTransformInitializer,
)


class LinearTransform(torch.nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        initializer:
        AbstractLinearTransformInitializer = EyeLinearTransformInitializer()):
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
        AbstractLinearTransformInitializer = EyeLinearTransformInitializer()):
        weights = initializer.generate(in_dim, out_dim)
        self._register_weights(weights)

    def forward(self, x):
        return x @ self._weights

    def extra_repr(self):
        return f"weights: (shape: {tuple(self._weights.shape)})"


# Aliases
Omega = LinearTransform
