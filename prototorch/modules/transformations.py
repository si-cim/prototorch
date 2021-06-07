"""ProtoTorch Transformation Layers."""

import torch
from torch.nn.parameter import Parameter

from .initializers import MatrixInitializer


def _precheck_initializer(initializer):
    if not isinstance(initializer, MatrixInitializer):
        emsg = f"`initializer` has to be some subtype of " \
            f"{MatrixInitializer}. " \
            f"You have provided: {initializer=} instead."
        raise TypeError(emsg)


class Omega(torch.nn.Module):
    """The Omega mapping used in GMLVQ."""
    def __init__(self,
                 num_replicas=1,
                 input_dim=None,
                 latent_dim=None,
                 initializer=None,
                 *,
                 initialized_weights=None):
        super().__init__()

        if initialized_weights is not None:
            self._register_weights(initialized_weights)
        else:
            if num_replicas == 1:
                shape = (input_dim, latent_dim)
            else:
                shape = (num_replicas, input_dim, latent_dim)
            self._initialize_weights(shape, initializer)

    def _register_weights(self, weights):
        self.register_parameter("_omega", Parameter(weights))

    def _initialize_weights(self, shape, initializer):
        _precheck_initializer(initializer)
        _omega = initializer.generate(shape)
        self._register_weights(_omega)

    def forward(self):
        return self._omega

    def extra_repr(self):
        return f"(omega): (shape: {tuple(self._omega.shape)})"
