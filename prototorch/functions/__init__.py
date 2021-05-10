"""ProtoTorch functions."""

from .activations import identity, sigmoid_beta, swish_beta
from .competitions import knnc, wtac

__all__ = [
    "identity",
    "sigmoid_beta",
    "swish_beta",
    "knnc",
    "wtac",
]
