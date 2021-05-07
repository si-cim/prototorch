"""ProtoTorch datasets."""

from .abstract import NumpyDataset
from .spiral import Spiral
from .tecator import Tecator

__all__ = [
    "NumpyDataset",
    "Spiral",
    "Tecator",
]
