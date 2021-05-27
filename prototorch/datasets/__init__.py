"""ProtoTorch datasets."""

from .abstract import NumpyDataset
from .iris import Iris
from .spiral import Spiral
from .tecator import Tecator

__all__ = ['Iris', 'Spiral', 'Tecator']
