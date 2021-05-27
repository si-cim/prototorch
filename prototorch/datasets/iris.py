"""Thin wrapper for the Iris classification dataset from sklearn.

URL:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

"""

from typing import Sequence

from prototorch.datasets.abstract import NumpyDataset
from sklearn.datasets import load_iris


class Iris(NumpyDataset):
    """
    Iris Dataset by Ronald Fisher introduced in 1936.

    The dataset contains four measurements from flowers of three species of iris.

    .. list-table:: Iris
        :header-rows: 1

        * - dimensions
          - classes
          - training size
          - validation size
          - test size
        * - 4
          - 3
          - 150
          - 0
          - 0

    :param dims: select a subset of dimensions
    """
    def __init__(self, dims: Sequence[int] = None):
        x, y = load_iris(return_X_y=True)
        if dims:
            x = x[:, dims]
        super().__init__(x, y)
