"""Thin wrapper for the Iris classification dataset from sklearn.

URL:
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

"""

from prototorch.datasets.abstract import NumpyDataset
from sklearn.datasets import load_iris


class Iris(NumpyDataset):
    def __init__(self):
        x, y = load_iris(return_X_y=True)
        super().__init__(x, y)
