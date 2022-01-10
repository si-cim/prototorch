"""Thin wrappers for a few scikit-learn datasets.

URL:
    https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

"""

import warnings
from typing import Sequence, Union

from sklearn.datasets import (
    load_iris,
    make_blobs,
    make_circles,
    make_classification,
    make_moons,
)

from prototorch.datasets.abstract import NumpyDataset


class Iris(NumpyDataset):
    """Iris Dataset by Ronald Fisher introduced in 1936.

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


class Blobs(NumpyDataset):
    """Generate isotropic Gaussian blobs for clustering.

    Read more at
    https://scikit-learn.org/stable/datasets/sample_generators.html#sample-generators.

    """

    def __init__(self,
                 num_samples: int = 300,
                 num_features: int = 2,
                 seed: Union[None, int] = 0):
        x, y = make_blobs(num_samples,
                          num_features,
                          centers=None,
                          random_state=seed,
                          shuffle=False)
        super().__init__(x, y)


class Random(NumpyDataset):
    """Generate a random n-class classification problem.

    Read more at
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html.

    Note: n_classes * n_clusters_per_class <= 2**n_informative must satisfy.
    """

    def __init__(self,
                 num_samples: int = 300,
                 num_features: int = 2,
                 num_classes: int = 2,
                 num_clusters: int = 2,
                 num_informative: Union[None, int] = None,
                 separation: float = 1.0,
                 seed: Union[None, int] = 0):
        if not num_informative:
            import math
            num_informative = math.ceil(math.log2(num_classes * num_clusters))
            if num_features < num_informative:
                warnings.warn("Generating more features than requested.")
                num_features = num_informative
        x, y = make_classification(num_samples,
                                   num_features,
                                   n_informative=num_informative,
                                   n_redundant=0,
                                   n_classes=num_classes,
                                   n_clusters_per_class=num_clusters,
                                   class_sep=separation,
                                   random_state=seed,
                                   shuffle=False)
        super().__init__(x, y)


class Circles(NumpyDataset):
    """Make a large circle containing a smaller circle in 2D.

    A simple toy dataset to visualize clustering and classification algorithms.

    Read more at
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html

    """

    def __init__(self,
                 num_samples: int = 300,
                 noise: float = 0.3,
                 factor: float = 0.8,
                 seed: Union[None, int] = 0):
        x, y = make_circles(num_samples,
                            noise=noise,
                            factor=factor,
                            random_state=seed,
                            shuffle=False)
        super().__init__(x, y)


class Moons(NumpyDataset):
    """Make two interleaving half circles.

    A simple toy dataset to visualize clustering and classification algorithms.

    Read more at
    https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html

    """

    def __init__(self,
                 num_samples: int = 300,
                 noise: float = 0.3,
                 seed: Union[None, int] = 0):
        x, y = make_moons(num_samples,
                          noise=noise,
                          random_state=seed,
                          shuffle=False)
        super().__init__(x, y)
