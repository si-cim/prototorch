"""ProtoTorch Labels."""

import torch
from prototorch.components.components import get_labels_initializer
from prototorch.components.initializers import (ClassAwareInitializer,
                                                ComponentsInitializer,
                                                EqualLabelsInitializer,
                                                UnequalLabelsInitializer)
from torch.nn.parameter import Parameter


def get_labels_initializer(distribution):
    if isinstance(distribution, dict):
        if "num_classes" in distribution.keys():
            labels = EqualLabelsInitializer(
                distribution["num_classes"],
                distribution["prototypes_per_class"])
        else:
            clabels = list(distribution.keys())
            dist = list(distribution.values())
            labels = UnequalLabelsInitializer(dist, clabels)
    elif isinstance(distribution, tuple):
        num_classes, prototypes_per_class = distribution
        labels = EqualLabelsInitializer(num_classes, prototypes_per_class)
    elif isinstance(distribution, list):
        labels = UnequalLabelsInitializer(distribution)
    else:
        msg = f"`distribution` not understood." \
            f"You have provided: {distribution=}."
        raise ValueError(msg)
    return labels


class Labels(torch.nn.Module):
    def __init__(self,
                 distribution=None,
                 initializer=None,
                 *,
                 initialized_labels=None):
        _labels = self.get_labels(distribution,
                                  initializer,
                                  initialized_labels=initialized_labels)
        self._register_labels(_labels)

    def _register_labels(self, labels):
        # self.register_buffer("_labels", labels)
        self.register_parameter("_labels",
                                Parameter(labels, requires_grad=False))

    def get_labels(self,
                   distribution=None,
                   initializer=None,
                   *,
                   initialized_labels=None):
        if initialized_labels is not None:
            _labels = initialized_labels
        else:
            labels_initializer = initializer or get_labels_initializer(
                distribution)
            self.initial_distribution = labels_initializer.distribution
            _labels = labels_initializer.generate()
        return _labels

    def add_labels(self,
                   distribution=None,
                   initializer=None,
                   *,
                   initialized_labels=None):
        new_labels = self.get_labels(distribution,
                                     initializer,
                                     initialized_labels=initialized_labels)
        _labels = torch.cat([self._labels, new_labels])
        self._register_labels(_labels)

    def remove_labels(self, indices=None):
        mask = torch.ones(len(self._labels, dtype=torch.bool))
        mask[indices] = False
        _labels = self._labels[mask]
        self._register_labels(_labels)

    @property
    def labels(self):
        return self._labels

    def forward(self):
        return self._labels
