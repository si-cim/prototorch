"""ProtoTorch components modules."""

import warnings
from typing import Tuple

import torch
from prototorch.components.initializers import (ComponentsInitializer,
                                                EqualLabelInitializer,
                                                ZeroReasoningsInitializer)
from prototorch.functions.initializers import get_initializer
from torch.nn.parameter import Parameter


class Components(torch.nn.Module):
    """Components is a set of learnable Tensors."""
    def __init__(self,
                 number_of_components=None,
                 initializer=None,
                 *,
                 initialized_components=None,
                 dtype=torch.float32):
        super().__init__()

        # Ignore all initialization settings if initialized_components is given.
        if initialized_components is not None:
            self._components = Parameter(initialized_components)
            if number_of_components is not None or initializer is not None:
                wmsg = "Arguments ignored while initializing Components"
                warnings.warn(wmsg)
        else:
            self._initialize_components(number_of_components, initializer)

    def _initialize_components(self, number_of_components, initializer):
        if not isinstance(initializer, ComponentsInitializer):
            emsg = f"`initializer` has to be some subtype of " \
                f"{ComponentsInitializer}. " \
                f"You have provided: {initializer=} instead."
            raise TypeError(emsg)
        self._components = Parameter(
            initializer.generate(number_of_components))

    @property
    def components(self):
        """Tensor containing the component tensors."""
        return self._components.detach().cpu()

    def forward(self):
        return self._components

    def extra_repr(self):
        return f"components.shape: {tuple(self._components.shape)}"


class LabeledComponents(Components):
    """LabeledComponents generate a set of components and a set of labels.

    Every Component has a label assigned.
    """
    def __init__(self,
                 labels=None,
                 initializer=None,
                 *,
                 initialized_components=None):
        if initialized_components is not None:
            super().__init__(initialized_components=initialized_components[0])
            self._labels = initialized_components[1]
        else:
            self._initialize_labels(labels)
            super().__init__(number_of_components=len(self._labels),
                             initializer=initializer)

    def _initialize_labels(self, labels):
        if type(labels) == tuple:
            num_classes, prototypes_per_class = labels
            labels = EqualLabelInitializer(num_classes, prototypes_per_class)

        self._labels = labels.generate()

    @property
    def component_labels(self):
        """Tensor containing the component tensors."""
        return self._labels.detach().cpu()

    def forward(self):
        return super().forward(), self._labels


class ReasoningComponents(Components):
    """ReasoningComponents generate a set of components and a set of reasoning matrices.

    Every Component has a reasoning matrix assigned.

    A reasoning matrix is a Nx2 matrix, where N is the number of Classes. The
    first element is called positive reasoning :math:`p`, the second negative
    reasoning :math:`n`. A components can reason in favour (positive) of a
    class, against (negative) a class or not at all (neutral).

    It holds that :math:`0 \leq n \leq 1`, :math:`0 \leq p \leq 1` and :math:`0
    \leq n+p \leq 1`. Therefore :math:`n` and :math:`p` are two elements of a
    three element probability distribution.

    """
    def __init__(self,
                 reasonings=None,
                 initializer=None,
                 *,
                 initialized_components=None):
        if initialized_components is not None:
            super().__init__(initialized_components=initialized_components[0])
            self._reasonings = initialized_components[1]
        else:
            self._initialize_reasonings(reasonings)
            super().__init__(number_of_components=len(self._reasonings),
                             initializer=initializer)

    def _initialize_reasonings(self, reasonings):
        if type(reasonings) == tuple:
            num_classes, number_of_components = reasonings
            reasonings = ZeroReasoningsInitializer(num_classes,
                                                   number_of_components)

        self._reasonings = reasonings.generate()

    @property
    def reasonings(self):
        """Returns Reasoning Matrix.

        Dimension NxCx2

        """
        return self._reasonings.detach().cpu()

    def forward(self):
        return super().forward(), self._reasonings
