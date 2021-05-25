"""ProtoTorch components modules."""

import warnings
from typing import Tuple

import torch
from prototorch.components.initializers import (ClassAwareInitializer,
                                                ComponentsInitializer,
                                                EqualLabelsInitializer,
                                                UnequalLabelsInitializer,
                                                ZeroReasoningsInitializer)
from prototorch.functions.initializers import get_initializer
from torch.nn.parameter import Parameter


class Components(torch.nn.Module):
    """Components is a set of learnable Tensors."""
    def __init__(self,
                 ncomps=None,
                 initializer=None,
                 *,
                 initialized_components=None):
        super().__init__()

        self.ncomps = ncomps

        # Ignore all initialization settings if initialized_components is given.
        if initialized_components is not None:
            self.register_parameter("_components",
                                    Parameter(initialized_components))
            if ncomps is not None or initializer is not None:
                wmsg = "Arguments ignored while initializing Components"
                warnings.warn(wmsg)
        else:
            self._initialize_components(initializer)

    def _precheck_initializer(self, initializer):
        if not isinstance(initializer, ComponentsInitializer):
            emsg = f"`initializer` has to be some subtype of " \
                f"{ComponentsInitializer}. " \
                f"You have provided: {initializer=} instead."
            raise TypeError(emsg)

    def _initialize_components(self, initializer):
        self._precheck_initializer(initializer)
        _components = initializer.generate(self.ncomps)
        self.register_parameter("_components", Parameter(_components))

    @property
    def components(self):
        """Tensor containing the component tensors."""
        return self._components.detach()

    def forward(self):
        return self._components

    def extra_repr(self):
        return f"components.shape: {tuple(self._components.shape)}"


class LabeledComponents(Components):
    """LabeledComponents generate a set of components and a set of labels.

    Every Component has a label assigned.
    """
    def __init__(self,
                 distribution=None,
                 initializer=None,
                 *,
                 initialized_components=None):
        if initialized_components is not None:
            components, component_labels = initialized_components
            super().__init__(initialized_components=components)
            self._labels = component_labels
        else:
            _labels = self._initialize_labels(distribution)
            super().__init__(len(_labels), initializer=initializer)
            self.register_buffer("_labels", _labels)

    def _initialize_components(self, initializer):
        if isinstance(initializer, ClassAwareInitializer):
            self._precheck_initializer(initializer)
            _components = initializer.generate(self.ncomps, self.distribution)
            self.register_parameter("_components", Parameter(_components))
        else:
            super()._initialize_components(initializer)

    def _initialize_labels(self, distribution):
        if type(distribution) == dict:
            labels = EqualLabelsInitializer(
                distribution["num_classes"],
                distribution["prototypes_per_class"])
        elif type(distribution) == tuple:
            num_classes, prototypes_per_class = distribution
            labels = EqualLabelsInitializer(num_classes, prototypes_per_class)
        elif type(distribution) == list:
            labels = UnequalLabelsInitializer(distribution)

        self.distribution = labels.distribution
        return labels.generate()

    @property
    def component_labels(self):
        """Tensor containing the component tensors."""
        return self._labels.detach()

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
            components, reasonings = initialized_components

            super().__init__(initialized_components=components)
            self.register_parameter("_reasonings", reasonings)
        else:
            self._initialize_reasonings(reasonings)
            super().__init__(len(self._reasonings), initializer=initializer)

    def _initialize_reasonings(self, reasonings):
        if type(reasonings) == tuple:
            num_classes, ncomps = reasonings
            reasonings = ZeroReasoningsInitializer(num_classes, ncomps)

        _reasonings = reasonings.generate()
        self.register_parameter("_reasonings", _reasonings)

    @property
    def reasonings(self):
        """Returns Reasoning Matrix.

        Dimension NxCx2

        """
        return self._reasonings.detach()

    def forward(self):
        return super().forward(), self._reasonings
