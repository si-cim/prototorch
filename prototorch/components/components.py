"""ProtoTorch components modules."""

import warnings

import torch
from prototorch.components.initializers import (ClassAwareInitializer,
                                                ComponentsInitializer,
                                                CustomLabelsInitializer,
                                                EqualLabelsInitializer,
                                                UnequalLabelsInitializer,
                                                ZeroReasoningsInitializer)
from torch.nn.parameter import Parameter


def get_labels_object(distribution):
    if isinstance(distribution, dict):
        if "num_classes" in distribution.keys():
            labels = EqualLabelsInitializer(
                distribution["num_classes"],
                distribution["prototypes_per_class"])
        else:
            labels = CustomLabelsInitializer(distribution)
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


class Components(torch.nn.Module):
    """Components is a set of learnable Tensors."""
    def __init__(self,
                 num_components=None,
                 initializer=None,
                 *,
                 initialized_components=None):
        super().__init__()

        # Ignore all initialization settings if initialized_components is given.
        if initialized_components is not None:
            self._register_components(initialized_components)
            if num_components is not None or initializer is not None:
                wmsg = "Arguments ignored while initializing Components"
                warnings.warn(wmsg)
        else:
            self._initialize_components(initializer, num_components)

    @property
    def num_components(self):
        # return len(self._components)
        return self._components.shape[0]

    def _register_components(self, components):
        self.register_parameter("_components", Parameter(components))

    def _precheck_initializer(self, initializer):
        if not isinstance(initializer, ComponentsInitializer):
            emsg = f"`initializer` has to be some subtype of " \
                f"{ComponentsInitializer}. " \
                f"You have provided: {initializer=} instead."
            raise TypeError(emsg)

    def _initialize_components(self, initializer, num_components):
        self._precheck_initializer(initializer)
        _components = initializer.generate(num_components)
        self._register_components(_components)

    def add_components(self,
                       initializer=None,
                       num=1,
                       *,
                       initialized_components=None):
        if initialized_components is not None:
            _components = torch.cat([self._components, initialized_components])
        else:
            self._precheck_initializer(initializer)
            _new = initializer.generate(num)
            _components = torch.cat([self._components, _new])
        self._register_components(_components)

    def remove_components(self, indices=None):
        mask = torch.ones(self.num_components, dtype=torch.bool)
        mask[indices] = False
        _components = self._components[mask]
        self._register_components(_components)
        return mask

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
            labels = get_labels_object(distribution)
            self.initial_distribution = labels.distribution
            _labels = labels.generate()
            super().__init__(len(_labels), initializer=initializer)
            self._register_labels(_labels)

    def _register_labels(self, labels):
        self.register_buffer("_labels", labels)

    @property
    def distribution(self):
        clabels, counts = torch.unique(self._labels, sorted=True, return_counts=True)
        return dict(zip(clabels.tolist(), counts.tolist()))

    def _initialize_components(self, initializer, num_components):
        if isinstance(initializer, ClassAwareInitializer):
            self._precheck_initializer(initializer)
            _components = initializer.generate(num_components,
                                               self.initial_distribution)
            self._register_components(_components)
        else:
            super()._initialize_components(initializer, num_components)

    def add_components(self, initializer, distribution):
        self._precheck_initializer(initializer)

        # Labels
        labels = get_labels_object(distribution)
        new_labels = labels.generate()
        _labels = torch.cat([self._labels, new_labels])
        self._register_labels(_labels)

        # Components
        if isinstance(initializer, ClassAwareInitializer):
            _new = initializer.generate(len(new_labels), labels.distribution)
        else:
            _new = initializer.generate(len(new_labels))
        _components = torch.cat([self._components, _new])
        self._register_components(_components)

    def remove_components(self, indices=None):
        # Components
        mask = super().remove_components(indices)

        # Labels
        _labels = self._labels[mask]
        self._register_labels(_labels)

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
        if isinstance(reasonings, tuple):
            num_classes, num_components = reasonings
            reasonings = ZeroReasoningsInitializer(num_classes, num_components)

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
