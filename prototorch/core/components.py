"""ProtoTorch components"""

import inspect
from typing import Union

import torch
from torch.nn.parameter import Parameter

from ..utils import parse_distribution
from .initializers import (
    AbstractComponentsInitializer,
    AbstractLabelsInitializer,
    AbstractReasoningsInitializer,
    ClassAwareCompInitializer,
    LabelsInitializer,
)


def validate_initializer(initializer, instanceof):
    if not isinstance(initializer, instanceof):
        emsg = f"`initializer` has to be an instance " \
            f"of some subtype of {instanceof}. " \
            f"You have provided: {initializer} instead. "
        helpmsg = ""
        if inspect.isclass(initializer):
            helpmsg = f"Perhaps you meant to say, {initializer.__name__}() " \
                f"with the brackets instead of just {initializer.__name__}?"
        raise TypeError(emsg + helpmsg)
    return True


def validate_components_initializer(initializer):
    return validate_initializer(initializer, AbstractComponentsInitializer)


def validate_labels_initializer(initializer):
    return validate_initializer(initializer, AbstractLabelsInitializer)


def validate_reasonings_initializer(initializer):
    return validate_initializer(initializer, AbstractReasoningsInitializer)


class AbstractComponents(torch.nn.Module):
    """Abstract class for all components modules."""
    @property
    def num_components(self):
        """Current number of components."""
        return len(self._components)

    @property
    def components(self):
        """Detached Tensor containing the components."""
        return self._components.detach()

    def _register_components(self, components):
        self.register_parameter("_components", Parameter(components))

    def extra_repr(self):
        return f"(components): (shape: {tuple(self._components.shape)})"


class Components(AbstractComponents):
    """A set of adaptable Tensors."""
    def __init__(self, num_components: int,
                 initializer: AbstractComponentsInitializer, **kwargs):
        super().__init__(**kwargs)
        self.add_components(num_components, initializer)

    def add_components(self, num: int,
                       initializer: AbstractComponentsInitializer):
        """Add new components."""
        assert validate_components_initializer(initializer)
        new_components = initializer.generate(num)
        # Register
        if hasattr(self, "_components"):
            _components = torch.cat([self._components, new_components])
        else:
            _components = new_components
        self._register_components(_components)
        return new_components

    def remove_components(self, indices):
        """Remove components at specified indices."""
        mask = torch.ones(self.num_components, dtype=torch.bool)
        mask[indices] = False
        _components = self._components[mask]
        self._register_components(_components)
        return mask

    def forward(self):
        """Simply return the components parameter Tensor."""
        return self._components


class LabeledComponents(AbstractComponents):
    """A set of adaptable components and corresponding unadaptable labels."""
    def __init__(self, distribution: Union[dict, list, tuple],
                 components_initializer: AbstractComponentsInitializer,
                 labels_initializer: AbstractLabelsInitializer, **kwargs):
        super().__init__(**kwargs)
        self.add_components(distribution, components_initializer,
                            labels_initializer)

    @property
    def component_labels(self):
        """Tensor containing the component tensors."""
        return self._labels.detach()

    def _register_labels(self, labels):
        self.register_buffer("_labels", labels)

    def add_components(
        self,
        distribution,
        components_initializer,
        labels_initializer: AbstractLabelsInitializer = LabelsInitializer()):
        # Checks
        assert validate_components_initializer(components_initializer)
        assert validate_labels_initializer(labels_initializer)

        distribution = parse_distribution(distribution)

        # Generate new components
        if isinstance(components_initializer, ClassAwareCompInitializer):
            new_components = components_initializer.generate(distribution)
        else:
            num_components = sum(distribution.values())
            new_components = components_initializer.generate(num_components)

        # Generate new labels
        new_labels = labels_initializer.generate(distribution)

        # Register
        if hasattr(self, "_components"):
            _components = torch.cat([self._components, new_components])
        else:
            _components = new_components
        if hasattr(self, "_labels"):
            _labels = torch.cat([self._labels, new_labels])
        else:
            _labels = new_labels
        self._register_components(_components)
        self._register_labels(_labels)

        return new_components, new_labels

    def remove_components(self, indices):
        """Remove components and labels at specified indices."""
        mask = torch.ones(self.num_components, dtype=torch.bool)
        mask[indices] = False
        _components = self._components[mask]
        _labels = self._labels[mask]
        self._register_components(_components)
        self._register_labels(_labels)
        return mask

    def forward(self):
        """Simply return the components parameter Tensor and labels."""
        return self._components, self._labels


class ReasoningComponents(AbstractComponents):
    """A set of components and a corresponding adapatable reasoning matrices.

    Every component has its own reasoning matrix.

    A reasoning matrix is an Nx2 matrix, where N is the number of classes. The
    first element is called positive reasoning :math:`p`, the second negative
    reasoning :math:`n`. A components can reason in favour (positive) of a
    class, against (negative) a class or not at all (neutral).

    It holds that :math:`0 \leq n \leq 1`, :math:`0 \leq p \leq 1` and :math:`0
    \leq n+p \leq 1`. Therefore :math:`n` and :math:`p` are two elements of a
    three element probability distribution.

    """
    def __init__(self, distribution: Union[dict, list, tuple],
                 components_initializer: AbstractComponentsInitializer,
                 reasonings_initializer: AbstractReasoningsInitializer,
                 **kwargs):
        super().__init__(**kwargs)
        self.add_components(distribution, components_initializer,
                            reasonings_initializer)

    @property
    def reasonings(self):
        """Returns Reasoning Matrix.

        Dimension NxCx2

        """
        return self._reasonings.detach()

    def _register_reasonings(self, reasonings):
        self.register_parameter("_reasonings", Parameter(reasonings))

    def add_components(self, distribution, components_initializer,
                       reasonings_initializer: AbstractReasoningsInitializer):
        # Checks
        assert validate_components_initializer(components_initializer)
        assert validate_reasonings_initializer(reasonings_initializer)

        distribution = parse_distribution(distribution)

        # Generate new components
        if isinstance(components_initializer, ClassAwareCompInitializer):
            new_components = components_initializer.generate(distribution)
        else:
            num_components = sum(distribution.values())
            new_components = components_initializer.generate(num_components)

        # Generate new reasonings
        new_reasonings = reasonings_initializer.generate(distribution)

        # Register
        if hasattr(self, "_components"):
            _components = torch.cat([self._components, new_components])
        else:
            _components = new_components
        if hasattr(self, "_reasonings"):
            _reasonings = torch.cat([self._reasonings, new_reasonings])
        else:
            _reasonings = new_reasonings
        self._register_components(_components)
        self._register_reasonings(_reasonings)

        return new_components, new_reasonings

    def remove_components(self, indices):
        """Remove components and labels at specified indices."""
        mask = torch.ones(self.num_components, dtype=torch.bool)
        mask[indices] = False
        _components = self._components[mask]
        # TODO
        # _reasonings = self._reasonings[mask]
        self._register_components(_components)
        # self._register_reasonings(_reasonings)
        return mask

    def forward(self):
        """Simply return the components and reasonings."""
        return self._components, self._reasonings
