"""ProtoTorch components"""

import inspect
from typing import Union

import torch
from torch.nn.parameter import Parameter

from ..utils import parse_distribution
from .initializers import (
    AbstractClassAwareCompInitializer,
    AbstractComponentsInitializer,
    AbstractLabelsInitializer,
    AbstractReasoningsInitializer,
    LabelsInitializer,
    PurePositiveReasoningsInitializer,
    RandomReasoningsInitializer,
)


def validate_initializer(initializer, instanceof):
    """Check if the initializer is valid."""
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


def gencat(ins, attr, init, *iargs, **ikwargs):
    """Generate new items and concatenate with existing items."""
    new_items = init.generate(*iargs, **ikwargs)
    if hasattr(ins, attr):
        items = torch.cat([getattr(ins, attr), new_items])
    else:
        items = new_items
    return items, new_items


def removeind(ins, attr, indices):
    """Remove items at specified indices."""
    mask = torch.ones(len(ins), dtype=torch.bool)
    mask[indices] = False
    items = getattr(ins, attr)[mask]
    return items, mask


def get_cikwargs(init, distribution):
    """Return appropriate key-word arguments for a component initializer."""
    if isinstance(init, AbstractClassAwareCompInitializer):
        cikwargs = dict(distribution=distribution)
    else:
        distribution = parse_distribution(distribution)
        num_components = sum(distribution.values())
        cikwargs = dict(num_components=num_components)
    return cikwargs


class AbstractComponents(torch.nn.Module):
    """Abstract class for all components modules."""
    @property
    def num_components(self):
        """Current number of components."""
        return len(self._components)

    @property
    def components(self):
        """Detached Tensor containing the components."""
        return self._components.detach().cpu()

    def _register_components(self, components):
        self.register_parameter("_components", Parameter(components))

    def extra_repr(self):
        return f"components: (shape: {tuple(self._components.shape)})"

    def __len__(self):
        return self.num_components


class Components(AbstractComponents):
    """A set of adaptable Tensors."""
    def __init__(self, num_components: int,
                 initializer: AbstractComponentsInitializer, **kwargs):
        super().__init__(**kwargs)
        self.add_components(num_components, initializer)

    def add_components(self, num_components: int,
                       initializer: AbstractComponentsInitializer):
        """Generate and add new components."""
        assert validate_initializer(initializer, AbstractComponentsInitializer)
        _components, new_components = gencat(self, "_components", initializer,
                                             num_components)
        self._register_components(_components)
        return new_components

    def remove_components(self, indices):
        """Remove components at specified indices."""
        _components, mask = removeind(self, "_components", indices)
        self._register_components(_components)
        return mask

    def forward(self):
        """Simply return the components parameter Tensor."""
        return self._components


class AbstractLabels(torch.nn.Module):
    """Abstract class for all labels modules."""
    @property
    def labels(self):
        return self._labels.cpu()

    @property
    def num_labels(self):
        return len(self._labels)

    @property
    def unique_labels(self):
        return torch.unique(self._labels)

    @property
    def num_unique(self):
        return len(self.unique_labels)

    @property
    def distribution(self):
        unique, counts = torch.unique(self._labels,
                                      sorted=True,
                                      return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def _register_labels(self, labels):
        self.register_buffer("_labels", labels)

    def extra_repr(self):
        r = f"num_labels: {self.num_labels}, num_unique: {self.num_unique}"
        if len(self.distribution) < 11:  # avoid lengthy representations
            d = self.distribution
            unique, counts = list(d.keys()), list(d.values())
            r += f", unique: {unique}, counts: {counts}"
        return r

    def __len__(self):
        return self.num_labels


class Labels(AbstractLabels):
    """A set of standalone labels."""
    def __init__(self,
                 distribution: Union[dict, list, tuple],
                 initializer: AbstractLabelsInitializer = LabelsInitializer(),
                 **kwargs):
        super().__init__(**kwargs)
        self.add_labels(distribution, initializer)

    def add_labels(
        self,
        distribution: Union[dict, tuple, list],
        initializer: AbstractLabelsInitializer = LabelsInitializer()):
        """Generate and add new labels."""
        assert validate_initializer(initializer, AbstractLabelsInitializer)
        _labels, new_labels = gencat(self, "_labels", initializer,
                                     distribution)
        self._register_labels(_labels)
        return new_labels

    def remove_labels(self, indices):
        """Remove labels at specified indices."""
        _labels, mask = removeind(self, "_labels", indices)
        self._register_labels(_labels)
        return mask

    def forward(self):
        """Simply return the labels."""
        return self._labels


class LabeledComponents(AbstractComponents):
    """A set of adaptable components and corresponding unadaptable labels."""
    def __init__(
            self,
            distribution: Union[dict, list, tuple],
            components_initializer: AbstractComponentsInitializer,
            labels_initializer: AbstractLabelsInitializer = LabelsInitializer(
            ),
            **kwargs):
        super().__init__(**kwargs)
        self.add_components(distribution, components_initializer,
                            labels_initializer)

    @property
    def distribution(self):
        unique, counts = torch.unique(self._labels,
                                      sorted=True,
                                      return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    @property
    def num_classes(self):
        return len(self.distribution.keys())

    @property
    def labels(self):
        """Tensor containing the component labels."""
        return self._labels.cpu()

    def _register_labels(self, labels):
        self.register_buffer("_labels", labels)

    def add_components(
        self,
        distribution,
        components_initializer,
        labels_initializer: AbstractLabelsInitializer = LabelsInitializer()):
        """Generate and add new components and labels."""
        assert validate_initializer(components_initializer,
                                    AbstractComponentsInitializer)
        assert validate_initializer(labels_initializer,
                                    AbstractLabelsInitializer)
        cikwargs = get_cikwargs(components_initializer, distribution)
        _components, new_components = gencat(self, "_components",
                                             components_initializer,
                                             **cikwargs)
        _labels, new_labels = gencat(self, "_labels", labels_initializer,
                                     distribution)
        self._register_components(_components)
        self._register_labels(_labels)
        return new_components, new_labels

    def remove_components(self, indices):
        """Remove components and labels at specified indices."""
        _components, mask = removeind(self, "_components", indices)
        _labels, mask = removeind(self, "_labels", indices)
        self._register_components(_components)
        self._register_labels(_labels)
        return mask

    def forward(self):
        """Simply return the components parameter Tensor and labels."""
        return self._components, self._labels


class Reasonings(torch.nn.Module):
    """A set of standalone reasoning matrices.

    The `reasonings` tensor is of shape [num_components, num_classes, 2].

    """
    def __init__(self,
                 distribution: Union[dict, list, tuple],
                 initializer:
                 AbstractReasoningsInitializer = RandomReasoningsInitializer(),
                 **kwargs):
        super().__init__(**kwargs)

    @property
    def num_classes(self):
        return self._reasonings.shape[1]

    @property
    def reasonings(self):
        """Tensor containing the reasoning matrices."""
        return self._reasonings.detach().cpu()

    def _register_reasonings(self, reasonings):
        self.register_buffer("_reasonings", reasonings)

    def add_reasonings(
        self,
        distribution: Union[dict, list, tuple],
        initializer:
        AbstractReasoningsInitializer = RandomReasoningsInitializer()):
        """Generate and add new reasonings."""
        assert validate_initializer(initializer, AbstractReasoningsInitializer)
        _reasonings, new_reasonings = gencat(self, "_reasonings", initializer,
                                             distribution)
        self._register_reasonings(_reasonings)
        return new_reasonings

    def remove_reasonings(self, indices):
        """Remove reasonings at specified indices."""
        _reasonings, mask = removeind(self, "_reasonings", indices)
        self._register_reasonings(_reasonings)
        return mask

    def forward(self):
        """Simply return the reasonings."""
        return self._reasonings


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
    def __init__(
            self,
            distribution: Union[dict, list, tuple],
            components_initializer: AbstractComponentsInitializer,
            reasonings_initializer:
        AbstractReasoningsInitializer = PurePositiveReasoningsInitializer(),
            **kwargs):
        super().__init__(**kwargs)
        self.add_components(distribution, components_initializer,
                            reasonings_initializer)

    @property
    def num_classes(self):
        return self._reasonings.shape[1]

    @property
    def reasonings(self):
        """Tensor containing the reasoning matrices."""
        return self._reasonings.detach().cpu()

    @property
    def reasoning_matrices(self):
        """Reasoning matrices for each class."""
        with torch.no_grad():
            A, B = self._reasonings.permute(2, 1, 0).clamp(0, 1)
            pk = A
            nk = (1 - pk) * B
            ik = 1 - pk - nk
            matrices = torch.stack([pk, nk, ik], dim=-1).permute(1, 2, 0)
        return matrices.cpu()

    def _register_reasonings(self, reasonings):
        self.register_parameter("_reasonings", Parameter(reasonings))

    def add_components(self, distribution, components_initializer,
                       reasonings_initializer: AbstractReasoningsInitializer):
        """Generate and add new components and reasonings."""
        assert validate_initializer(components_initializer,
                                    AbstractComponentsInitializer)
        assert validate_initializer(reasonings_initializer,
                                    AbstractReasoningsInitializer)
        cikwargs = get_cikwargs(components_initializer, distribution)
        _components, new_components = gencat(self, "_components",
                                             components_initializer,
                                             **cikwargs)
        _reasonings, new_reasonings = gencat(self, "_reasonings",
                                             reasonings_initializer,
                                             distribution)
        self._register_components(_components)
        self._register_reasonings(_reasonings)
        return new_components, new_reasonings

    def remove_components(self, indices):
        """Remove components and reasonings at specified indices."""
        _components, mask = removeind(self, "_components", indices)
        _reasonings, mask = removeind(self, "_reasonings", indices)
        self._register_components(_components)
        self._register_reasonings(_reasonings)
        return mask

    def forward(self):
        """Simply return the components and reasonings."""
        return self._components, self._reasonings
