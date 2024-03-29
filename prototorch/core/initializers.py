"""ProtoTorch code initializers"""

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import (
    Callable,
    Type,
    Union,
)

import torch

from prototorch.utils import parse_data_arg, parse_distribution


# Components
class AbstractComponentsInitializer(ABC):
    """Abstract class for all components initializers."""
    ...


class LiteralCompInitializer(AbstractComponentsInitializer):
    """'Generate' the provided components.

    Use this to 'generate' pre-initialized components elsewhere.

    """

    def __init__(self, components):
        self.components = components

    def generate(self, num_components: int = 0):
        """Ignore `num_components` and simply return `self.components`."""
        provided_num_components = len(self.components)
        if provided_num_components != num_components:
            wmsg = f"The number of components ({provided_num_components}) " \
                f"provided to {self.__class__.__name__} " \
                f"does not match the expected number ({num_components})."
            warnings.warn(wmsg)
        if not isinstance(self.components, torch.Tensor):
            wmsg = f"Converting components to {torch.Tensor}..."
            warnings.warn(wmsg)
            self.components = torch.Tensor(self.components)
        return self.components


class ShapeAwareCompInitializer(AbstractComponentsInitializer):
    """Abstract class for all dimension-aware components initializers."""

    def __init__(self, shape: Union[Iterable, int]):
        if isinstance(shape, Iterable):
            self.component_shape = tuple(shape)
        else:
            self.component_shape = (shape, )

    @abstractmethod
    def generate(self, num_components: int):
        ...


class ZerosCompInitializer(ShapeAwareCompInitializer):
    """Generate zeros corresponding to the components shape."""

    def generate(self, num_components: int):
        components = torch.zeros((num_components, ) + self.component_shape)
        return components


class OnesCompInitializer(ShapeAwareCompInitializer):
    """Generate ones corresponding to the components shape."""

    def generate(self, num_components: int):
        components = torch.ones((num_components, ) + self.component_shape)
        return components


class FillValueCompInitializer(OnesCompInitializer):
    """Generate components with the provided `fill_value`."""

    def __init__(self, shape, fill_value: float = 1.0):
        super().__init__(shape)
        self.fill_value = fill_value

    def generate(self, num_components: int):
        ones = super().generate(num_components)
        components = ones.fill_(self.fill_value)
        return components


class UniformCompInitializer(OnesCompInitializer):
    """Generate components by sampling from a continuous uniform distribution."""

    def __init__(self, shape, minimum=0.0, maximum=1.0, scale=1.0):
        super().__init__(shape)
        self.minimum = minimum
        self.maximum = maximum
        self.scale = scale

    def generate(self, num_components: int):
        ones = super().generate(num_components)
        components = self.scale * ones.uniform_(self.minimum, self.maximum)
        return components


class RandomNormalCompInitializer(OnesCompInitializer):
    """Generate components by sampling from a standard normal distribution."""

    def __init__(self, shape, shift=0.0, scale=1.0):
        super().__init__(shape)
        self.shift = shift
        self.scale = scale

    def generate(self, num_components: int):
        ones = super().generate(num_components)
        components = self.scale * (torch.randn_like(ones) + self.shift)
        return components


class AbstractDataAwareCompInitializer(AbstractComponentsInitializer):
    """Abstract class for all data-aware components initializers.

    Components generated by data-aware components initializers inherit the shape
    of the provided data.

    `data` has to be a torch tensor.

    """

    def __init__(self,
                 data: torch.Tensor,
                 noise: float = 0.0,
                 transform: Callable = torch.nn.Identity()):
        self.data = data
        self.noise = noise
        self.transform = transform

    def generate_end_hook(self, samples):
        drift = torch.rand_like(samples) * self.noise
        components = self.transform(samples + drift)
        return components

    @abstractmethod
    def generate(self, num_components: int):
        ...
        return self.generate_end_hook(...)

    def __del__(self):
        del self.data


class DataAwareCompInitializer(AbstractDataAwareCompInitializer):
    """'Generate' the components from the provided data."""

    def generate(self, num_components: int = 0):
        """Ignore `num_components` and simply return transformed `self.data`."""
        components = self.generate_end_hook(self.data)
        return components


class SelectionCompInitializer(AbstractDataAwareCompInitializer):
    """Generate components by uniformly sampling from the provided data."""

    def generate(self, num_components: int):
        indices = torch.LongTensor(num_components).random_(0, len(self.data))
        samples = self.data[indices]
        components = self.generate_end_hook(samples)
        return components


class MeanCompInitializer(AbstractDataAwareCompInitializer):
    """Generate components by computing the mean of the provided data."""

    def generate(self, num_components: int):
        mean = self.data.mean(dim=0)
        repeat_dim = [num_components] + [1] * len(mean.shape)
        samples = mean.repeat(repeat_dim)
        components = self.generate_end_hook(samples)
        return components


class AbstractClassAwareCompInitializer(AbstractComponentsInitializer):
    """Abstract class for all class-aware components initializers.

    Components generated by class-aware components initializers inherit the shape
    of the provided data.

    `data` could be a torch Dataset or DataLoader or a list/tuple of data and
    target tensors.

    """

    def __init__(self,
                 data,
                 noise: float = 0.0,
                 transform: Callable = torch.nn.Identity()):
        self.data, self.targets = parse_data_arg(data)
        self.noise = noise
        self.transform = transform
        self.clabels = torch.unique(self.targets).int().tolist()
        self.num_classes = len(self.clabels)

    def generate_end_hook(self, samples):
        drift = torch.rand_like(samples) * self.noise
        components = self.transform(samples + drift)
        return components

    @abstractmethod
    def generate(self, distribution: Union[dict, list, tuple]):
        ...
        return self.generate_end_hook(...)

    def __del__(self):
        del self.data
        del self.targets


class ClassAwareCompInitializer(AbstractClassAwareCompInitializer):
    """'Generate' components from provided data and requested distribution."""

    def generate(self, distribution: Union[dict, list, tuple]):
        """Ignore `distribution` and simply return transformed `self.data`."""
        components = self.generate_end_hook(self.data)
        return components


class AbstractStratifiedCompInitializer(AbstractClassAwareCompInitializer):
    """Abstract class for all stratified components initializers."""

    @property
    @abstractmethod
    def subinit_type(self) -> Type[AbstractDataAwareCompInitializer]:
        ...

    def generate(self, distribution: Union[dict, list, tuple]):
        distribution = parse_distribution(distribution)
        components = torch.tensor([])
        for k, v in distribution.items():
            stratified_data = self.data[self.targets == k]
            if len(stratified_data) == 0:
                raise ValueError(f"No data available for class {k}.")
            initializer = self.subinit_type(
                stratified_data,
                noise=self.noise,
                transform=self.transform,
            )
            samples = initializer.generate(num_components=v)
            components = torch.cat([components, samples])
        return components


class StratifiedSelectionCompInitializer(AbstractStratifiedCompInitializer):
    """Generate components using stratified sampling from the provided data."""

    @property
    def subinit_type(self):
        return SelectionCompInitializer


class StratifiedMeanCompInitializer(AbstractStratifiedCompInitializer):
    """Generate components at stratified means of the provided data."""

    @property
    def subinit_type(self):
        return MeanCompInitializer


# Labels
class AbstractLabelsInitializer(ABC):
    """Abstract class for all labels initializers."""

    @abstractmethod
    def generate(self, distribution: Union[dict, list, tuple]):
        ...


class LiteralLabelsInitializer(AbstractLabelsInitializer):
    """'Generate' the provided labels.

    Use this to 'generate' pre-initialized labels elsewhere.

    """

    def __init__(self, labels):
        self.labels = labels

    def generate(self, distribution: Union[dict, list, tuple]):
        """Ignore `distribution` and simply return `self.labels`.

        Convert to long tensor, if necessary.
        """
        labels = self.labels
        if not isinstance(labels, torch.LongTensor):
            wmsg = f"Converting labels to {torch.LongTensor}..."
            warnings.warn(wmsg)
            labels = torch.LongTensor(labels)
        return labels


class DataAwareLabelsInitializer(AbstractLabelsInitializer):
    """'Generate' the labels from a torch Dataset."""

    def __init__(self, data):
        self.data, self.targets = parse_data_arg(data)

    def generate(self, distribution: Union[dict, list, tuple]):
        """Ignore `num_components` and simply return `self.targets`."""
        return self.targets


class LabelsInitializer(AbstractLabelsInitializer):
    """Generate labels from `distribution`."""

    def generate(self, distribution: Union[dict, list, tuple]):
        distribution = parse_distribution(distribution)
        labels_list = []
        for k, v in distribution.items():
            labels_list.extend([k] * v)
        labels = torch.LongTensor(labels_list)
        return labels


class OneHotLabelsInitializer(LabelsInitializer):
    """Generate one-hot-encoded labels from `distribution`."""

    def generate(self, distribution: Union[dict, list, tuple]):
        distribution = parse_distribution(distribution)
        num_classes = len(distribution.keys())
        # this breaks if class labels are not [0,...,nclasses]
        labels = torch.eye(num_classes)[super().generate(distribution)]
        return labels


# Reasonings
def compute_distribution_shape(distribution):
    distribution = parse_distribution(distribution)
    num_components = sum(distribution.values())
    num_classes = len(distribution.keys())
    return (num_components, num_classes, 2)


class AbstractReasoningsInitializer(ABC):
    """Abstract class for all reasonings initializers."""

    def __init__(self, components_first: bool = True):
        self.components_first = components_first

    def generate_end_hook(self, reasonings):
        if not self.components_first:
            reasonings = reasonings.permute(2, 1, 0)
        return reasonings

    @abstractmethod
    def generate(self, distribution: Union[dict, list, tuple]):
        ...
        return self.generate_end_hook(...)


class LiteralReasoningsInitializer(AbstractReasoningsInitializer):
    """'Generate' the provided reasonings.

    Use this to 'generate' pre-initialized reasonings elsewhere.

    """

    def __init__(self, reasonings, **kwargs):
        super().__init__(**kwargs)
        self.reasonings = reasonings

    def generate(self, distribution: Union[dict, list, tuple]):
        """Ignore `distributuion` and simply return self.reasonings."""
        reasonings = self.reasonings
        if not isinstance(reasonings, torch.Tensor):
            wmsg = f"Converting reasonings to {torch.Tensor}..."
            warnings.warn(wmsg)
            reasonings = torch.Tensor(reasonings)
        reasonings = self.generate_end_hook(reasonings)
        return reasonings


class ZerosReasoningsInitializer(AbstractReasoningsInitializer):
    """Reasonings are all initialized with zeros."""

    def generate(self, distribution: Union[dict, list, tuple]):
        shape = compute_distribution_shape(distribution)
        reasonings = torch.zeros(*shape)
        reasonings = self.generate_end_hook(reasonings)
        return reasonings


class OnesReasoningsInitializer(AbstractReasoningsInitializer):
    """Reasonings are all initialized with ones."""

    def generate(self, distribution: Union[dict, list, tuple]):
        shape = compute_distribution_shape(distribution)
        reasonings = torch.ones(*shape)
        reasonings = self.generate_end_hook(reasonings)
        return reasonings


class RandomReasoningsInitializer(AbstractReasoningsInitializer):
    """Reasonings are randomly initialized."""

    def __init__(self, minimum=0.4, maximum=0.6, **kwargs):
        super().__init__(**kwargs)
        self.minimum = minimum
        self.maximum = maximum

    def generate(self, distribution: Union[dict, list, tuple]):
        shape = compute_distribution_shape(distribution)
        reasonings = torch.ones(*shape).uniform_(self.minimum, self.maximum)
        reasonings = self.generate_end_hook(reasonings)
        return reasonings


class PurePositiveReasoningsInitializer(AbstractReasoningsInitializer):
    """Each component reasons positively for exactly one class."""

    def generate(self, distribution: Union[dict, list, tuple]):
        num_components, num_classes, _ = compute_distribution_shape(
            distribution)
        A = OneHotLabelsInitializer().generate(distribution)
        B = torch.zeros(num_components, num_classes)
        reasonings = torch.stack([A, B], dim=-1)
        reasonings = self.generate_end_hook(reasonings)
        return reasonings


# Transforms
class AbstractTransformInitializer(ABC):
    """Abstract class for all transform initializers."""
    ...


class AbstractLinearTransformInitializer(AbstractTransformInitializer):
    """Abstract class for all linear transform initializers."""

    def __init__(self, out_dim_first: bool = False):
        self.out_dim_first = out_dim_first

    def generate_end_hook(self, weights):
        if self.out_dim_first:
            weights = weights.permute(1, 0)
        return weights

    @abstractmethod
    def generate(self, in_dim: int, out_dim: int):
        ...
        return self.generate_end_hook(...)


class ZerosLinearTransformInitializer(AbstractLinearTransformInitializer):
    """Initialize a matrix with zeros."""

    def generate(self, in_dim: int, out_dim: int):
        weights = torch.zeros(in_dim, out_dim)
        return self.generate_end_hook(weights)


class OnesLinearTransformInitializer(AbstractLinearTransformInitializer):
    """Initialize a matrix with ones."""

    def generate(self, in_dim: int, out_dim: int):
        weights = torch.ones(in_dim, out_dim)
        return self.generate_end_hook(weights)


class RandomLinearTransformInitializer(AbstractLinearTransformInitializer):
    """Initialize a matrix with random values."""

    def generate(self, in_dim: int, out_dim: int):
        weights = torch.rand(in_dim, out_dim)
        return self.generate_end_hook(weights)


class EyeLinearTransformInitializer(AbstractLinearTransformInitializer):
    """Initialize a matrix with the largest possible identity matrix."""

    def generate(self, in_dim: int, out_dim: int):
        weights = torch.zeros(in_dim, out_dim)
        I = torch.eye(min(in_dim, out_dim))
        weights[:I.shape[0], :I.shape[1]] = I
        return self.generate_end_hook(weights)


class AbstractDataAwareLTInitializer(AbstractLinearTransformInitializer):
    """Abstract class for all data-aware linear transform initializers."""

    def __init__(self,
                 data: torch.Tensor,
                 noise: float = 0.0,
                 transform: Callable = torch.nn.Identity(),
                 out_dim_first: bool = False):
        super().__init__(out_dim_first)
        self.data = data
        self.noise = noise
        self.transform = transform

    def generate_end_hook(self, weights: torch.Tensor):
        drift = torch.rand_like(weights) * self.noise
        weights = self.transform(weights + drift)
        if self.out_dim_first:
            weights = weights.permute(1, 0)
        return weights


class PCALinearTransformInitializer(AbstractDataAwareLTInitializer):
    """Initialize a matrix with Eigenvectors from the data."""

    def generate(self, in_dim: int, out_dim: int):
        _, _, weights = torch.pca_lowrank(self.data, q=out_dim)
        return self.generate_end_hook(weights)


class LiteralLinearTransformInitializer(AbstractDataAwareLTInitializer):
    """'Generate' the provided weights."""

    def generate(self, in_dim: int, out_dim: int):
        return self.generate_end_hook(self.data)


# Aliases - Components
CACI = ClassAwareCompInitializer
DACI = DataAwareCompInitializer
FVCI = FillValueCompInitializer
LCI = LiteralCompInitializer
MCI = MeanCompInitializer
OCI = OnesCompInitializer
RNCI = RandomNormalCompInitializer
SCI = SelectionCompInitializer
SMCI = StratifiedMeanCompInitializer
SSCI = StratifiedSelectionCompInitializer
UCI = UniformCompInitializer
ZCI = ZerosCompInitializer

# Aliases - Labels
DLI = DataAwareLabelsInitializer
LI = LabelsInitializer
LLI = LiteralLabelsInitializer
OHLI = OneHotLabelsInitializer

# Aliases - Reasonings
LRI = LiteralReasoningsInitializer
ORI = OnesReasoningsInitializer
PPRI = PurePositiveReasoningsInitializer
RRI = RandomReasoningsInitializer
ZRI = ZerosReasoningsInitializer

# Aliases - Transforms
ELTI = Eye = EyeLinearTransformInitializer
OLTI = OnesLinearTransformInitializer
RLTI = RandomLinearTransformInitializer
ZLTI = ZerosLinearTransformInitializer
PCALTI = PCALinearTransformInitializer
LLTI = LiteralLinearTransformInitializer
