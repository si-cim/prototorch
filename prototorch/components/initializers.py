"""ProtoTroch Initializers."""
import warnings
from collections.abc import Iterable
from itertools import chain

import torch
from torch.utils.data import DataLoader, Dataset


def parse_data_arg(data_arg):
    if isinstance(data_arg, Dataset):
        data_arg = DataLoader(data_arg, batch_size=len(data_arg))

    if isinstance(data_arg, DataLoader):
        data = torch.tensor([])
        targets = torch.tensor([])
        for x, y in data_arg:
            data = torch.cat([data, x])
            targets = torch.cat([targets, y])
    else:
        data, targets = data_arg
        if not isinstance(data, torch.Tensor):
            wmsg = f"Converting data to {torch.Tensor}."
            warnings.warn(wmsg)
            data = torch.Tensor(data)
        if not isinstance(targets, torch.Tensor):
            wmsg = f"Converting targets to {torch.Tensor}."
            warnings.warn(wmsg)
            targets = torch.Tensor(targets)
    return data, targets


def get_subinitializers(data, targets, clabels, subinit_type):
    initializers = dict()
    for clabel in clabels:
        class_data = data[targets == clabel]
        class_initializer = subinit_type(class_data)
        initializers[clabel] = (class_initializer)
    return initializers


# Components
class ComponentsInitializer(object):
    def generate(self, number_of_components):
        raise NotImplementedError("Subclasses should implement this!")


class DimensionAwareInitializer(ComponentsInitializer):
    def __init__(self, dims):
        super().__init__()
        if isinstance(dims, Iterable):
            self.components_dims = tuple(dims)
        else:
            self.components_dims = (dims, )


class OnesInitializer(DimensionAwareInitializer):
    def __init__(self, dims, scale=1.0):
        super().__init__(dims)
        self.scale = scale

    def generate(self, length):
        gen_dims = (length, ) + self.components_dims
        return torch.ones(gen_dims) * self.scale


class ZerosInitializer(DimensionAwareInitializer):
    def generate(self, length):
        gen_dims = (length, ) + self.components_dims
        return torch.zeros(gen_dims)


class UniformInitializer(DimensionAwareInitializer):
    def __init__(self, dims, minimum=0.0, maximum=1.0, scale=1.0):
        super().__init__(dims)
        self.minimum = minimum
        self.maximum = maximum
        self.scale = scale

    def generate(self, length):
        gen_dims = (length, ) + self.components_dims
        return torch.ones(gen_dims).uniform_(self.minimum,
                                             self.maximum) * self.scale


class DataAwareInitializer(ComponentsInitializer):
    def __init__(self, data, transform=torch.nn.Identity()):
        super().__init__()
        self.data = data
        self.transform = transform

    def __del__(self):
        del self.data


class SelectionInitializer(DataAwareInitializer):
    def generate(self, length):
        indices = torch.LongTensor(length).random_(0, len(self.data))
        return self.transform(self.data[indices])


class MeanInitializer(DataAwareInitializer):
    def generate(self, length):
        mean = torch.mean(self.data, dim=0)
        repeat_dim = [length] + [1] * len(mean.shape)
        return self.transform(mean.repeat(repeat_dim))


class ClassAwareInitializer(DataAwareInitializer):
    def __init__(self, data, transform=torch.nn.Identity()):
        data, targets = parse_data_arg(data)
        super().__init__(data, transform)
        self.targets = targets
        self.clabels = torch.unique(self.targets).int().tolist()
        self.num_classes = len(self.clabels)

    def _get_samples_from_initializer(self, length, dist):
        if not dist:
            per_class = length // self.num_classes
            dist = dict(zip(self.clabels, self.num_classes * [per_class]))
        if isinstance(dist, list):
            dist = dict(zip(self.clabels, dist))
        samples = [self.initializers[k].generate(n) for k, n in dist.items()]
        out = torch.vstack(samples)
        with torch.no_grad():
            out = self.transform(out)
        return out

    def __del__(self):
        del self.data
        del self.targets


class StratifiedMeanInitializer(ClassAwareInitializer):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)
        self.initializers = get_subinitializers(self.data, self.targets,
                                                self.clabels, MeanInitializer)

    def generate(self, length, dist):
        samples = self._get_samples_from_initializer(length, dist)
        return samples


class StratifiedSelectionInitializer(ClassAwareInitializer):
    def __init__(self, data, noise=None, **kwargs):
        super().__init__(data, **kwargs)
        self.noise = noise
        self.initializers = get_subinitializers(self.data, self.targets,
                                                self.clabels,
                                                SelectionInitializer)

    def add_noise_v1(self, x):
        return x + self.noise

    def add_noise_v2(self, x):
        """Shifts some dimensions of the data randomly."""
        n1 = torch.rand_like(x)
        n2 = torch.rand_like(x)
        mask = torch.bernoulli(n1) - torch.bernoulli(n2)
        return x + (self.noise * mask)

    def generate(self, length, dist):
        samples = self._get_samples_from_initializer(length, dist)
        if self.noise is not None:
            samples = self.add_noise_v1(samples)
        return samples


# Labels
class LabelsInitializer:
    def generate(self):
        raise NotImplementedError("Subclasses should implement this!")


class UnequalLabelsInitializer(LabelsInitializer):
    def __init__(self, dist, clabels=None):
        self.dist = dist
        self.clabels = clabels or range(len(self.dist))

    @property
    def distribution(self):
        return self.dist

    def generate(self):
        targets = list(
            chain(*[[i] * n for i, n in zip(self.clabels, self.dist)]))
        return torch.LongTensor(targets)


class EqualLabelsInitializer(LabelsInitializer):
    def __init__(self, classes, per_class):
        self.classes = classes
        self.per_class = per_class

    @property
    def distribution(self):
        return self.classes * [self.per_class]

    def generate(self):
        return torch.arange(self.classes).repeat(self.per_class, 1).T.flatten()


# Reasonings
class ReasoningsInitializer:
    def generate(self, length):
        raise NotImplementedError("Subclasses should implement this!")


class ZeroReasoningsInitializer(ReasoningsInitializer):
    def __init__(self, classes, length):
        self.classes = classes
        self.length = length

    def generate(self):
        return torch.zeros((self.length, self.classes, 2))


# Aliases
SSI = StratifiedSampleInitializer = StratifiedSelectionInitializer
SMI = StratifiedMeanInitializer
Random = RandomInitializer = UniformInitializer
Zeros = ZerosInitializer
Ones = OnesInitializer
