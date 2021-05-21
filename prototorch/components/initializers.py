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
        labels = torch.tensor([])
        for x, y in data_arg:
            data = torch.cat([data, x])
            labels = torch.cat([labels, y])
    else:
        data, labels = data_arg
        if not isinstance(data, torch.Tensor):
            wmsg = f"Converting data to {torch.Tensor}."
            warnings.warn(wmsg)
            data = torch.Tensor(data)
        if not isinstance(labels, torch.Tensor):
            wmsg = f"Converting labels to {torch.Tensor}."
            warnings.warn(wmsg)
            labels = torch.Tensor(labels)
    return data, labels


# Components
class ComponentsInitializer(object):
    def generate(self, number_of_components):
        raise NotImplementedError("Subclasses should implement this!")


class DimensionAwareInitializer(ComponentsInitializer):
    def __init__(self, c_dims):
        super().__init__()
        if isinstance(c_dims, Iterable):
            self.components_dims = tuple(c_dims)
        else:
            self.components_dims = (c_dims, )


class OnesInitializer(DimensionAwareInitializer):
    def generate(self, length):
        gen_dims = (length, ) + self.components_dims
        return torch.ones(gen_dims)


class ZerosInitializer(DimensionAwareInitializer):
    def generate(self, length):
        gen_dims = (length, ) + self.components_dims
        return torch.zeros(gen_dims)


class UniformInitializer(DimensionAwareInitializer):
    def __init__(self, c_dims, min=0.0, max=1.0):
        super().__init__(c_dims)

        self.min = min
        self.max = max

    def generate(self, length):
        gen_dims = (length, ) + self.components_dims
        return torch.ones(gen_dims).uniform_(self.min, self.max)


class DataAwareInitializer(ComponentsInitializer):
    def __init__(self, data):
        super().__init__()
        self.data = data


class SelectionInitializer(DataAwareInitializer):
    def generate(self, length):
        indices = torch.LongTensor(length).random_(0, len(self.data))
        return self.data[indices]


class MeanInitializer(DataAwareInitializer):
    def generate(self, length):
        mean = torch.mean(self.data, dim=0)
        repeat_dim = [length] + [1] * len(mean.shape)
        return mean.repeat(repeat_dim)


class ClassAwareInitializer(ComponentsInitializer):
    def __init__(self, data, transform=torch.nn.Identity()):
        super().__init__()
        data, labels = parse_data_arg(data)
        self.data = data
        self.labels = labels

        self.transform = transform

        self.clabels = torch.unique(self.labels)
        self.num_classes = len(self.clabels)

    def _get_samples_from_initializer(self, length, dist):
        if not dist:
            per_class = length // self.num_classes
            dist = self.num_classes * [per_class]
        samples_list = [
            init.generate(n) for init, n in zip(self.initializers, dist)
        ]
        out = torch.vstack(samples_list)
        with torch.no_grad():
            out = self.transform(out)
        return out

    def __del__(self):
        del self.data
        del self.labels


class StratifiedMeanInitializer(ClassAwareInitializer):
    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

        self.initializers = []
        for clabel in self.clabels:
            class_data = self.data[self.labels == clabel]
            class_initializer = MeanInitializer(class_data)
            self.initializers.append(class_initializer)

    def generate(self, length, dist=[]):
        samples = self._get_samples_from_initializer(length, dist)
        return samples


class StratifiedSelectionInitializer(ClassAwareInitializer):
    def __init__(self, data, noise=None, **kwargs):
        super().__init__(data, **kwargs)
        self.noise = noise

        self.initializers = []
        for clabel in self.clabels:
            class_data = self.data[self.labels == clabel]
            class_initializer = SelectionInitializer(class_data)
            self.initializers.append(class_initializer)

    def add_noise_v1(self, x):
        return x + self.noise

    def add_noise_v2(self, x):
        """Shifts some dimensions of the data randomly."""
        n1 = torch.rand_like(x)
        n2 = torch.rand_like(x)
        mask = torch.bernoulli(n1) - torch.bernoulli(n2)
        return x + (self.noise * mask)

    def generate(self, length, dist=[]):
        samples = self._get_samples_from_initializer(length, dist)
        if self.noise is not None:
            samples = self.add_noise_v1(samples)
        return samples


# Labels
class LabelsInitializer:
    def generate(self):
        raise NotImplementedError("Subclasses should implement this!")


class UnequalLabelsInitializer(LabelsInitializer):
    def __init__(self, dist):
        self.dist = dist

    @property
    def distribution(self):
        return self.dist

    def generate(self):
        clabels = range(len(self.dist))
        labels = list(chain(*[[i] * n for i, n in zip(clabels, self.dist)]))
        return torch.tensor(labels)


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
