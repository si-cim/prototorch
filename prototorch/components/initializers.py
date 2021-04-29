import torch
from collections.abc import Iterable


# Components
class ComponentsInitializer:
    def generate(self, number_of_components):
        pass


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
        return torch.FloatTensor(gen_dims).uniform_(self.min, self.max)


class PositionAwareInitializer(ComponentsInitializer):
    def __init__(self, positions):
        super().__init__()
        self.data = positions


class SelectionInitializer(PositionAwareInitializer):
    def generate(self, length):
        indices = torch.LongTensor(length).random_(0, len(self.data))
        return self.data[indices]


class MeanInitializer(PositionAwareInitializer):
    def generate(self, length):
        mean = torch.mean(self.data, dim=0)
        repeat_dim = [length] + [1] * len(mean.shape)
        return mean.repeat(repeat_dim)


class ClassAwareInitializer(ComponentsInitializer):
    def __init__(self, positions, classes):
        super().__init__()
        self.data = positions
        self.classes = classes

        self.names = torch.unique(self.classes)
        self.num_classes = len(self.names)


class StratifiedMeanInitializer(ClassAwareInitializer):
    def __init__(self, positions, classes):
        super().__init__(positions, classes)

        self.initializers = []
        for name in self.names:
            class_data = self.data[self.classes == name]
            class_initializer = MeanInitializer(class_data)
            self.initializers.append(class_initializer)

    def generate(self, length):
        per_class = length // self.num_classes
        return torch.vstack(
            [init.generate(per_class) for init in self.initializers])


class StratifiedSelectionInitializer(ClassAwareInitializer):
    def __init__(self, positions, classes):
        super().__init__(positions, classes)

        self.initializers = []
        for name in self.names:
            class_data = self.data[self.classes == name]
            class_initializer = SelectionInitializer(class_data)
            self.initializers.append(class_initializer)

    def generate(self, length):
        per_class = length // self.num_classes
        return torch.vstack(
            [init.generate(per_class) for init in self.initializers])


# Labels
class LabelsInitializer:
    def generate(self):
        pass


class EqualLabelInitializer(LabelsInitializer):
    def __init__(self, classes, per_class):
        self.classes = classes
        self.per_class = per_class

    def generate(self):
        return torch.arange(self.classes).repeat(self.per_class, 1).T.flatten()


# Reasonings
class ReasoningsInitializer:
    def generate(self, length):
        pass


class ZeroReasoningsInitializer(ReasoningsInitializer):
    def __init__(self, classes, length):
        self.classes = classes
        self.length = length

    def generate(self):
        return torch.zeros((self.length, self.classes, 2))
