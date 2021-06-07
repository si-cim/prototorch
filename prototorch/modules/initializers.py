"""ProtoTroch Module Initializers."""

import torch


# Transformations
class MatrixInitializer(object):
    def __init__(self, *args, **kwargs):
        ...

    def generate(self, shape):
        raise NotImplementedError("Subclasses should implement this!")


class ZerosInitializer(MatrixInitializer):
    def generate(self, shape):
        return torch.zeros(shape)


class OnesInitializer(MatrixInitializer):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def generate(self, shape):
        return torch.ones(shape) * self.scale


class UniformInitializer(MatrixInitializer):
    def __init__(self, minimum=0.0, maximum=1.0, scale=1.0):
        super().__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.scale = scale

    def generate(self, shape):
        return torch.ones(shape).uniform_(self.minimum,
                                          self.maximum) * self.scale


class DataAwareInitializer(MatrixInitializer):
    def __init__(self, data, transform=torch.nn.Identity()):
        super().__init__()
        self.data = data
        self.transform = transform

    def __del__(self):
        del self.data


class EigenVectorInitializer(DataAwareInitializer):
    def generate(self, shape):
        # TODO
        raise NotImplementedError()


# Aliases
EV = EigenVectorInitializer
Random = RandomInitializer = UniformInitializer
Zeros = ZerosInitializer
Ones = OnesInitializer
