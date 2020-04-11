"""ProtoTorch prototype modules."""

import warnings

import torch

from prototorch.functions.initializers import get_initializer


class Prototypes1D(torch.nn.Module):
    def __init__(self,
                 prototypes_per_class=1,
                 prototype_distribution=None,
                 prototype_initializer='ones',
                 data=None,
                 dtype=torch.float32,
                 **kwargs):

        # Accept PyTorch tensors, but convert to python lists before processing
        if torch.is_tensor(prototype_distribution):
            prototype_distribution = prototype_distribution.tolist()

        if data is None:
            if 'input_dim' not in kwargs:
                raise NameError('`input_dim` required if '
                                'no `data` is provided.')
            if prototype_distribution:
                nclasses = sum(prototype_distribution)
            else:
                if 'nclasses' not in kwargs:
                    raise NameError('`prototype_distribution` required if '
                                    'both `data` and `nclasses` are not '
                                    'provided.')
                nclasses = kwargs.pop('nclasses')
            input_dim = kwargs.pop('input_dim')
            if prototype_initializer in [
                    'stratified_mean', 'stratified_random'
            ]:
                warnings.warn(
                    f'`prototype_initializer`: `{prototype_initializer}` '
                    'requires `data`, but `data` is not provided. '
                    'Using randomly generated data instead.')
            x_train = torch.rand(nclasses, input_dim)
            y_train = torch.arange(nclasses)
            data = [x_train, y_train]

        x_train, y_train = data
        x_train = torch.as_tensor(x_train).type(dtype)
        y_train = torch.as_tensor(y_train).type(dtype)
        nclasses = torch.unique(y_train).shape[0]

        assert x_train.ndim == 2

        # Verify input dimension if `input_dim` is provided
        if 'input_dim' in kwargs:
            assert kwargs.pop('input_dim') == x_train.shape[1]

        # Verify the number of classes if `nclasses` is provided
        if 'nclasses' in kwargs:
            assert nclasses == kwargs.pop('nclasses')

        super().__init__(**kwargs)

        if not prototype_distribution:
            prototype_distribution = [prototypes_per_class] * nclasses
        with torch.no_grad():
            self.prototype_distribution = torch.tensor(prototype_distribution)

        self.prototype_initializer = get_initializer(prototype_initializer)
        prototypes, prototype_labels = self.prototype_initializer(
            x_train,
            y_train,
            prototype_distribution=self.prototype_distribution)

        # Register module parameters
        self.prototypes = torch.nn.Parameter(prototypes)
        self.prototype_labels = prototype_labels

    def forward(self):
        return self.prototypes, self.prototype_labels
