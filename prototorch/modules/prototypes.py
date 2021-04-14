"""ProtoTorch prototype modules."""

import warnings

import torch

from prototorch.functions.initializers import get_initializer


class _Prototypes(torch.nn.Module):
    """Abstract prototypes class."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _validate_prototype_distribution(self):
        if 0 in self.prototype_distribution:
            warnings.warn('Are you sure about the `0` in '
                          '`prototype_distribution`?')

    def extra_repr(self):
        return f'prototypes.shape: {tuple(self.prototypes.shape)}'

    def forward(self):
        return self.prototypes, self.prototype_labels


class Prototypes1D(_Prototypes):
    """Create a learnable set of one-dimensional prototypes.

    TODO Complete this doc-string.
    """
    def __init__(self,
                 prototypes_per_class=1,
                 prototype_initializer='ones',
                 prototype_distribution=None,
                 data=None,
                 dtype=torch.float32,
                 one_hot_labels=False,
                 **kwargs):

        # Convert tensors to python lists before processing
        if prototype_distribution is not None:
            if not isinstance(prototype_distribution, list):
                prototype_distribution = prototype_distribution.tolist()

        if data is None:
            if 'input_dim' not in kwargs:
                raise NameError('`input_dim` required if '
                                'no `data` is provided.')
            if prototype_distribution:
                kwargs_nclasses = sum(prototype_distribution)
            else:
                if 'nclasses' not in kwargs:
                    raise NameError('`prototype_distribution` required if '
                                    'both `data` and `nclasses` are not '
                                    'provided.')
                kwargs_nclasses = kwargs.pop('nclasses')
            input_dim = kwargs.pop('input_dim')
            if prototype_initializer in [
                    'stratified_mean', 'stratified_random'
            ]:
                warnings.warn(
                    f'`prototype_initializer`: `{prototype_initializer}` '
                    'requires `data`, but `data` is not provided. '
                    'Using randomly generated data instead.')
            x_train = torch.rand(kwargs_nclasses, input_dim)
            y_train = torch.arange(kwargs_nclasses)
            if one_hot_labels:
                y_train = torch.eye(kwargs_nclasses)[y_train]
            data = [x_train, y_train]

        x_train, y_train = data
        x_train = torch.as_tensor(x_train).type(dtype)
        y_train = torch.as_tensor(y_train).type(torch.int)
        nclasses = torch.unique(y_train, dim=-1).shape[-1]

        if nclasses == 1:
            warnings.warn('Are you sure about having one class only?')

        if x_train.ndim != 2:
            raise ValueError('`data[0].ndim != 2`.')

        if y_train.ndim == 2:
            if y_train.shape[1] == 1 and one_hot_labels:
                raise ValueError('`one_hot_labels` is set to `True` '
                                 'but target labels are not one-hot-encoded.')
            if y_train.shape[1] != 1 and not one_hot_labels:
                raise ValueError('`one_hot_labels` is set to `False` '
                                 'but target labels in `data` '
                                 'are one-hot-encoded.')
        if y_train.ndim == 1 and one_hot_labels:
            raise ValueError('`one_hot_labels` is set to `True` '
                             'but target labels are not one-hot-encoded.')

        # Verify input dimension if `input_dim` is provided
        if 'input_dim' in kwargs:
            input_dim = kwargs.pop('input_dim')
            if input_dim != x_train.shape[1]:
                raise ValueError(f'Provided `input_dim`={input_dim} does '
                                 'not match data dimension '
                                 f'`data[0].shape[1]`={x_train.shape[1]}')

        # Verify the number of classes if `nclasses` is provided
        if 'nclasses' in kwargs:
            kwargs_nclasses = kwargs.pop('nclasses')
            if kwargs_nclasses != nclasses:
                raise ValueError(f'Provided `nclasses={kwargs_nclasses}` does '
                                 'not match data labels '
                                 '`torch.unique(data[1]).shape[0]`'
                                 f'={nclasses}')

        super().__init__(**kwargs)

        if not prototype_distribution:
            prototype_distribution = [prototypes_per_class] * nclasses
        with torch.no_grad():
            self.prototype_distribution = torch.tensor(prototype_distribution)

        self._validate_prototype_distribution()

        self.prototype_initializer = get_initializer(prototype_initializer)
        prototypes, prototype_labels = self.prototype_initializer(
            x_train,
            y_train,
            prototype_distribution=self.prototype_distribution,
            one_hot=one_hot_labels,
        )

        # Register module parameters
        self.prototypes = torch.nn.Parameter(prototypes)
        self.prototype_labels = torch.nn.Parameter(
            prototype_labels.type(dtype)).requires_grad_(False)
